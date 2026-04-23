"""
collect_cot.py — Génère un dataset JSONL de CoT réussies.

Supports multiple datasets via --dataset:
  gsm8k           — numeric math (default)
  winogrande      — coreference resolution (A/B choice)
  triviaqa        — open-domain QA (string match)
  commonsenseqa   — common sense reasoning (A-E choice)
  local_math       — JSONL local {query, ground_truth} (--local_file requis)
  piqa             — physical intuition QA (A/B choice)
  mmlu             — MMLU train, 4 choices A-D
  mmlu_pro         — MMLU-Pro train, up to 10 choices A-J
  gpqa_diamond     — graduate-level science MCQ (A-D choice)
  hello            — greetings + identity questions (50/50), LLM-judged
  creative_writing — LLM-generated prompts, LLM-judged responses

Pour chaque exemple :
  - Génère G completions
  - Si ≥1 correcte → garde toutes les correctes
  - Si 0 correcte  → garde les partielles (dataset-dependent)
  - Calcule reward et advantage dans le groupe retenu
  - Écrit dans le JSONL

Usage:
    python collect_cot.py --dataset gsm8k        # → data/gsm8k_cot.jsonl
    python collect_cot.py --dataset winogrande  # → data/winogrande_cot.jsonl
    python collect_cot.py --dataset triviaqa    # → data/triviaqa_cot.jsonl
"""
import argparse, json, os, random, re, string, time
import numpy as np

from seqcond.torch.generator import TorchGenerator
from train_grpo import load_gsm8k, check_answer, _compute_advantages


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_answer_after_thinking(text):
    """Text after the last <|think_end|>, or full text if absent."""
    if "<|think_end|>" in text:
        text = text.split("<|think_end|>")[-1]
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    return text.strip()


def _parse_choice(answer_text, valid_choices, options=None):
    """Parse a choice letter (A/B/C/D) from model answer text."""
    letters = "".join(sorted(valid_choices))
    answer_text = answer_text.strip()
    if "\\boxed{" in answer_text:
        choice = answer_text.split("\\boxed{")[-1].split("}")[0]
        if len(choice) == 1 and choice.upper() in valid_choices:
            return choice.upper()
    if not answer_text:
        return None
    if len(answer_text) == 1 and answer_text.upper() in valid_choices:
        return answer_text.upper()
    tail = answer_text[-400:] if len(answer_text) > 400 else answer_text
    lc = f"[{letters}{letters.lower()}]"
    for pattern in [
        rf"\bfinal\s+answer\b\s*[:\-]?\s*({lc})(?:\.|\b)",
        rf"\banswer\b\s*(?:is\s*)?[:\-]?\s*({lc})(?:\.|\b)",
    ]:
        matches = re.findall(pattern, tail, re.IGNORECASE)
        if matches:
            letter = matches[-1].upper()
            if letter in valid_choices:
                return letter
    m = re.search(rf"^\s*([{letters}])\.(?:\s|$)", answer_text)
    if m:
        return m.group(1)
    normalized = answer_text.strip().strip(".:")  
    if len(normalized) == 1 and normalized.upper() in valid_choices:
        return normalized.upper()
    m = re.search(rf"\b([{letters}])\b", tail)
    if m:
        return m.group(1)
    if options:
        lower = answer_text.lower()
        letter_labels = sorted(valid_choices)
        matches = [lbl for lbl, opt in zip(letter_labels, options) if opt.lower() in lower]
        if len(matches) == 1:
            return matches[0]
    return None


def _normalize_triviaqa_answer(text):
    """Normalize for TriviaQA comparison."""
    text = text.lower().strip()
    for article in ["a ", "an ", "the "]:
        if text.startswith(article):
            text = text[len(article):]
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _triviaqa_match(predicted, answer_obj):
    """Check if predicted matches any accepted TriviaQA answer."""
    if not predicted:
        return False
    norm_pred = _normalize_triviaqa_answer(predicted)
    if not norm_pred:
        return False
    accepted = []
    if isinstance(answer_obj, dict):
        if answer_obj.get("value"):
            accepted.append(answer_obj["value"])
        accepted.extend(answer_obj.get("aliases", []))
        accepted.extend(answer_obj.get("normalized_aliases", []))
    for ref in accepted:
        if _normalize_triviaqa_answer(ref) == norm_pred:
            return True
    for ref in accepted:
        norm_ref = _normalize_triviaqa_answer(ref)
        if norm_ref and (norm_pred in norm_ref or norm_ref in norm_pred):
            return True
    return False


def _extract_short_answer(text):
    """Extract a short factual answer from model output."""
    m = re.search(r"(?:final\s+)?answer\s*(?:is|:)\s*(.+?)(?:\.|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if "\\boxed{" in text:
        return text.split("\\boxed{")[-1].split("}")[0].strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        return lines[0] if len(lines[0]) < 200 else lines[-1]
    return text.strip()


# ── GSM8K helpers (kept from original) ────────────────────────────────────────

def _extract_number(text: str) -> float | None:
    parts = text.split("<|think_end|>")
    zone = parts[-1] if len(parts) > 1 else text
    candidates = re.findall(r"-?[\d,]+\.?\d*", zone)
    if not candidates:
        candidates = re.findall(r"-?[\d,]+\.?\d*", text)
    for c in reversed(candidates):
        try:
            return float(c.replace(",", ""))
        except ValueError:
            continue
    return None


def _within_pct(text: str, ground_truth: str, pct: float = 0.10) -> bool:
    try:
        gt = float(ground_truth.replace(",", ""))
    except ValueError:
        return False
    if gt == 0:
        return False
    pred = _extract_number(text)
    if pred is None:
        return False
    return abs(pred - gt) / abs(gt) <= pct


# ── MATH-500 helpers ─────────────────────────────────────────────────────────

def _extract_boxed(text):
    """Extract content from \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i]
            depth -= 1
    return text[start:]


def _normalize_math_answer(answer):
    if answer is None:
        return None
    s = str(answer).strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\[,;!]", "", s)
    s = s.replace("\\displaystyle", "").replace("$", "")
    s = re.sub(r"\s+", " ", s).strip().rstrip(".")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return s.rstrip("\\").strip()


def _math_answers_equal(predicted, reference):
    if predicted is None or reference is None:
        return False
    norm_pred = _normalize_math_answer(predicted)
    norm_ref  = _normalize_math_answer(reference)
    if not norm_pred or not norm_ref:
        return False
    if norm_pred == norm_ref:
        return True
    try:
        if abs(float(norm_pred.replace(",", "")) - float(norm_ref.replace(",", ""))) < 1e-6:
            return True
    except (ValueError, OverflowError):
        pass
    try:
        from sympy.parsing.latex import parse_latex
        if parse_latex(norm_pred).equals(parse_latex(norm_ref)):
            return True
    except Exception:
        pass
    return False


def _extract_math_answer(text):
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed.strip()
    m = re.search(r"(?:final\s+)?answer\s*(?:is|:)\s*(.+?)(?:\.|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else text.strip()


# ── Hello / identity (LLM-generated prompts + LLM judge) ─────────────────────

_HELLO_GREETING_SYSTEM = """\
Generate diverse greeting prompts a user might send to an AI assistant.
Mix languages (English, French, Spanish, Italian, German, Portuguese, Dutch, Polish, Romanian, etc.).
Mix styles: casual, formal, enthusiastic, shy, one-word, full sentence.
Examples: "Hi!", "Bonjour !", "¡Hola! ¿Cómo estás?", "Guten Tag.", "Hey there :)", "Salut !".
Return ONLY a valid JSON array of strings, nothing else."""

_HELLO_IDENTITY_SYSTEM = """\
Generate diverse prompts a user might send to ask an AI assistant who or what it is.
Mix languages (English, French, Spanish, Italian, German, Portuguese, Dutch, Polish, etc.).
Mix styles: curious, direct, philosophical, casual, formal.
Examples: "Who are you?", "Qui es-tu ?", "¿Qué eres?", "What kind of AI are you?", "Tell me about yourself.", "Was bist du?".
Return ONLY a valid JSON array of strings, nothing else."""


def _make_hello_loader(api_key, judge_model):
    import asyncio
    from openai import AsyncOpenAI

    async def _gen_batch(client, system, n_batch, variation):
        msg = await client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Generate {n_batch} prompts."},
            ],
            temperature=min(0.8 + variation * 0.04, 1.3),
            max_tokens=1500,
        )
        raw = msg.choices[0].message.content.strip()
        s, e = raw.find("["), raw.rfind("]")
        if s != -1 and e > s:
            try:
                return json.loads(raw[s : e + 1])
            except Exception:
                pass
        return []

    def _load(split, seed, max_examples):
        n = max_examples or 400
        n_greet  = n // 2
        n_ident  = n - n_greet
        per_batch = 20

        async def _run_all():
            client = AsyncOpenAI(api_key=api_key)
            def _batches(total, system):
                nb = (total + per_batch - 1) // per_batch
                return [
                    _gen_batch(client, system, min(per_batch, total - i * per_batch), i)
                    for i in range(nb)
                ]
            greet_tasks = _batches(n_greet, _HELLO_GREETING_SYSTEM)
            ident_tasks = _batches(n_ident, _HELLO_IDENTITY_SYSTEM)
            results = await asyncio.gather(*(greet_tasks + ident_tasks))
            await client.close()
            greet_prompts, ident_prompts = [], []
            split_idx = len(greet_tasks)
            for r in results[:split_idx]:
                greet_prompts.extend(r)
            for r in results[split_idx:]:
                ident_prompts.extend(r)
            return greet_prompts[:n_greet], ident_prompts[:n_ident]

        greet_prompts, ident_prompts = asyncio.run(_run_all())
        examples = []
        for p in greet_prompts:
            examples.append({"prompt": p, "question": p, "ground_truth": "", "instruction": p, "hello_type": "greeting"})
        for p in ident_prompts:
            examples.append({"prompt": p, "question": p, "ground_truth": "", "instruction": p, "hello_type": "identity"})
        random.Random(seed).shuffle(examples)
        print(f"  hello: {len(greet_prompts)} greeting + {len(ident_prompts)} identity prompts generated by {judge_model}")
        return examples

    return _load


def _make_hello_batch_scorer(api_key, judge_model):
    from train_grpo_rlai import _llm_judgments, _judgment_reward

    def _batch_score(texts, ex):
        judgments = _llm_judgments(
            ex["instruction"], texts, api_key,
            reference_answer="",
            judge_model=judge_model,
        )
        return [_judgment_reward(j) for j in judgments]

    return _batch_score


def _make_hello_record():
    def _record(ex, text, reward, adv, group_type):
        return {
            "query":        ex["prompt"],
            "reasoning":    text,
            "ground_truth": "",
            "reward":       round(reward, 4),
            "advantage":    round(float(adv), 4),
            "group_type":   group_type,
            "dataset":      "hello",
        }
    return _record


# ── Creative writing (LLM-generated prompts + LLM judge) ─────────────────────

_CREATIVE_WRITING_PROMPT_SYSTEM = """\
You are a creative writing instructor. Generate diverse, engaging writing prompts.
Vary the form: short story, poem, dialogue, descriptive passage, flash fiction, letter, monologue.
Vary the topic: nature, human relationships, sci-fi, mystery, humor, grief, memory, history.
Each prompt must be 1-3 sentences, specific and self-contained.
Return ONLY a valid JSON array of strings, nothing else."""


def _make_creative_writing_loader(api_key, judge_model):
    import asyncio
    from openai import AsyncOpenAI

    async def _gen_batch(client, n_batch, variation):
        msg = await client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": _CREATIVE_WRITING_PROMPT_SYSTEM},
                {"role": "user",   "content": f"Generate {n_batch} writing prompts."},
            ],
            temperature=min(0.7 + variation * 0.05, 1.2),
            max_tokens=2000,
        )
        raw = msg.choices[0].message.content.strip()
        s, e = raw.find("["), raw.rfind("]")
        if s != -1 and e > s:
            try:
                return json.loads(raw[s : e + 1])
            except Exception:
                pass
        return []

    def _load(split, seed, max_examples):
        n = max_examples or 500
        per_batch = 20

        async def _run_all():
            client = AsyncOpenAI(api_key=api_key)
            n_batches = (n + per_batch - 1) // per_batch
            tasks = [
                _gen_batch(client, min(per_batch, n - i * per_batch), i)
                for i in range(n_batches)
            ]
            results = await asyncio.gather(*tasks)
            await client.close()
            prompts = []
            for r in results:
                prompts.extend(r)
            return prompts[:n]

        prompts = asyncio.run(_run_all())
        random.Random(seed).shuffle(prompts)
        print(f"  creative_writing: {len(prompts)} prompts generated by {judge_model}")
        return [
            {"prompt": p, "question": p, "ground_truth": "", "instruction": p}
            for p in prompts
        ]

    return _load


def _make_creative_writing_batch_scorer(api_key, judge_model):
    from train_grpo_rlai import _llm_judgments, _judgment_reward

    def _batch_score(texts, ex):
        judgments = _llm_judgments(
            ex["instruction"], texts, api_key,
            reference_answer="",
            judge_model=judge_model,
        )
        return [_judgment_reward(j) for j in judgments]

    return _batch_score


def _make_creative_writing_record():
    def _record(ex, text, reward, adv, group_type):
        return {
            "query":        ex["prompt"],
            "reasoning":    text,
            "ground_truth": "",
            "reward":       round(reward, 4),
            "advantage":    round(float(adv), 4),
            "group_type":   group_type,
            "dataset":      "creative_writing",
        }
    return _record


# ── Dataset registry ──────────────────────────────────────────────────────────

def _load_gsm8k(split, seed, max_examples):
    examples = load_gsm8k(split=split, seed=seed)
    if max_examples:
        examples = examples[:max_examples]
    return examples


def _score_gsm8k(text, ex):
    """1.0 exact, 0.5 within 10%, 0.0 otherwise.

    Only looks in the answer zone (after <|think_end|>) to avoid false
    positives where the ground truth number appears in the reasoning.
    """
    zone = _extract_answer_after_thinking(text)
    gt = ex["ground_truth"]
    if check_answer(zone, gt):
        return 1.0
    if _within_pct(zone, gt, pct=0.10):
        return 0.5
    return 0.0


def _record_gsm8k(ex, text, reward, adv, group_type):
    return {
        "query":        ex["question"],
        "reasoning":    text,
        "ground_truth": ex["ground_truth"],
        "reward":       round(reward, 4),
        "advantage":    round(float(adv), 4),
        "group_type":   group_type,
        "dataset":      "gsm8k",
    }


def _load_winogrande(split, seed, max_examples):
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split=split)
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    if max_examples:
        rows = rows[:max_examples]
    rng = random.Random(seed + 1)
    examples = []
    for row in rows:
        sentence = row["sentence"]
        option1, option2 = row["option1"], row["option2"]
        answer = row["answer"]  # "1" or "2"
        query, end_of_target = sentence.split("_")
        # Randomize option order
        if rng.random() < 0.5:
            opt_a, opt_b = option1, option2
            correct_letter = "A" if answer == "1" else "B"
        else:
            opt_a, opt_b = option2, option1
            correct_letter = "B" if answer == "1" else "A"
        prompt = (
            f"Choose the best option to fill in the blank.\n\n{query}\n\n"
            f"A. {opt_a} {end_of_target}\n"
            f"B. {opt_b} {end_of_target}"
        )
        examples.append({
            "prompt": prompt,
            "question": sentence,
            "ground_truth": correct_letter,
            "opt_a": opt_a,
            "opt_b": opt_b,
        })
    return examples


def _score_winogrande(text, ex):
    """1.0 if correct choice, 0.0 otherwise."""
    answer_text = _extract_answer_after_thinking(text)
    predicted = _parse_choice(answer_text, {"A", "B"}, options=[ex["opt_a"], ex["opt_b"]])
    return 1.0 if predicted == ex["ground_truth"] else 0.0


def _record_winogrande(ex, text, reward, adv, group_type):
    return {
        "query":        ex["prompt"],  # full prompt including A/B options
        "reasoning":    text,
        "ground_truth": ex["ground_truth"],
        "reward":       round(reward, 4),
        "advantage":    round(float(adv), 4),
        "group_type":   group_type,
        "dataset":      "winogrande",
    }


def _load_triviaqa(split, seed, max_examples):
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "unfiltered.nocontext", split=split)
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    if max_examples:
        rows = rows[:max_examples]
    examples = []
    for row in rows:
        question = row["question"]
        answer_obj = row["answer"]
        if random.random() < 0.13:
            prompt = (
                f"Answer the following question with a short factual answer.\n\n"
                f"Question: {question}\n\nAnswer:"
            )
        elif random.random() < 0.26:
            prompt = (
                f"Answer the following question with a short factual answer.\n"
                f"{question}"
            )
        else:
            prompt = question
        examples.append({
            "prompt":       prompt,
            "question":     question,
            "ground_truth": answer_obj.get("value", ""),
            "answer_obj":   answer_obj,
        })
    return examples


def _score_triviaqa(text, ex):
    """1.0 if matches any accepted answer, 0.0 otherwise."""
    answer_text = _extract_answer_after_thinking(text)
    predicted = _extract_short_answer(answer_text)
    return 1.0 if _triviaqa_match(predicted, ex["answer_obj"]) else 0.0


def _record_triviaqa(ex, text, reward, adv, group_type):
    return {
        "query":        ex["prompt"],  # full prompt with "Answer:" suffix
        "reasoning":    text,
        "ground_truth": ex["ground_truth"],
        "reward":       round(reward, 4),
        "advantage":    round(float(adv), 4),
        "group_type":   group_type,
        "dataset":      "triviaqa",
    }


def _make_local_math_loader(path):
    def _load(split, seed, max_examples):
        with open(path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        random.Random(seed).shuffle(rows)
        if max_examples:
            rows = rows[:max_examples]
        examples = []
        for row in rows:
            query = row["query"]
            gt    = str(row.get("ground_truth", row.get("answer", "")))
            examples.append({
                "prompt":       query,
                "question":     query,
                "ground_truth": gt,
                "source":       row.get("source", path),
            })
        print(f"  local_math: {len(examples)} exemples chargés depuis {path}")
        return examples
    return _load


def _score_local_math(text, ex):
    zone = _extract_answer_after_thinking(text)
    gt   = ex["ground_truth"]
    if check_answer(zone, gt):
        return 1.0
    if _within_pct(zone, gt, pct=0.10):
        return 0.5
    return 0.0


def _record_local_math(ex, text, reward, adv, group_type):
    return {
        "query":        ex["question"],
        "reasoning":    text,
        "ground_truth": ex["ground_truth"],
        "reward":       round(reward, 4),
        "advantage":    round(float(adv), 4),
        "group_type":   group_type,
        "dataset":      "local_math",
        "source":       ex.get("source", ""),
    }


def _load_piqa(split, seed, max_examples):
    from datasets import load_dataset
    ds = load_dataset("ybisk/piqa", split=split)
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    if max_examples:
        rows = rows[:max_examples]
    rng = random.Random(seed + 1)
    examples = []
    for row in rows:
        goal  = row["goal"]
        sol1, sol2 = row["sol1"], row["sol2"]
        label = int(row["label"])   # 0 → sol1 correct, 1 → sol2 correct
        # Randomize option order
        if rng.random() < 0.5:
            opt_a, opt_b = sol1, sol2
            correct_letter = "A" if label == 0 else "B"
        else:
            opt_a, opt_b = sol2, sol1
            correct_letter = "B" if label == 0 else "A"
        prompt = (
            f"{goal}\n\n"
            f"A. {opt_a}\n"
            f"B. {opt_b}"
        )
        if rng.random() < 0.5:
            prompt += "\n\nAnswer with the letter only, like 'A' or 'B'."
        examples.append({
            "prompt":       prompt,
            "question":     goal,
            "ground_truth": correct_letter,
            "opt_a":        opt_a,
            "opt_b":        opt_b,
        })
    return examples


def _score_piqa(text, ex):
    answer_text = _extract_answer_after_thinking(text)
    predicted = _parse_choice(answer_text, {"A", "B"}, options=[ex["opt_a"], ex["opt_b"]])
    return 1.0 if predicted == ex["ground_truth"] else 0.0


def _record_piqa(ex, text, reward, adv, group_type):
    return {
        "query":        ex["prompt"],
        "reasoning":    text,
        "ground_truth": ex["ground_truth"],
        "reward":       round(reward, 4),
        "advantage":    round(float(adv), 4),
        "group_type":   group_type,
        "dataset":      "piqa",
    }


def _load_mmlu(split, seed, max_examples):
    from datasets import load_dataset
    # MMLU n'a pas de split "train" — auxiliary_train est le split d'entraînement
    hf_split = "auxiliary_train" if split == "train" else split
    ds = load_dataset("cais/mmlu", "all", split=hf_split)
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    if max_examples:
        rows = rows[:max_examples]
    rng = random.Random(seed + 1)
    examples = []
    for row in rows:
        question  = row["question"]
        choices   = list(row["choices"])   # always 4
        answer_idx = int(row["answer"])    # 0-3
        indexed = list(enumerate(choices))
        rng.shuffle(indexed)
        shuffled = [c for _, c in indexed]
        new_correct_idx = next(j for j, (orig, _) in enumerate(indexed) if orig == answer_idx)
        correct_letter  = chr(ord("A") + new_correct_idx)
        prompt = (
            f"{question}\n\n"
            + "\n".join(f"{chr(ord('A') + j)}. {c}" for j, c in enumerate(shuffled))
            + "\n\nAnswer with the letter only."
        )
        examples.append({
            "prompt":        prompt,
            "question":      question,
            "ground_truth":  correct_letter,
            "choices":       shuffled,
        })
    return examples


def _score_mmlu(text, ex):
    answer_text = _extract_answer_after_thinking(text)
    predicted = _parse_choice(answer_text, {"A", "B", "C", "D"}, options=ex["choices"])
    return 1.0 if predicted == ex["ground_truth"] else 0.0


def _record_mmlu(ex, text, reward, adv, group_type):
    return {
        "query":        ex["prompt"],
        "reasoning":    text,
        "ground_truth": ex["ground_truth"],
        "reward":       round(reward, 4),
        "advantage":    round(float(adv), 4),
        "group_type":   group_type,
        "dataset":      "mmlu",
    }


def _load_mmlu_pro(split, seed, max_examples):
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    if max_examples:
        rows = rows[:max_examples]
    rng = random.Random(seed + 1)
    examples = []
    for row in rows:
        question   = row["question"]
        choices    = list(row["options"])
        answer_idx = int(row["answer_index"])
        indexed = list(enumerate(choices))
        rng.shuffle(indexed)
        shuffled = [c for _, c in indexed]
        new_correct_idx = next(j for j, (orig, _) in enumerate(indexed) if orig == answer_idx)
        correct_letter  = chr(ord("A") + new_correct_idx)
        valid = {chr(ord("A") + j) for j in range(len(shuffled))}
        prompt = (
            f"{question}\n\n"
            + "\n".join(f"{chr(ord('A') + j)}. {c}" for j, c in enumerate(shuffled))
            + "\n\nAnswer with the letter only."
        )
        examples.append({
            "prompt":        prompt,
            "question":      question,
            "ground_truth":  correct_letter,
            "choices":       shuffled,
            "valid_letters": valid,
        })
    return examples


def _score_mmlu_pro(text, ex):
    answer_text = _extract_answer_after_thinking(text)
    predicted = _parse_choice(answer_text, ex["valid_letters"], options=ex["choices"])
    return 1.0 if predicted == ex["ground_truth"] else 0.0


def _record_mmlu_pro(ex, text, reward, adv, group_type):
    return {
        "query":        ex["prompt"],
        "reasoning":    text,
        "ground_truth": ex["ground_truth"],
        "reward":       round(reward, 4),
        "advantage":    round(float(adv), 4),
        "group_type":   group_type,
        "dataset":      "mmlu_pro",
    }


def _load_gpqa_diamond(split, seed, max_examples):
    from datasets import load_dataset
    import os
    token = os.environ.get("HF_TOKEN", True)
    # GPQA only has a train split
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", token=token)
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    if max_examples:
        rows = rows[:max_examples]
    rng = random.Random(seed + 1)
    examples = []
    for row in rows:
        question = row["Question"]
        choices_with_label = [
            (row["Correct Answer"],    True),
            (row["Incorrect Answer 1"], False),
            (row["Incorrect Answer 2"], False),
            (row["Incorrect Answer 3"], False),
        ]
        rng.shuffle(choices_with_label)
        choices = [c for c, _ in choices_with_label]
        correct_idx = next(
            i for i, (_, is_correct) in enumerate(choices_with_label) if is_correct
        )
        correct_letter = chr(ord("A") + correct_idx)
        prompt = (
            f"{question}\n\n"
            + "\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
            + "\n\nAnswer only with a letter, like 'A' or 'B'."
        )
        examples.append({
            "prompt":        prompt,
            "question":      question,
            "ground_truth":  correct_letter,
            "choices":       choices,
        })
    return examples


def _score_gpqa_diamond(text, ex):
    """1.0 if correct choice (A-D), 0.0 otherwise."""
    answer_text = _extract_answer_after_thinking(text)
    predicted = _parse_choice(answer_text, {"A", "B", "C", "D"}, options=ex["choices"])
    return 1.0 if predicted == ex["ground_truth"] else 0.0


def _record_gpqa_diamond(ex, text, reward, adv, group_type):
    return {
        "query":        ex["prompt"],
        "reasoning":    text,
        "ground_truth": ex["ground_truth"],
        "reward":       round(reward, 4),
        "advantage":    round(float(adv), 4),
        "group_type":   group_type,
        "dataset":      "gpqa_diamond",
    }


def _load_commonsenseqa(split, seed, max_examples):
    from datasets import load_dataset
    hf_split = "train" if split == "train" else "validation"
    ds = load_dataset("commonsense_qa", split=hf_split)
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    if max_examples:
        rows = rows[:max_examples]
    rng = random.Random(seed + 1)
    examples = []
    for row in rows:
        question   = row["question"]
        choices    = row["choices"]["text"]
        labels     = row["choices"]["label"]
        answer_key = row["answerKey"]
        # Randomize option order (same as eval_reasoning.py)
        indexed = list(enumerate(choices))
        rng.shuffle(indexed)
        shuffled_choices = [c for _, c in indexed]
        orig_correct_idx = (
            labels.index(answer_key) if answer_key in labels
            else (ord(answer_key) - ord("A"))
        )
        new_correct_idx = next(
            j for j, (orig_i, _) in enumerate(indexed) if orig_i == orig_correct_idx
        )
        correct_letter = chr(ord("A") + new_correct_idx)
        prompt = (
            f"{question}\n\n"
            + "\n".join(f"{chr(ord('A') + j)}. {c}" for j, c in enumerate(shuffled_choices))
            + "\n\nAnswer with the letter only."
        )
        examples.append({
            "prompt":        prompt,
            "question":      question,
            "ground_truth":  correct_letter,
            "shuffled_choices": shuffled_choices,
        })
    return examples


def _score_commonsenseqa(text, ex):
    """1.0 if correct choice (A-E), 0.0 otherwise."""
    answer_text = _extract_answer_after_thinking(text)
    valid = {chr(ord("A") + i) for i in range(len(ex["shuffled_choices"]))}
    predicted = _parse_choice(answer_text, valid, options=ex["shuffled_choices"])
    return 1.0 if predicted == ex["ground_truth"] else 0.0


def _record_commonsenseqa(ex, text, reward, adv, group_type):
    return {
        "query":        ex["prompt"],
        "reasoning":    text,
        "ground_truth": ex["ground_truth"],
        "reward":       round(reward, 4),
        "advantage":    round(float(adv), 4),
        "group_type":   group_type,
        "dataset":      "commonsenseqa",
    }


def _load_math500(split, seed, max_examples):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")  # only test split exists
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    if max_examples:
        rows = rows[:max_examples]
    examples = []
    for row in rows:
        problem = row["problem"]
        answer  = row["answer"]
        examples.append({
            "prompt":       problem,
            "question":     problem,
            "ground_truth": answer,
        })
    return examples


def _score_math500(text, ex):
    """1.0 if answer matches (normalized string or numeric), 0.0 otherwise."""
    answer_text = _extract_answer_after_thinking(text)
    predicted   = _extract_math_answer(answer_text)
    return 1.0 if _math_answers_equal(predicted, ex["ground_truth"]) else 0.0


def _record_math500(ex, text, reward, adv, group_type):
    return {
        "query":        ex["prompt"],
        "reasoning":    text,
        "ground_truth": ex["ground_truth"],
        "reward":       round(reward, 4),
        "advantage":    round(float(adv), 4),
        "group_type":   group_type,
        "dataset":      "math500",
    }


DATASETS = {
    "gsm8k": {
        "load":              _load_gsm8k,
        "score":             _score_gsm8k,
        "record":            _record_gsm8k,
        "partial_threshold": 0.0,    # keep items with reward > 0 as partial
        "use_synth_template": False, # prompts already include full formatting
    },
    "winogrande": {
        "load":              _load_winogrande,
        "score":             _score_winogrande,
        "record":            _record_winogrande,
        "partial_threshold": None,   # binary correct/incorrect
        "use_synth_template": True,  # model needs <|im_start|>...<|think_start|>
    },
    "triviaqa": {
        "load":              _load_triviaqa,
        "score":             _score_triviaqa,
        "record":            _record_triviaqa,
        "partial_threshold": None,   # binary correct/incorrect
        "use_synth_template": True,  # model needs <|im_start|>...<|think_start|>
    },
    "math500": {
        "load":              _load_math500,
        "score":             _score_math500,
        "record":            _record_math500,
        "partial_threshold": None,   # binary correct/incorrect
        "use_synth_template": True,  # model needs <|im_start|>...<|think_start|>
    },
    "local_math": {
        "load":              None,   # set at runtime via --local_file
        "score":             _score_local_math,
        "record":            _record_local_math,
        "partial_threshold": 0.0,    # garder les partiels (comme gsm8k)
        "use_synth_template": True,
    },
    "piqa": {
        "load":              _load_piqa,
        "score":             _score_piqa,
        "record":            _record_piqa,
        "partial_threshold": None,
        "use_synth_template": True,
    },
    "mmlu": {
        "load":              _load_mmlu,
        "score":             _score_mmlu,
        "record":            _record_mmlu,
        "partial_threshold": None,
        "use_synth_template": True,
    },
    "mmlu_pro": {
        "load":              _load_mmlu_pro,
        "score":             _score_mmlu_pro,
        "record":            _record_mmlu_pro,
        "partial_threshold": None,
        "use_synth_template": True,
    },
    "gpqa_diamond": {
        "load":              _load_gpqa_diamond,
        "score":             _score_gpqa_diamond,
        "record":            _record_gpqa_diamond,
        "partial_threshold": None,   # binary correct/incorrect
        "use_synth_template": True,
    },
    "commonsenseqa": {
        "load":              _load_commonsenseqa,
        "score":             _score_commonsenseqa,
        "record":            _record_commonsenseqa,
        "partial_threshold": None,
        "use_synth_template": True,
    },
    "hello": {
        "load":              None,   # set at runtime via factory
        "score":             None,
        "batch_score":       None,   # set at runtime via factory
        "record":            None,   # set at runtime via factory
        "partial_threshold": None,
        "selection":         "best",
        "use_synth_template": True,
    },
    "creative_writing": {
        "load":              None,   # set at runtime via factory (needs api_key)
        "score":             None,   # not used — batch_score replaces it
        "batch_score":       None,   # set at runtime via factory
        "record":            None,   # set at runtime via factory
        "partial_threshold": None,
        "selection":         "best", # keep only max-scored item(s)
        "use_synth_template": True,
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default="./checkpoints/graft/start.pt")
    p.add_argument("--dataset",       default="gsm8k", choices=list(DATASETS.keys()))
    p.add_argument("--output",        default=None, help="output JSONL (default: data/<dataset>_cot.jsonl)")
    p.add_argument("--split",         default="train",   help="train ou test")
    p.add_argument("--max_examples",  type=int, default=None, help="limiter le nombre d'exemples")
    p.add_argument("--g",             type=int, default=16,   help="completions par exemple")
    p.add_argument("--temperature",   type=float, default=0.4)
    p.add_argument("--max_tokens",    type=int, default=1000)
    p.add_argument("--batch_size",    type=int, default=8,    help="prompts en parallèle")
    p.add_argument("--seed",          type=int, default=24)
    p.add_argument("--local_file",    default=None,
                   help="chemin vers un JSONL local {query, ground_truth} (requis pour local_math)")
    p.add_argument("--max_prompt_tokens", type=int, default=None,
                   help="filtrer les prompts plus longs que N tokens (évite les OOM)")
    p.add_argument("--dump_all", action="store_true",
                   help="écrire toutes les completions sans filtrage par reward/advantage")
    p.add_argument("--openai_api_key", default=os.environ.get("OPENAI_API_KEY", ""),
                   help="OpenAI API key (required for creative_writing / hello)")
    p.add_argument("--judge_model",   default="gpt-4.1-mini",
                   help="Judge model for creative_writing / hello scoring")
    args = p.parse_args()

    if args.output is None:
        args.output = f"./data/{args.dataset}_cot.jsonl"

    # Instantiate runtime-factories pour les datasets à loader dynamique
    if args.dataset == "local_math":
        if not args.local_file:
            raise ValueError("--local_file requis pour --dataset local_math")
        DATASETS["local_math"]["load"] = _make_local_math_loader(args.local_file)

    if args.dataset in ("creative_writing", "hello"):
        if not args.openai_api_key:
            raise ValueError(f"--openai_api_key (or $OPENAI_API_KEY) required for {args.dataset}")
        if args.dataset == "creative_writing":
            DATASETS["creative_writing"]["load"]        = _make_creative_writing_loader(args.openai_api_key, args.judge_model)
            DATASETS["creative_writing"]["batch_score"] = _make_creative_writing_batch_scorer(args.openai_api_key, args.judge_model)
            DATASETS["creative_writing"]["record"]      = _make_creative_writing_record()
        elif args.dataset == "hello":
            DATASETS["hello"]["load"]        = _make_hello_loader(args.openai_api_key, args.judge_model)
            DATASETS["hello"]["batch_score"] = _make_hello_batch_scorer(args.openai_api_key, args.judge_model)
            DATASETS["hello"]["record"]      = _make_hello_record()

    ds_cfg            = DATASETS[args.dataset]
    score_fn          = ds_cfg.get("score")
    batch_score_fn    = ds_cfg.get("batch_score")
    record_fn         = ds_cfg["record"]
    partial_threshold = ds_cfg["partial_threshold"]
    selection         = ds_cfg.get("selection", "correct")
    use_synth_template = ds_cfg["use_synth_template"]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading {args.checkpoint} …")
    gen = TorchGenerator(args.checkpoint, device="cuda")

    examples = ds_cfg["load"](args.split, args.seed, args.max_examples)

    if args.max_prompt_tokens:
        before = len(examples)
        examples = [
            ex for ex in examples
            if len(gen.tokenizer([ex["prompt"]])[0]) <= args.max_prompt_tokens
        ]
        print(f"  filtre prompt>{args.max_prompt_tokens} tokens: {before} → {len(examples)} exemples")

    print(f"  dataset={args.dataset}  {len(examples)} exemples  G={args.g}  temp={args.temperature}")

    n_written = 0
    n_correct_groups = 0
    n_partial_groups = 0
    n_empty_groups   = 0
    t0 = time.perf_counter()

    G = args.g
    with open(args.output, "w") as f:
        for batch_start in range(0, len(examples), args.batch_size):
            batch = examples[batch_start : batch_start + args.batch_size]

            # Flatten: each prompt repeated G times → batch_size*G in one call
            flat_prompts = [ex["prompt"] for ex in batch for _ in range(G)]
            flat_texts = gen.generate_batch(
                flat_prompts,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                use_synth_template=use_synth_template,
                return_token_ids=False,
                top_k=10,
            )

            for j, ex in enumerate(batch):
                i = batch_start + j
                texts = flat_texts[j * G : (j + 1) * G]

                if batch_score_fn is not None:
                    rewards = batch_score_fn(texts, ex)
                else:
                    rewards = [score_fn(t, ex) for t in texts]

                advantages = list(_compute_advantages(rewards, normalize_std=False))

                if args.dump_all:
                    kept = [(t, r, a) for t, r, a in zip(texts, rewards, advantages)]
                    group_type = "all"
                    if any(r == 1.0 for r in rewards):
                        n_correct_groups += 1
                    elif any(r > 0.0 for r in rewards):
                        n_partial_groups += 1
                    else:
                        n_empty_groups += 1
                elif selection == "best":
                    max_r = max(rewards) if rewards else 0.0
                    if max_r > 0.0:
                        kept = [(t, r, a) for t, r, a in zip(texts, rewards, advantages)
                                if r >= max_r - 1e-6]
                        group_type = "best"
                        n_correct_groups += 1
                    else:
                        kept = []
                        group_type = "empty"
                        n_empty_groups += 1
                else:
                    has_correct = any(r == 1.0 for r in rewards)
                    if has_correct:
                        kept = [(t, r, a) for t, r, a in zip(texts, rewards, advantages) if r == 1.0]
                        group_type = "correct"
                        n_correct_groups += 1
                    elif partial_threshold is not None:
                        kept = [(t, r, a) for t, r, a in zip(texts, rewards, advantages) if r > partial_threshold]
                        group_type = "partial"
                        if kept:
                            n_partial_groups += 1
                        else:
                            n_empty_groups += 1
                    else:
                        kept = []
                        group_type = "empty"
                        n_empty_groups += 1

                for text, reward, adv in kept:
                    f.write(json.dumps(record_fn(ex, text, reward, adv, group_type), ensure_ascii=False) + "\n")
                    n_written += 1

            n_done = min(batch_start + args.batch_size, len(examples))
            if n_done % 50 < args.batch_size or n_done == len(examples):
                elapsed = time.perf_counter() - t0
                eta_s = elapsed / n_done * (len(examples) - n_done)
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_s))
                pct_correct = 100 * n_correct_groups / n_done
                pct_partial = 100 * n_partial_groups / n_done
                pct_empty   = 100 * n_empty_groups   / n_done
                print(f"  [{n_done:5d}/{len(examples)}]  "
                      f"correct={pct_correct:.1f}%  partial={pct_partial:.1f}%  "
                      f"empty={pct_empty:.1f}%  "
                      f"written={n_written}  {elapsed:.0f}s  ETA {eta_str}")
                f.flush()

    elapsed = time.perf_counter() - t0
    print(f"\nTerminé en {elapsed:.0f}s")
    print(f"  {n_written} entrées écrites → {args.output}")
    print(f"  groupes : correct={n_correct_groups}  partial={n_partial_groups}  "
          f"vides (ignorés)={n_empty_groups}")


if __name__ == "__main__":
    main()

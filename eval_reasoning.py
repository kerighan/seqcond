#!/usr/bin/env python3
"""Reasoning-based evaluation for instruction-tuned models (SYNTH).

Unlike eval.py which uses log-probability scoring for base models,
this script prompts the model to generate an answer (with optional
<|think_start|>...<|think_end|> reasoning) and parses the response.
"""
import re
import time
import torch
import random
import subprocess
from datasets import load_dataset

from seqcond.torch.generator import TorchGenerator


def _format_dist(choice_counts, total, parse_failures):
    """Format choice distribution string for progress display."""
    answered = total - parse_failures
    if answered <= 0:
        return ""
    return " ".join(
        f"{k}:{choice_counts.get(k, 0)/answered*100:.0f}%"
        for k in sorted(choice_counts)
    )


def _send_notification(title: str, message: str):
    """Send a Linux desktop notification."""
    try:
        subprocess.run(
            ["notify-send", title, message],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


def _maybe_shuffle_dataset(dataset, seed: int, enabled: bool = True):
    if not enabled:
        return dataset
    try:
        return dataset.shuffle(seed=seed)
    except Exception:
        return dataset


def _collect_generation(gen, prompt, max_new_tokens=512, temperature=0.0, **kwargs):
    """Run generate() and collect all yielded tokens into a single string."""
    tokens = []
    for tok in gen.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=1.0,
        top_k=0,
        verbose=False,
        use_cuda_graph=False,
        use_synth_template=True,
        **kwargs,
    ):
        tokens.append(tok)
    return "".join(tokens)


def _extract_answer_after_thinking(text, debug: bool = False, debug_id: str = ""):
    """Extract the final answer from model output, ignoring thinking tokens.

    The model may produce:
      <|think_start|>...reasoning...<|think_end|>The answer is 1.
    or just:
      The answer is 2.

    Returns the text after the last <|think_end|>, or the full text if
    no thinking tokens are present.
    """
    if debug:
        prefix = f"[EXTRACT{(' ' + debug_id) if debug_id else ''}]"
        print(f"  {prefix} raw_start={text[:120]!r}")
        print(f"  {prefix} raw_end={text[-120:]!r}")
    if "<|think_end|>" in text:
        # Take everything after the last <|think_end|>
        if debug:
            print(f"  {prefix} found <|think_end|> (using text after last)")
        text = text.split("<|think_end|>")[-1]
    elif "<|think_start|>" in text:
        # Model started thinking but never finished (ran out of tokens)
        # Try to use text before thinking started, if any
        if debug:
            print(f"  {prefix} found <|think_start|> without <|think_end|>")
        before = text.split("<|think_start|>")[0].strip()
        if before:
            if debug:
                print(f"  {prefix} using prefix before <|think_start|>")
            text = before
        # else: no answer outside thinking block, keep full text for fallback parsing
    # Also strip any remaining special tokens
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    if debug:
        cleaned = text.strip()
        print(f"  {prefix} cleaned={cleaned[:200]!r}")
    return text.strip()


def _is_debug_extract_enabled(gen) -> bool:
    return bool(getattr(gen, "debug_extract", False))


def _parse_choice(answer_text, valid_choices, options=None):
    """Parse a choice (e.g. 'A', 'B', 'C', 'D') from the model's answer text.

    Tries several strategies:
    1. Direct match of the letter alone
    2. Patterns like "answer is A", "option A"
    3. First standalone occurrence of a valid letter
    4. Fallback: match option text back to letter
    """
    letters = "".join(sorted(valid_choices))
    answer_text = answer_text.strip()
    if not answer_text:
        return None

    # Guardrail: reject prompt-like enumerations such as "A, B, C, D, E".
    # These often appear when the model echoes the instruction instead of answering.
    enum_pat = rf"^\s*(?:[{letters}](?:\.|\b)\s*[,/;]\s*)*(?:[{letters}](?:\.|\b))(?:\s*(?:,|and|or)\s*[{letters}](?:\.|\b))*\s*$"
    if re.match(enum_pat, answer_text, flags=re.IGNORECASE):
        return None

    # Work on a tail window to reduce interference from earlier option lists.
    tail = answer_text[-400:] if len(answer_text) > 400 else answer_text

    # Strategy 1: explicit answer patterns (case-insensitive is OK here)
    # Take the LAST match in the tail (models sometimes restate or correct themselves).
    letter_class_any_case = f"[{letters}{letters.lower()}]"
    for pattern in [
        rf"\bfinal\s+answer\b\s*[:\-]?\s*({letter_class_any_case})(?:\.|\b)",
        rf"\banswer\b\s*(?:is\s*)?[:\-]?\s*({letter_class_any_case})(?:\.|\b)",
        rf"\*\*\s*(?:final\s+answer|answer)\s*[:\-]?\s*({letter_class_any_case})(?:\.|\b)",
    ]:
        matches = re.findall(pattern, tail, re.IGNORECASE)
        if matches:
            letter = matches[-1].upper()
            if letter in valid_choices:
                return letter

    # Strategy 2: leading "A." / "B." style answers
    # NOTE: do this AFTER explicit patterns, because the model often re-lists options
    # at the beginning of the answer text.
    m = re.search(rf"^\s*([{letters}])\.(?:\s|$)", answer_text)
    if m:
        return m.group(1)

    # Strategy 3: the entire answer is just the letter
    normalized = answer_text.strip().strip(".:")
    if len(normalized) == 1 and normalized.upper() in valid_choices:
        return normalized.upper()

    # Strategy 4: case-sensitive standalone letter/letter-dot near end.
    # (Do NOT use IGNORECASE here, to avoid matching English 'a' -> 'A'.)
    m = re.search(rf"\b([{letters}])\.(?:\s|$)", tail)
    if m:
        return m.group(1)
    m = re.search(rf"\b([{letters}])\b", tail)
    if m:
        return m.group(1)

    # Strategy 5: option text matching
    if options:
        lower = answer_text.lower()
        letter_labels = sorted(valid_choices)
        matches = [
            lbl for lbl, opt in zip(letter_labels, options) if opt.lower() in lower
        ]
        if len(matches) == 1:
            return matches[0]

    return None


def evaluate_winogrande(
    gen,
    dataset,
    max_samples=None,
    max_new_tokens=512,
    batch_size=16,
    verbose_examples=5,
    max_thinking_tokens=None,
    output_constraints=None,
):
    """Evaluate on Winogrande using batched generative reasoning."""
    correct = 0
    total = 0
    parse_failures = 0
    choice_counts = {}
    rng = random.Random(42)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    n = len(dataset)
    print(
        f"Evaluating on {n} samples (batched generative reasoning, bs={batch_size})..."
    )
    start_time = time.time()

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = dataset.select(range(batch_start, batch_end))

        # Build prompts for the batch
        prompts = []
        metadata = []  # (option1, option2, correct_letter) per sample
        for example in batch:
            sentence = example["sentence"]
            option1 = example["option1"]
            option2 = example["option2"]
            answer = example["answer"]  # "1" or "2"

            # Randomize option order to prevent positional bias
            if rng.random() < 0.5:
                opt_a, opt_b = option1, option2
                correct_letter = "A" if answer == "1" else "B"
            else:
                opt_a, opt_b = option2, option1
                correct_letter = "B" if answer == "1" else "A"

            prompt = (
                # f"Complete the following sentence by choosing the correct option.\n\n"
                f"{sentence}\n"
                f"A. {opt_a}\n"
                f"B. {opt_b}\n"
            )
            prompts.append(prompt)
            metadata.append((opt_a, opt_b, correct_letter))

        # Batched generation
        outputs = gen.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            max_thinking_tokens=max_thinking_tokens,
            output_constraints=output_constraints,
        )

        # Parse results
        for i, (output, (opt_a, opt_b, correct_letter)) in enumerate(
            zip(outputs, metadata)
        ):
            idx = batch_start + i
            answer_text = _extract_answer_after_thinking(output)
            predicted = _parse_choice(answer_text, {"A", "B"}, options=[opt_a, opt_b])

            if predicted is None:
                parse_failures += 1
                if idx < verbose_examples:
                    print(f"  [PARSE FAIL] idx={idx}")
                    print(f"    Output: {output[:200]}")
                    print(f"    Answer text: {answer_text[:200]}")
            else:
                choice_counts[predicted] = choice_counts.get(predicted, 0) + 1
                if predicted == correct_letter:
                    correct += 1

            total += 1

        # Progress report after each batch
        elapsed = time.time() - start_time
        speed = total / elapsed if elapsed > 0 else 0
        acc = correct / total * 100
        dist = _format_dist(choice_counts, total, parse_failures)
        print(
            f"  {total}/{n} | Acc: {acc:.1f}% "
            f"| Parse fails: {parse_failures} "
            f"| Dist: [{dist}] "
            f"| Speed: {speed:.1f} samples/s"
        )

    elapsed = time.time() - start_time
    accuracy = correct / total * 100 if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Winogrande (Reasoning) Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(
        f"  Parse failures: {parse_failures}/{total} ({parse_failures/total*100:.1f}%)"
    )
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {total/elapsed:.1f} samples/s")
    print(f"{'='*60}")

    return accuracy


def evaluate_gpqa(
    gen,
    dataset,
    max_samples=None,
    max_new_tokens=512,
    batch_size=16,
    verbose_examples=5,
    max_thinking_tokens=None,
    output_constraints=None,
):
    """Evaluate on GPQA using batched generative reasoning (4-choice)."""
    correct = 0
    total = 0
    parse_failures = 0
    choice_counts = {}
    rng = random.Random(0)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    n = len(dataset)
    print(
        f"Evaluating on {n} samples (batched generative reasoning, bs={batch_size})..."
    )
    start_time = time.time()

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = dataset.select(range(batch_start, batch_end))

        prompts = []
        metadata = []  # (options_list, correct_letter) per sample
        for example in batch:
            question = example["Question"]
            choices_with_label = [
                (example["Correct Answer"], True),
                (example["Incorrect Answer 1"], False),
                (example["Incorrect Answer 2"], False),
                (example["Incorrect Answer 3"], False),
            ]
            rng.shuffle(choices_with_label)
            choices = [c for c, _ in choices_with_label]
            correct_idx = next(
                i for i, (_, is_correct) in enumerate(choices_with_label) if is_correct
            )
            correct_letter = chr(ord("A") + correct_idx)

            assert len(choices) == 4

            prompt = (
                f"{question}\n\n"
                f"A. {choices[0]}\n"
                f"B. {choices[1]}\n"
                f"C. {choices[2]}\n"
                f"D. {choices[3]}"
            )
            # Truncate prompt if too long (whole prompt, to handle long choices)
            model_maxlen = getattr(gen.model, "maxlen", 2048)
            max_prompt_tokens = model_maxlen // 2  # leave half for thinking + answer
            prompt_tokens = gen.tokenizer.encode(prompt)
            if len(prompt_tokens) > max_prompt_tokens:
                prompt = gen.tokenizer.decode(prompt_tokens[:max_prompt_tokens])
            prompts.append(prompt)
            metadata.append((choices, correct_letter))

        outputs = gen.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            max_thinking_tokens=max_thinking_tokens,
            output_constraints=output_constraints,
        )

        for i, (output, (choices, correct_letter)) in enumerate(zip(outputs, metadata)):
            idx = batch_start + i
            answer_text = _extract_answer_after_thinking(output)
            predicted = _parse_choice(
                answer_text, {"A", "B", "C", "D"}, options=choices
            )

            if predicted is None:
                parse_failures += 1
                if idx < verbose_examples:
                    print(f"  [PARSE FAIL] idx={idx}")
                    print(f"    Output: {output[:200]}")
                    print(f"    Answer text: {answer_text[:200]}")
            else:
                choice_counts[predicted] = choice_counts.get(predicted, 0) + 1
                if predicted == correct_letter:
                    correct += 1

            total += 1

        elapsed = time.time() - start_time
        speed = total / elapsed if elapsed > 0 else 0
        acc = correct / total * 100
        dist = _format_dist(choice_counts, total, parse_failures)
        print(
            f"  {total}/{n} | Acc: {acc:.1f}% "
            f"| Parse fails: {parse_failures} "
            f"| Dist: [{dist}] "
            f"| Speed: {speed:.1f} samples/s"
        )

    elapsed = time.time() - start_time
    accuracy = correct / total * 100 if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"GPQA (Reasoning) Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(
        f"  Parse failures: {parse_failures}/{total} ({parse_failures/total*100:.1f}%)"
    )
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {total/elapsed:.1f} samples/s")
    print(f"{'='*60}")

    return accuracy


def evaluate_openbookqa(
    gen,
    dataset,
    max_samples=None,
    max_new_tokens=512,
    batch_size=16,
    verbose_examples=5,
    max_thinking_tokens=None,
    output_constraints=None,
):
    """Evaluate on OpenBookQA using batched generative reasoning (4-choice)."""
    correct = 0
    total = 0
    parse_failures = 0
    choice_counts = {}  # track distribution of predicted letters
    rng = random.Random(42)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    n = len(dataset)
    print(
        f"Evaluating on {n} samples (batched generative reasoning, bs={batch_size})..."
    )
    start_time = time.time()

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = dataset.select(range(batch_start, batch_end))

        prompts = []
        metadata = []  # (choices_list, correct_letter) per sample
        for example in batch:
            question = example.get("question_stem")
            if question is None and isinstance(example.get("question"), dict):
                question = example["question"].get("stem")
            if question is None:
                question = ""

            choices_obj = example.get("choices")
            if isinstance(choices_obj, dict):
                choices = choices_obj.get("text") or []
                labels = choices_obj.get("label") or []
            else:
                choices = []
                labels = []

            answer_key = example.get("answerKey") or example.get("answer") or ""
            answer_key = str(answer_key).strip().upper()

            # Randomize option order
            indexed = list(enumerate(choices))
            rng.shuffle(indexed)
            shuffled_choices = [c for _, c in indexed]
            # Find where the correct answer ended up
            orig_correct_idx = (
                labels.index(answer_key)
                if answer_key in labels
                else (ord(answer_key) - ord("A"))
            )
            new_correct_idx = next(
                j for j, (orig_i, _) in enumerate(indexed) if orig_i == orig_correct_idx
            )
            correct_letter = chr(ord("A") + new_correct_idx)

            prompt = f"{question}\n\n" + "\n".join(
                f"{chr(ord('A') + j)}. {c}" for j, c in enumerate(shuffled_choices)
            )
            prompts.append(prompt)
            metadata.append((shuffled_choices, correct_letter))

        outputs = gen.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            max_thinking_tokens=max_thinking_tokens,
            output_constraints=output_constraints,
        )

        for i, (output, (choices, correct_letter)) in enumerate(zip(outputs, metadata)):
            idx = batch_start + i
            debug_extract = bool(verbose_examples) and idx < verbose_examples
            answer_text = _extract_answer_after_thinking(
                output,
                debug=debug_extract and _is_debug_extract_enabled(gen),
                debug_id=f"idx={idx}",
            )
            valid = {chr(ord("A") + j) for j in range(len(choices))}
            predicted = _parse_choice(answer_text, valid, options=choices)

            if idx < verbose_examples:
                print(
                    f"  [DEBUG] idx={idx} predicted={predicted} correct={correct_letter}"
                )
                print(f"    Answer text: {answer_text[:300]}")
                print(f"    Choices: {[c[:50] for c in choices]}")

            if predicted is None:
                parse_failures += 1
            else:
                choice_counts[predicted] = choice_counts.get(predicted, 0) + 1
                if predicted == correct_letter:
                    correct += 1

            total += 1

        elapsed = time.time() - start_time
        speed = total / elapsed if elapsed > 0 else 0
        acc = correct / total * 100
        answered = total - parse_failures
        dist = (
            " ".join(
                f"{k}:{choice_counts.get(k, 0)/answered*100:.0f}%"
                for k in sorted(choice_counts)
            )
            if answered > 0
            else ""
        )
        print(
            f"  {total}/{n} | Acc: {acc:.1f}% "
            f"| Parse fails: {parse_failures} "
            f"| Dist: [{dist}] "
            f"| Speed: {speed:.1f} samples/s"
        )

    elapsed = time.time() - start_time
    accuracy = correct / total * 100 if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"OpenBookQA (Reasoning) Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(
        f"  Parse failures: {parse_failures}/{total} ({parse_failures/total*100:.1f}%)"
    )
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {total/elapsed:.1f} samples/s")
    print(f"{'='*60}")

    return accuracy


def evaluate_commonsenseqa(
    gen,
    dataset,
    max_samples=None,
    max_new_tokens=512,
    batch_size=16,
    verbose_examples=5,
    max_thinking_tokens=None,
    output_constraints=None,
):
    """Evaluate on CommonsenseQA using batched generative reasoning (5-choice)."""
    correct = 0
    total = 0
    parse_failures = 0
    choice_counts = {}
    rng = random.Random(42)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    n = len(dataset)
    print(
        f"Evaluating on {n} samples (batched generative reasoning, bs={batch_size})..."
    )
    start_time = time.time()

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = dataset.select(range(batch_start, batch_end))

        prompts = []
        metadata = []
        for example in batch:
            question = example["question"]
            choices = example["choices"]["text"]
            labels = example["choices"]["label"]
            answer_key = example["answerKey"]

            # Randomize option order
            indexed = list(enumerate(choices))
            rng.shuffle(indexed)
            shuffled_choices = [c for _, c in indexed]
            orig_correct_idx = (
                labels.index(answer_key)
                if answer_key in labels
                else (ord(answer_key) - ord("A"))
            )
            new_correct_idx = next(
                j for j, (orig_i, _) in enumerate(indexed) if orig_i == orig_correct_idx
            )
            correct_letter = chr(ord("A") + new_correct_idx)
            num_choices = len(shuffled_choices)

            prompt = (
                f"{question}\n\n"
                + "\n".join(
                    f"{chr(ord('A') + j)}. {c}" for j, c in enumerate(shuffled_choices)
                )
                + "\n"
            )
            prompts.append(prompt)
            metadata.append((prompt, shuffled_choices, correct_letter))

        outputs = gen.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            max_thinking_tokens=max_thinking_tokens,
            output_constraints=output_constraints,
        )

        for i, (output, (prompt, choices, correct_letter)) in enumerate(
            zip(outputs, metadata)
        ):
            idx = batch_start + i
            debug_extract = bool(verbose_examples) and idx < verbose_examples
            answer_text = _extract_answer_after_thinking(
                output,
                debug=debug_extract and getattr(gen, "debug_extract", False),
                debug_id=f"idx={idx}",
            )
            valid = {chr(ord("A") + j) for j in range(len(choices))}
            predicted = _parse_choice(answer_text, valid, options=choices)

            if predicted is None:
                parse_failures += 1
                if idx < verbose_examples:
                    print(f"  [PARSE FAIL] idx={idx}")
                    print(f"    Prompt: {prompt[:300]}")
                    print(f"    Output: {output[:300]}")
                    print(f"    Answer text: {answer_text[:300]}")
            else:
                choice_counts[predicted] = choice_counts.get(predicted, 0) + 1
                if idx < verbose_examples:
                    print(
                        f"  [DEBUG] idx={idx} predicted={predicted} correct={correct_letter}"
                    )
                if predicted == correct_letter:
                    correct += 1

            total += 1

        elapsed = time.time() - start_time
        speed = total / elapsed if elapsed > 0 else 0
        acc = correct / total * 100
        dist = _format_dist(choice_counts, total, parse_failures)
        print(
            f"  {total}/{n} | Acc: {acc:.1f}% "
            f"| Parse fails: {parse_failures} "
            f"| Dist: [{dist}] "
            f"| Speed: {speed:.1f} samples/s"
        )

    elapsed = time.time() - start_time
    accuracy = correct / total * 100 if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"CommonsenseQA (Reasoning) Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(
        f"  Parse failures: {parse_failures}/{total} ({parse_failures/total*100:.1f}%)"
    )
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {total/elapsed:.1f} samples/s")
    print(f"{'='*60}")

    return accuracy


def evaluate_hellaswag(
    gen,
    dataset,
    max_samples=None,
    max_new_tokens=512,
    batch_size=16,
    verbose_examples=5,
    max_thinking_tokens=None,
    output_constraints=None,
):
    """Evaluate on HellaSwag using batched generative reasoning (4-choice)."""
    correct = 0
    total = 0
    parse_failures = 0
    choice_counts = {}
    rng = random.Random(42)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    n = len(dataset)
    print(
        f"Evaluating on {n} samples (batched generative reasoning, bs={batch_size})..."
    )
    start_time = time.time()

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = dataset.select(range(batch_start, batch_end))

        prompts = []
        metadata = []
        for example in batch:
            ctx = example["ctx"]
            endings = example["endings"]
            label = int(example["label"])

            # Randomize option order
            indexed = list(enumerate(endings))
            rng.shuffle(indexed)
            shuffled_endings = [e for _, e in indexed]
            new_correct_idx = next(
                j for j, (orig_i, _) in enumerate(indexed) if orig_i == label
            )
            correct_letter = chr(ord("A") + new_correct_idx)

            prompt = (
                f"Complete the following sentence by choosing the correct ending.\n\n"
                f"Context: {ctx}\n\n"
                + "\n".join(
                    f"{chr(ord('A') + j)}. {e}" for j, e in enumerate(shuffled_endings)
                )
            )
            prompts.append(prompt)
            metadata.append((shuffled_endings, correct_letter))

        outputs = gen.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            max_thinking_tokens=max_thinking_tokens,
            output_constraints=output_constraints,
        )

        for i, (output, (endings, correct_letter)) in enumerate(zip(outputs, metadata)):
            idx = batch_start + i
            answer_text = _extract_answer_after_thinking(output)
            valid = {"A", "B", "C", "D"}
            predicted = _parse_choice(answer_text, valid, options=endings)

            if predicted is None:
                parse_failures += 1
            else:
                choice_counts[predicted] = choice_counts.get(predicted, 0) + 1
                if predicted == correct_letter:
                    correct += 1

            total += 1

        elapsed = time.time() - start_time
        speed = total / elapsed if elapsed > 0 else 0
        acc = correct / total * 100
        dist = _format_dist(choice_counts, total, parse_failures)
        print(
            f"  {total}/{n} | Acc: {acc:.1f}% "
            f"| Parse fails: {parse_failures} "
            f"| Dist: [{dist}] "
            f"| Speed: {speed:.1f} samples/s"
        )

    elapsed = time.time() - start_time
    accuracy = correct / total * 100 if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"HellaSwag (Reasoning) Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(
        f"  Parse failures: {parse_failures}/{total} ({parse_failures/total*100:.1f}%)"
    )
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {total/elapsed:.1f} samples/s")
    print(f"{'='*60}")

    return accuracy


def evaluate_piqa(
    gen,
    dataset,
    max_samples=None,
    max_new_tokens=512,
    batch_size=16,
    verbose_examples=5,
    max_thinking_tokens=None,
    output_constraints=None,
):
    """Evaluate on PIQA using batched generative reasoning (2-choice)."""
    correct = 0
    total = 0
    parse_failures = 0
    choice_counts = {}
    rng = random.Random(42)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    n = len(dataset)
    print(
        f"Evaluating on {n} samples (batched generative reasoning, bs={batch_size})..."
    )
    start_time = time.time()

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = dataset.select(range(batch_start, batch_end))

        prompts = []
        metadata = []
        for example in batch:
            goal = example["goal"]
            solutions = [example["sol1"], example["sol2"]]
            label = int(example["label"])

            # Randomize option order
            if rng.random() < 0.5:
                opt_a, opt_b = solutions[0], solutions[1]
                correct_letter = "A" if label == 0 else "B"
            else:
                opt_a, opt_b = solutions[1], solutions[0]
                correct_letter = "B" if label == 0 else "A"

            prompt = f"{goal}\n\n" f"A. {opt_a}\n" f"B. {opt_b}\n"
            prompts.append(prompt)
            metadata.append(([opt_a, opt_b], correct_letter))

        outputs = gen.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            max_thinking_tokens=max_thinking_tokens,
            output_constraints=output_constraints,
        )

        for i, (output, (options, correct_letter)) in enumerate(zip(outputs, metadata)):
            idx = batch_start + i

            # get prompt
            prompt = prompts[i]

            answer_text = _extract_answer_after_thinking(output)
            # print(answer_text)
            # print(correct_letter)
            # print("-" * 50)
            valid = {"A", "B"}
            predicted = _parse_choice(answer_text, valid, options=options)
            print(prompt)
            print(output)
            print(predicted, "=>", correct_letter)
            print("-" * 80)

            if predicted is None:
                parse_failures += 1
            else:
                choice_counts[predicted] = choice_counts.get(predicted, 0) + 1
                if predicted == correct_letter:
                    correct += 1

            total += 1

        elapsed = time.time() - start_time
        speed = total / elapsed if elapsed > 0 else 0
        acc = correct / total * 100
        dist = _format_dist(choice_counts, total, parse_failures)
        print(
            f"  {total}/{n} | Acc: {acc:.1f}% "
            f"| Parse fails: {parse_failures} "
            f"| Dist: [{dist}] "
            f"| Speed: {speed:.1f} samples/s"
        )

    elapsed = time.time() - start_time
    accuracy = correct / total * 100 if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"PIQA (Reasoning) Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(
        f"  Parse failures: {parse_failures}/{total} ({parse_failures/total*100:.1f}%)"
    )
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {total/elapsed:.1f} samples/s")
    print(f"{'='*60}")

    return accuracy


def evaluate_arc(
    gen,
    dataset,
    max_samples=None,
    max_new_tokens=512,
    batch_size=16,
    verbose_examples=5,
    max_thinking_tokens=None,
    output_constraints=None,
):
    """Evaluate on ARC using batched generative reasoning (3-5 choices)."""
    correct = 0
    total = 0
    parse_failures = 0
    choice_counts = {}
    rng = random.Random(42)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    n = len(dataset)
    print(
        f"Evaluating on {n} samples (batched generative reasoning, bs={batch_size})..."
    )
    start_time = time.time()

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = dataset.select(range(batch_start, batch_end))

        prompts = []
        metadata = []
        for example in batch:
            question = example["question"]
            choices = example["choices"]["text"]
            label = example["answerKey"]

            # Convert label to index
            if label.isdigit():
                orig_correct_idx = int(label) - 1
            else:
                orig_correct_idx = ord(label.upper()) - ord("A")

            # Randomize option order
            indexed = list(enumerate(choices))
            rng.shuffle(indexed)
            shuffled_choices = [c for _, c in indexed]
            new_correct_idx = next(
                j for j, (orig_i, _) in enumerate(indexed) if orig_i == orig_correct_idx
            )
            correct_letter = chr(ord("A") + new_correct_idx)
            num_choices = len(shuffled_choices)

            prompt = f"{question}\n\n" + "\n".join(
                f"{chr(ord('A') + j)}. {c}" for j, c in enumerate(shuffled_choices)
            )
            prompts.append(prompt.strip())
            metadata.append((shuffled_choices, correct_letter))

        outputs = gen.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            max_thinking_tokens=max_thinking_tokens,
            output_constraints=output_constraints,
        )

        for i, (output, (choices, correct_letter)) in enumerate(zip(outputs, metadata)):
            idx = batch_start + i
            answer_text = _extract_answer_after_thinking(output)
            valid = {chr(ord("A") + j) for j in range(len(choices))}
            predicted = _parse_choice(answer_text, valid, options=choices)

            if predicted is None:
                parse_failures += 1
            else:
                choice_counts[predicted] = choice_counts.get(predicted, 0) + 1
                if predicted == correct_letter:
                    correct += 1

            total += 1

        elapsed = time.time() - start_time
        speed = total / elapsed if elapsed > 0 else 0
        acc = correct / total * 100
        dist = _format_dist(choice_counts, total, parse_failures)
        print(
            f"  {total}/{n} | Acc: {acc:.1f}% "
            f"| Parse fails: {parse_failures} "
            f"| Dist: [{dist}] "
            f"| Speed: {speed:.1f} samples/s"
        )

    elapsed = time.time() - start_time
    accuracy = correct / total * 100 if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"ARC (Reasoning) Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(
        f"  Parse failures: {parse_failures}/{total} ({parse_failures/total*100:.1f}%)"
    )
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {total/elapsed:.1f} samples/s")
    print(f"{'='*60}")

    return accuracy


def _extract_number(text):
    """Extract a number from text. Tries structured patterns first, then falls back to first number."""
    import re

    _NUM = r"-?\d+(?:,\d{3})*(?:\.\d+)?"

    # Try #### pattern first (GSM8K format)
    m = re.search(rf"####\s*({_NUM})", text)
    if m:
        return m.group(1).replace(",", "")
    # Try "Answer: X" pattern (with optional markdown bold **)
    m = re.search(rf"\*?\*?[Aa]nswer:?\*?\*?\s*\$?\s*({_NUM})", text)
    if m:
        return m.group(1).replace(",", "")
    # Try "answer is X" / "equals X" / "= X" pattern
    m = re.search(
        rf"(?:answer is|equals?|=)\s*\$?\s*({_NUM})",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).replace(",", "")
    # Fallback: first number in text
    numbers = re.findall(rf"{_NUM}", text)
    if numbers:
        return numbers[0].replace(",", "")
    return None


def evaluate_gsm8k(
    gen,
    dataset,
    max_samples=None,
    max_new_tokens=512,
    batch_size=16,
    verbose_examples=5,
    max_thinking_tokens=None,
):
    """Evaluate on GSM8K using batched generative reasoning (numerical answer)."""
    import re

    correct = 0
    total = 0
    parse_failures = 0

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    n = len(dataset)
    print(
        f"Evaluating on {n} samples (batched generative reasoning, bs={batch_size})..."
    )
    start_time = time.time()

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = dataset.select(range(batch_start, batch_end))

        prompts = []
        ref_answers = []
        for example in batch:
            question = example["question"]
            answer = example["answer"]

            # Extract reference numerical answer (format: "#### 42")
            ref_match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", answer)
            ref_val = ref_match.group(1).replace(",", "") if ref_match else None
            ref_answers.append(ref_val)

            prompt = f"Solve the following math problem. Give your final numerical answer after your reasoning.\n\n{question}"
            # Truncate if too long
            model_maxlen = getattr(gen.model, "maxlen", 2048)
            max_prompt_tokens = model_maxlen // 2
            prompt_tokens = gen.tokenizer.encode(prompt)
            if len(prompt_tokens) > max_prompt_tokens:
                prompt = gen.tokenizer.decode(prompt_tokens[:max_prompt_tokens])
            prompts.append(prompt)

        outputs = gen.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            max_thinking_tokens=max_thinking_tokens,
        )

        for i, (output, ref_val) in enumerate(zip(outputs, ref_answers)):
            idx = batch_start + i
            answer_text = _extract_answer_after_thinking(output)
            predicted = _extract_number(answer_text)

            if idx < verbose_examples:
                print(f"  [DEBUG] idx={idx} predicted={predicted} ref={ref_val}")
                print(f"    Answer text: {answer_text[:200]!r}")

            if ref_val is None:
                parse_failures += 1
            elif predicted is None:
                parse_failures += 1
                if idx < verbose_examples:
                    print(f"  [PARSE FAIL] idx={idx}")
                    print(f"    Output: {output[:300]}")
            else:
                try:
                    if float(predicted) == float(ref_val):
                        correct += 1
                except ValueError:
                    pass

            total += 1

        elapsed = time.time() - start_time
        speed = total / elapsed if elapsed > 0 else 0
        acc = correct / total * 100
        print(
            f"  {total}/{n} | Acc: {acc:.1f}% "
            f"| Parse fails: {parse_failures} "
            f"| Speed: {speed:.1f} samples/s"
        )

    elapsed = time.time() - start_time
    accuracy = correct / total * 100 if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"GSM8K (Reasoning) Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(
        f"  Parse failures: {parse_failures}/{total} ({parse_failures/total*100:.1f}%)"
    )
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {total/elapsed:.1f} samples/s")
    print(f"{'='*60}")

    return accuracy


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Reasoning-based evaluation for instruction-tuned models"
    )
    parser.add_argument("--checkpoint", default="checkpoints/seqcond_torch_200k.pt")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="winogrande",
        help="Benchmark to evaluate on",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to evaluate (None for all)",
    )
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Max tokens to generate per sample (includes reasoning)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for parallel generation",
    )
    parser.add_argument(
        "--verbose_examples",
        type=int,
        default=10,
        help="Number of first examples to print detailed output for",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Deterministically shuffle dataset order (helps stabilize running accuracy)",
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Disable deterministic dataset shuffling",
    )
    parser.add_argument(
        "--max_thinking_tokens",
        type=int,
        default=1000,
        help="Absolute position limit: inject <|think_end|> when prompt_len + thinking_count >= this value",
    )
    parser.add_argument(
        "--debug_extract",
        action="store_true",
        help="Print _extract_answer_after_thinking decisions for the first verbose examples",
    )
    parser.add_argument(
        "--constrain_output",
        action="store_true",
        help="After thinking, force output to start with a valid choice letter (e.g. 'A.', 'B.')",
    )
    args = parser.parse_args()

    print("Loading model...")
    gen = TorchGenerator(args.checkpoint)
    gen.debug_extract = bool(args.debug_extract)

    def _make_constraints(n_choices):
        """Build output_constraints list like ['A.', 'B.', ...] if --constrain_output."""
        if not args.constrain_output:
            return None
        return [f"{chr(ord('A') + i)}." for i in range(n_choices)]

    if args.benchmark.startswith("winogrande"):
        parts = args.benchmark.split(":")
        subset = parts[1] if len(parts) == 2 else "winogrande_xl"
        if not subset.startswith("winogrande_"):
            subset = f"winogrande_{subset}"
        print(f"\nLoading Winogrande dataset (subset={subset}, split={args.split})...")
        dataset = load_dataset("allenai/winogrande", subset, split=args.split)
        dataset = _maybe_shuffle_dataset(
            dataset, seed=args.shuffle_seed, enabled=not args.no_shuffle
        )
        print(f"Dataset loaded: {len(dataset)} examples\n")
        accuracy = evaluate_winogrande(
            gen,
            dataset,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            verbose_examples=args.verbose_examples,
            max_thinking_tokens=args.max_thinking_tokens,
            output_constraints=_make_constraints(2),
        )
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
        _send_notification(
            "Évaluation terminée", f"Winogrande (reasoning): {accuracy:.2f}%"
        )
    elif args.benchmark == "openbookqa":
        print(f"\nLoading OpenBookQA dataset (split={args.split})...")
        dataset = load_dataset("openbookqa", "main", split=args.split)
        dataset = _maybe_shuffle_dataset(
            dataset, seed=args.shuffle_seed, enabled=not args.no_shuffle
        )
        print(f"Dataset loaded: {len(dataset)} examples\n")
        accuracy = evaluate_openbookqa(
            gen,
            dataset,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            verbose_examples=args.verbose_examples,
            max_thinking_tokens=args.max_thinking_tokens,
            output_constraints=_make_constraints(4),
        )
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
        _send_notification(
            "Évaluation terminée", f"OpenBookQA (reasoning): {accuracy:.2f}%"
        )
    elif args.benchmark == "commonsenseqa":
        print(f"\nLoading CommonsenseQA dataset (split={args.split})...")
        dataset = load_dataset("commonsense_qa", split=args.split)
        dataset = _maybe_shuffle_dataset(
            dataset, seed=args.shuffle_seed, enabled=not args.no_shuffle
        )
        print(f"Dataset loaded: {len(dataset)} examples\n")
        accuracy = evaluate_commonsenseqa(
            gen,
            dataset,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            verbose_examples=args.verbose_examples,
            max_thinking_tokens=args.max_thinking_tokens,
            output_constraints=_make_constraints(5),
        )
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
        _send_notification(
            "Évaluation terminée", f"CommonsenseQA (reasoning): {accuracy:.2f}%"
        )
    elif args.benchmark == "piqa":
        print(f"\nLoading PIQA dataset (split={args.split})...")
        dataset = load_dataset(
            "ybisk/piqa", split=args.split, revision="refs/convert/parquet"
        )
        dataset = _maybe_shuffle_dataset(
            dataset, seed=args.shuffle_seed, enabled=not args.no_shuffle
        )
        print(f"Dataset loaded: {len(dataset)} examples\n")
        accuracy = evaluate_piqa(
            gen,
            dataset,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            verbose_examples=args.verbose_examples,
            max_thinking_tokens=args.max_thinking_tokens,
            output_constraints=_make_constraints(2),
        )
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
        _send_notification("Évaluation terminée", f"PIQA (reasoning): {accuracy:.2f}%")
    elif args.benchmark == "hellaswag":
        print(f"\nLoading HellaSwag dataset (split={args.split})...")
        dataset = load_dataset("hellaswag", split=args.split)
        dataset = _maybe_shuffle_dataset(
            dataset, seed=args.shuffle_seed, enabled=not args.no_shuffle
        )
        print(f"Dataset loaded: {len(dataset)} examples\n")
        accuracy = evaluate_hellaswag(
            gen,
            dataset,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            verbose_examples=args.verbose_examples,
            max_thinking_tokens=args.max_thinking_tokens,
            output_constraints=_make_constraints(4),
        )
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
        _send_notification(
            "Évaluation terminée", f"HellaSwag (reasoning): {accuracy:.2f}%"
        )
    elif args.benchmark.startswith("arc"):
        parts = args.benchmark.split(":")
        if len(parts) == 2:
            subsets = [parts[1].capitalize()]
        else:
            subsets = ["Easy", "Challenge"]
        accuracies = {}
        for subset in subsets:
            print(f"\nLoading ARC-{subset} dataset (split={args.split})...")
            dataset = load_dataset("allenai/ai2_arc", f"ARC-{subset}", split=args.split)
            dataset = _maybe_shuffle_dataset(
                dataset, seed=args.shuffle_seed, enabled=not args.no_shuffle
            )
            print(f"Dataset loaded: {len(dataset)} examples\n")
            accuracy = evaluate_arc(
                gen,
                dataset,
                max_samples=args.max_samples,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                verbose_examples=args.verbose_examples,
                max_thinking_tokens=args.max_thinking_tokens,
                output_constraints=_make_constraints(4),
            )
            accuracies[subset] = accuracy
            print(f"\nARC-{subset} Final Accuracy: {accuracy:.2f}%")
            _send_notification(
                "Évaluation terminée", f"ARC-{subset} (reasoning): {accuracy:.2f}%"
            )
        if len(accuracies) > 1:
            avg_accuracy = sum(accuracies.values()) / len(accuracies)
            print(f"\n{'='*60}")
            print(f"ARC (average): {avg_accuracy:.2f}%")
            print(f"{'='*60}")
            _send_notification("Évaluation terminée", f"ARC (avg): {avg_accuracy:.2f}%")
    elif args.benchmark == "gsm8k":
        print(f"\nLoading GSM8K dataset (split={args.split})...")
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        dataset = _maybe_shuffle_dataset(
            dataset, seed=args.shuffle_seed, enabled=not args.no_shuffle
        )
        print(f"Dataset loaded: {len(dataset)} examples\n")
        accuracy = evaluate_gsm8k(
            gen,
            dataset,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            verbose_examples=args.verbose_examples,
            max_thinking_tokens=args.max_thinking_tokens,
        )
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
        _send_notification("Évaluation terminée", f"GSM8K (reasoning): {accuracy:.2f}%")
    elif args.benchmark.startswith("gpqa"):
        parts = args.benchmark.split(":")
        subset = parts[1] if len(parts) == 2 else "diamond"
        subset = f"gpqa_{subset}"
        print(f"\nLoading GPQA dataset (subset={subset}, split=train)...")
        try:
            import os

            token = os.environ.get("HF_TOKEN", True)
            dataset = load_dataset(
                "Idavidrein/gpqa", subset, split="train", token=token
            )
            dataset = _maybe_shuffle_dataset(
                dataset, seed=args.shuffle_seed, enabled=not args.no_shuffle
            )
        except Exception as e:
            print(f"Failed to load GPQA: {e}")
            print(
                "GPQA requires authentication. Set HF_TOKEN or run `huggingface-cli login`."
            )
            return
        print(f"Dataset loaded: {len(dataset)} examples\n")
        accuracy = evaluate_gpqa(
            gen,
            dataset,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            verbose_examples=args.verbose_examples,
            max_thinking_tokens=args.max_thinking_tokens,
            output_constraints=_make_constraints(4),
        )
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
        _send_notification("Évaluation terminée", f"GPQA (reasoning): {accuracy:.2f}%")
    else:
        print(f"Benchmark '{args.benchmark}' not implemented yet in eval_reasoning.py")


if __name__ == "__main__":
    main()

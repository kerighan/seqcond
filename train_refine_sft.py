"""
train_refine_sft.py — SFT online avec reformulation LLM des générations imparfaites.

Pour chaque batch de questions :
  1. Génère G completions avec le modèle courant (policy actuelle)
  2. Score chaque completion
  3. Les imparfaites (reward < 1.0) sont reformulées via LLM (changements minimes)
  4. Affiche BEFORE / AFTER pour chaque reformulation
  5. Gradient step SFT sur les completions finalement correctes

Usage:
    python train_refine_sft.py \\
        --checkpoint checkpoints/seqcond_lin5.pt \\
        --dataset gsm8k \\
        --openai_api_key $OPENAI_API_KEY
"""
import argparse, asyncio, copy, difflib, json, os, random, time
import numpy as np
import torch

from openai import AsyncOpenAI

from seqcond.torch.generator import TorchGenerator
from train_grpo import load_gsm8k, _compute_advantages
from collect_cot import DATASETS, _make_local_math_loader
from train_sft import (
    gradient_step, _set_trainable_parameters, save_ckpt,
    eval_pass1, _check_answer_for_dataset,
    _CACHE_PREFIXES,
)

W = 88
def sep(c="─"): print(c * W)
def dsep():     print("═" * W)


# ── Refiner ───────────────────────────────────────────────────────────────────

REFINER_SYSTEM = """\
You are a minimal editor. A model produced a reasoning trace that led to the WRONG answer.
Your job: find the FIRST incorrect computation or logical step, fix it, and propagate the correction to subsequent steps if needed.

STRICT RULES:
- Preserve the writing style exactly: vocabulary, sentence length, structure, tone, formatting symbols (●, →, ∴, ※, etc.)
- Change ONLY the specific step(s) that cause the wrong answer — leave everything else verbatim
- Do NOT add new steps, do NOT expand reasoning, do NOT restructure
- The result must be indistinguishable in style from the original
- The corrected trace MUST end with the correct answer (given in the user message)
- If you cannot find a minimal fix that reaches the correct answer, output the original trace unchanged
- Output ONLY the corrected reasoning trace, nothing else — no preamble, no commentary, no "Here is the corrected..." prefix"""


async def _refine_one(client, refine_model, question, text, ground_truth):
    resp = await client.chat.completions.create(
        model=refine_model,
        messages=[
            {"role": "system", "content": REFINER_SYSTEM},
            {"role": "user", "content": (
                f"QUESTION:\n{question}\n\n"
                f"CORRECT ANSWER: {ground_truth}\n"
                f"(The fixed trace must reach exactly this answer.)\n\n"
                f"REASONING TO FIX:\n{text}"
            )},
        ],
        temperature=0.1,
        max_tokens=2000,
    )
    return resp.choices[0].message.content.strip()


async def _refine_batch_async(api_key, refine_model, items):
    async with AsyncOpenAI(api_key=api_key) as client:
        tasks = [_refine_one(client, refine_model, q, t, gt) for q, t, gt in items]
        return await asyncio.gather(*tasks)


def _print_before_after(label, before, after):
    sep()
    print(f"  ── {label} ──")
    sep()
    print("BEFORE:")
    print(before[:700] + ("…" if len(before) > 700 else ""))
    sep("·")
    print("AFTER:")
    print(after[:700] + ("…" if len(after) > 700 else ""))
    sep(); print()


# ── Tokenisation ──────────────────────────────────────────────────────────────

def _make_sft_prompt(ex, use_synth_template):
    """Reconstruit le prompt formaté (se termine par <|think_start|>)."""
    if use_synth_template:
        return (
            "<|im_start|>user\n" + ex["prompt"]
            + "\n<|im_end|><|im_start|>assistant\n<|think_start|>"
        )
    else:
        # load_gsm8k inclut déjà le template complet
        return ex["prompt"]


def _tokenize_completion(tokenizer, sft_prompt, completion_text, weight=1.0, max_seq_len=4096):
    """Retourne un item SFT {ids, pl, weight} ou None si trop long."""
    completion = completion_text.strip()
    if not completion.endswith("<|im_end|>"):
        completion = completion + "\n<|im_end|>"
    prompt_ids = tokenizer([sft_prompt])[0]
    full_ids   = tokenizer([sft_prompt + completion])[0]
    comp_ids   = full_ids[len(prompt_ids):]
    if not comp_ids:
        return None
    if max_seq_len > 0 and len(full_ids) > max_seq_len:
        return None
    return {"ids": prompt_ids + comp_ids, "pl": len(prompt_ids), "weight": weight}


# ── Eval helpers ──────────────────────────────────────────────────────────────

def _build_eval_examples(examples):
    seen = set()
    out  = []
    for ex in examples:
        key = ex.get("question", ex["prompt"])
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "question":     key,
            "ground_truth": ex["ground_truth"],
            "dataset":      ex.get("dataset", "gsm8k"),
            "prompt":       ex["prompt"],   # used by eval_pass1
        })
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     default="./checkpoints/seqcond_lin5.pt")
    p.add_argument("--dataset",        default="gsm8k",
                   choices=[k for k in DATASETS if DATASETS[k].get("score")])
    p.add_argument("--split",          default="train")
    p.add_argument("--max_examples",   type=int,   default=None)
    p.add_argument("--g",              type=int,   default=8,    help="completions par question")
    p.add_argument("--temperature",    type=float, default=1)
    p.add_argument("--max_tokens",     type=int,   default=1000)
    p.add_argument("--gen_batch",      type=int,   default=4,    help="questions par appel generate_batch")
    p.add_argument("--grad_batch",     type=int,   default=16,   help="items SFT par gradient step")
    p.add_argument("--lr",             type=float, default=3e-6)
    p.add_argument("--epochs",         type=float, default=1.0)
    p.add_argument("--kl_beta",        type=float, default=0.01, help="poids KL vs modèle initial (0=désactivé)")
    p.add_argument("--max_edit_ratio", type=float, default=0.2,
                   help="ratio max d'édition orig→raffiné accepté via SequenceMatcher (0=désactivé, défaut=0.35)")
    p.add_argument("--train",          default="all")
    p.add_argument("--eval_every",     type=int,   default=50)
    p.add_argument("--eval_n",         type=int,   default=200)
    p.add_argument("--eval_batch",     type=int,   default=16)
    p.add_argument("--save_every",     type=int,   default=1)
    p.add_argument("--save_path",      default=None)
    p.add_argument("--refine_model",   default="gpt-4.1-mini")
    p.add_argument("--openai_api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--local_file",     default=None)
    p.add_argument("--mixed_bfloat16", action="store_true")
    p.add_argument("--seed",           type=int,   default=42)
    args = p.parse_args()

    if not args.openai_api_key:
        raise ValueError("--openai_api_key (or $OPENAI_API_KEY) requis")

    if args.save_path is None:
        base = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.save_path = os.path.join("checkpoints", "refine_sft", f"{base}_refine_sft.pt")

    if args.dataset == "local_math":
        if not args.local_file:
            raise ValueError("--local_file requis pour local_math")
        DATASETS["local_math"]["load"] = _make_local_math_loader(args.local_file)

    random.seed(args.seed); np.random.seed(args.seed)

    ds_cfg             = DATASETS[args.dataset]
    score_fn           = ds_cfg["score"]
    use_synth_template = ds_cfg["use_synth_template"]

    # ── Modèle ────────────────────────────────────────────────────────────────
    print(f"Loading {args.checkpoint} …")
    torch_gen = TorchGenerator(args.checkpoint, device="cuda")
    model     = torch_gen.model
    tokenizer = torch_gen.tokenizer
    config    = torch_gen.config

    ref_model = None
    if args.kl_beta > 0:
        ref_model = copy.deepcopy(model).cuda().eval()
        for p_ in ref_model.parameters():
            p_.requires_grad = False
        print(f"  KL ref model chargé  β={args.kl_beta}")

    unfrozen = _set_trainable_parameters(model, args.train)
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    n_trainable = sum(p_.numel() for p_ in trainable)
    print(f"  {n_trainable:,} paramètres entraînables")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.99))

    # ── Dataset ───────────────────────────────────────────────────────────────
    examples = ds_cfg["load"](args.split, args.seed, args.max_examples)

    # Déduplique par question (utile quand le JSONL vient de collect_cot avec G>1)
    seen, unique = set(), []
    for ex in examples:
        key = ex.get("question", ex["prompt"])
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    if len(unique) < len(examples):
        print(f"  déduplication : {len(examples)} → {len(unique)} questions uniques")
    examples = unique

    print(f"  dataset={args.dataset}  {len(examples)} exemples  G={args.g}  temp={args.temperature}")

    eval_examples = []
    if args.eval_n > 0:
        test_data = load_gsm8k(split="test", seed=args.seed)
        eval_examples = test_data[:args.eval_n]

    # Shuffle pour N epochs
    data_buf = []
    for _ in range(max(1, int(np.ceil(args.epochs)))):
        epoch = list(examples)
        random.shuffle(epoch)
        data_buf.extend(epoch)

    G = args.g
    n_q = len(data_buf)
    # Nombre de gradient steps ≈ questions traitées / grad_batch (approx, dépend du yield)
    total_q = n_q
    print(f"  {total_q} questions  gen_batch={args.gen_batch}  grad_batch={args.grad_batch}")

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # ── Baseline eval ─────────────────────────────────────────────────────────
    if eval_examples:
        dsep()
        print(f"  BASELINE EVAL  test_n={args.eval_n}")
        sep()
        t_eval = time.perf_counter()
        baseline_test = eval_pass1(torch_gen, eval_examples, args.max_tokens, args.eval_batch)
        print(f"  baseline_test={baseline_test:.4f}  ({time.perf_counter()-t_eval:.0f}s)")
        dsep(); print()
        prev_test = baseline_test
    else:
        baseline_test = prev_test = None

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    sft_buf  = []          # accumulateur d'items SFT entre gradient steps
    step     = 0
    eval_count = 0
    step_losses, step_gnorms = [], []
    n_refined = n_gain = n_skipped = 0
    t0 = time.perf_counter()

    for q_start in range(0, total_q, args.gen_batch):
        q_batch = data_buf[q_start : q_start + args.gen_batch]

        # ── 1. Génération ──────────────────────────────────────────────────
        model.eval()
        flat_prompts = [ex["prompt"] for ex in q_batch for _ in range(G)]
        flat_texts = torch_gen.generate_batch(
            flat_prompts,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            use_synth_template=use_synth_template,
            top_k=10,
        )

        # ── 2. Score ───────────────────────────────────────────────────────
        groups = []
        to_refine = []   # (group_idx, local_idx, question, text, gt)

        for j, ex in enumerate(q_batch):
            texts   = list(flat_texts[j * G : (j + 1) * G])
            rewards = [score_fn(t, ex) for t in texts]
            groups.append({"ex": ex, "texts": texts, "rewards": rewards})

            q_text = ex.get("question", ex["prompt"])
            gt     = ex["ground_truth"]
            for k, (t, r) in enumerate(zip(texts, rewards)):
                if r < 1.0:
                    to_refine.append((j, k, q_text, t, gt))

        # ── 3. Reformulation LLM ───────────────────────────────────────────
        if to_refine:
            items_llm = [(q_, t_, gt_) for _, _, q_, t_, gt_ in to_refine]
            refined_texts = asyncio.run(
                _refine_batch_async(args.openai_api_key, args.refine_model, items_llm)
            )

            for (j, k, q_text, orig, gt), refined in zip(to_refine, refined_texts):
                edit_ratio = 1.0 - difflib.SequenceMatcher(
                    None, orig, refined, autojunk=False
                ).ratio()
                off_policy = args.max_edit_ratio > 0 and edit_ratio > args.max_edit_ratio
                # Restore <|think_end|> + answer suffix if GPT dropped it
                if "<|think_end|>" in orig and "<|think_end|>" not in refined:
                    orig_suffix = orig.split("<|think_end|>", 1)[1]
                    refined = refined.rstrip() + "<|think_end|>" + orig_suffix

                n_refined += 1
                if off_policy:
                    n_skipped += 1
                    continue
                new_r = score_fn(refined, groups[j]["ex"])
                if new_r > groups[j]["rewards"][k]:
                    n_gain += 1
                groups[j]["texts"][k]   = refined
                groups[j]["rewards"][k] = new_r

        # ── 4. Tokenise et accumule ────────────────────────────────────────
        for grp in groups:
            ex, texts, rewards = grp["ex"], grp["texts"], grp["rewards"]
            sft_prompt = _make_sft_prompt(ex, use_synth_template)
            for t, r in zip(texts, rewards):
                if r == 1.0:
                    item = _tokenize_completion(tokenizer, sft_prompt, t)
                    if item is not None:
                        sft_buf.append(item)

        # ── 5. Gradient step dès qu'on a assez d'items ────────────────────
        while len(sft_buf) >= args.grad_batch:
            batch = sft_buf[:args.grad_batch]
            sft_buf = sft_buf[args.grad_batch:]

            step += 1
            model.train()
            loss, gnorm = gradient_step(
                model, optimizer, trainable, batch,
                kl_beta=args.kl_beta, ref_model=ref_model,
                use_mixed_bfloat16=args.mixed_bfloat16,
            )
            step_losses.append(loss); step_gnorms.append(gnorm)

            elapsed = time.perf_counter() - t0
            lr_cur  = optimizer.param_groups[0]["lr"]
            print(
                f"step {step:5d}  loss={loss:.5f}  gnorm={gnorm:.3f}  "
                f"refined={n_refined}  gain={n_gain}  "
                f"kept={100*(n_refined-n_skipped)/max(1,n_refined):.0f}%  "
                f"{elapsed:.0f}s  lr={lr_cur:.2e}",
                flush=True,
            )

            if args.eval_every > 0 and step % args.eval_every == 0 and eval_examples:
                sep()
                t_eval = time.perf_counter()
                test_score = eval_pass1(torch_gen, eval_examples, args.max_tokens, args.eval_batch)
                dtest = test_score - prev_test if prev_test is not None else 0.0
                dsep()
                print(
                    f"  EVAL step={step}  "
                    f"test={test_score:.4f} (Δ={dtest:+.4f})  "
                    f"(baseline={baseline_test:.4f})  {time.perf_counter()-t_eval:.0f}s"
                )
                dsep(); print()
                prev_test = test_score
                eval_count += 1

                if args.save_every > 0 and eval_count % args.save_every == 0:
                    base, ext = os.path.splitext(args.save_path)
                    save_ckpt(torch_gen, config, f"{base}_step{step:05d}{ext}", step)

    # ── Final eval + save ──────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    dsep()
    if eval_examples:
        final_test = eval_pass1(torch_gen, eval_examples, args.max_tokens, args.eval_batch)
        print(
            f"  FINAL  test={final_test:.4f}  Δtest={final_test - baseline_test:+.4f}  "
            f"(baseline={baseline_test:.4f})  steps={step}  {elapsed:.0f}s"
        )
    else:
        print(f"  FINAL  steps={step}  refined={n_refined}  gain={n_gain}  {elapsed:.0f}s")
    dsep()
    save_ckpt(torch_gen, config, args.save_path, step)


if __name__ == "__main__":
    main()

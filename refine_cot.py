"""
refine_cot.py — Génère par groupe et raffine via LLM (scaffolding minimal).

Pipeline par exemple :
  1. Génère G completions avec le modèle local
  2. Score chaque completion
  3. Pour les completions imparfaites (reward < 1.0), demande à un LLM de
     les corriger avec des changements MINIMES — rester au plus près de la
     policy originale, juste corriger l'erreur
  4. Affiche BEFORE / AFTER pour chaque reformulation
  5. Écrit dans un JSONL les completions correctes (originales ou raffinées)

Usage:
    python refine_cot.py \\
        --checkpoint checkpoints/seqcond_lin5.pt \\
        --dataset gsm8k \\
        --openai_api_key $OPENAI_API_KEY
"""
import argparse, asyncio, json, os, time

from openai import AsyncOpenAI

from seqcond.torch.generator import TorchGenerator
from train_grpo import _compute_advantages
from collect_cot import DATASETS, _make_local_math_loader

W = 88
def sep(c="─"): print(c * W)


REFINER_SYSTEM = """\
You are a minimal editor. A model produced a reasoning trace that led to the WRONG answer.
Make the ABSOLUTE MINIMUM edits so the reasoning arrives at the correct answer.

STRICT RULES:
- Preserve the writing style exactly: vocabulary, sentence length, structure, tone, formatting
- Change ONLY the specific step(s) that cause the wrong answer — leave everything else verbatim
- Do NOT add new steps, do NOT expand reasoning, do NOT restructure
- The result must be indistinguishable in style from the original
- Output ONLY the corrected reasoning trace, nothing else — no preamble, no commentary"""


async def _refine_one(client, refine_model, question, text, ground_truth):
    resp = await client.chat.completions.create(
        model=refine_model,
        messages=[
            {"role": "system", "content": REFINER_SYSTEM},
            {"role": "user", "content": (
                f"QUESTION:\n{question}\n\n"
                f"CORRECT ANSWER: {ground_truth}\n\n"
                f"REASONING TO FIX:\n{text}"
            )},
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    return resp.choices[0].message.content.strip()


async def _refine_batch(api_key, refine_model, items):
    """items: list of (question, text, ground_truth)"""
    async with AsyncOpenAI(api_key=api_key) as client:
        tasks = [_refine_one(client, refine_model, q, t, gt) for q, t, gt in items]
        return await asyncio.gather(*tasks)


def _print_before_after(label, before, after):
    sep()
    print(f"  ── {label} ──")
    sep()
    print("BEFORE:")
    print(before[:800] + ("…" if len(before) > 800 else ""))
    sep("·")
    print("AFTER:")
    print(after[:800] + ("…" if len(after) > 800 else ""))
    sep()
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     default="./checkpoints/seqcond_lin5.pt")
    p.add_argument("--dataset",        default="gsm8k",
                   choices=[k for k in DATASETS if DATASETS[k].get("score")])
    p.add_argument("--output",         default=None)
    p.add_argument("--split",          default="train")
    p.add_argument("--max_examples",   type=int,   default=None)
    p.add_argument("--g",              type=int,   default=8,   help="completions par exemple")
    p.add_argument("--temperature",    type=float, default=0.4)
    p.add_argument("--max_tokens",     type=int,   default=1000)
    p.add_argument("--batch_size",     type=int,   default=4,   help="exemples en parallèle (génération)")
    p.add_argument("--refine_model",   default="gpt-4.1-mini")
    p.add_argument("--openai_api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--local_file",     default=None)
    p.add_argument("--seed",           type=int,   default=24)
    args = p.parse_args()

    if args.output is None:
        args.output = f"./data/{args.dataset}_refined.jsonl"

    if args.dataset == "local_math":
        if not args.local_file:
            raise ValueError("--local_file requis pour local_math")
        DATASETS["local_math"]["load"] = _make_local_math_loader(args.local_file)

    if not args.openai_api_key:
        raise ValueError("--openai_api_key (or $OPENAI_API_KEY) requis")

    ds_cfg             = DATASETS[args.dataset]
    score_fn           = ds_cfg["score"]
    record_fn          = ds_cfg["record"]
    use_synth_template = ds_cfg["use_synth_template"]

    print(f"Loading {args.checkpoint} …")
    gen = TorchGenerator(args.checkpoint, device="cuda")
    examples = ds_cfg["load"](args.split, args.seed, args.max_examples)
    print(f"  dataset={args.dataset}  {len(examples)} exemples  G={args.g}  temp={args.temperature}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    G = args.g
    n_written = n_refined = n_gain = 0
    t0 = time.perf_counter()

    with open(args.output, "w") as out_f:
        for batch_start in range(0, len(examples), args.batch_size):
            batch = examples[batch_start : batch_start + args.batch_size]

            # ── 1. Génération ─────────────────────────────────────────────────
            flat_prompts = [ex["prompt"] for ex in batch for _ in range(G)]
            flat_texts = gen.generate_batch(
                flat_prompts,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                use_synth_template=use_synth_template,
            )

            # ── 2. Score initial ──────────────────────────────────────────────
            groups = []
            to_refine = []   # (group_idx, local_idx, question, text, gt)

            for j, ex in enumerate(batch):
                texts   = list(flat_texts[j * G : (j + 1) * G])
                rewards = [score_fn(t, ex) for t in texts]
                groups.append({"ex": ex, "texts": texts, "rewards": rewards})

                q  = ex.get("question", ex["prompt"])
                gt = ex["ground_truth"]
                for k, (t, r) in enumerate(zip(texts, rewards)):
                    if r < 1.0:
                        to_refine.append((j, k, q, t, gt))

            # ── 3. Reformulation LLM (async, en une passe) ───────────────────
            if to_refine:
                items_for_llm = [(q, t, gt) for _, _, q, t, gt in to_refine]
                refined_texts = asyncio.run(
                    _refine_batch(args.openai_api_key, args.refine_model, items_for_llm)
                )

                for (j, k, q, orig, gt), refined in zip(to_refine, refined_texts):
                    _print_before_after(
                        f"ex={batch_start + j}  k={k}  gt={gt}",
                        orig, refined,
                    )
                    new_r = score_fn(refined, groups[j]["ex"])
                    if new_r > groups[j]["rewards"][k]:
                        n_gain += 1
                    groups[j]["texts"][k]   = refined
                    groups[j]["rewards"][k] = new_r
                    n_refined += 1

            # ── 4. Écriture des completions correctes ─────────────────────────
            for grp in groups:
                ex, texts, rewards = grp["ex"], grp["texts"], grp["rewards"]
                advantages = list(_compute_advantages(rewards, normalize_std=False))
                for t, r, a in zip(texts, rewards, advantages):
                    if r == 1.0:
                        out_f.write(json.dumps(
                            record_fn(ex, t, r, a, "refined"), ensure_ascii=False
                        ) + "\n")
                        n_written += 1

            n_done  = min(batch_start + args.batch_size, len(examples))
            elapsed = time.perf_counter() - t0
            print(
                f"  [{n_done:4d}/{len(examples)}]  "
                f"written={n_written}  refined={n_refined}  gain={n_gain}  {elapsed:.0f}s",
                flush=True,
            )
            out_f.flush()

    elapsed = time.perf_counter() - t0
    print(f"\nTerminé en {elapsed:.0f}s  →  {n_written} entrées  →  {args.output}")
    print(f"  refined={n_refined}  gain (reward↑ après refine)={n_gain}")


if __name__ == "__main__":
    main()

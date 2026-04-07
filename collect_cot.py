"""
collect_cot.py — Génère un dataset JSONL de CoT réussies pour chaque exemple GSM8K.

Pour chaque exemple :
  - Génère G completions
  - Si ≥1 correcte → garde toutes les correctes
  - Si 0 correcte  → garde celles dont la réponse est dans ±10% du ground truth
  - Calcule reward et advantage dans le groupe retenu
  - Écrit dans le JSONL

Usage:
    python collect_cot.py --checkpoint checkpoints/seqcond_lin5.pt --output data/cot_dataset.jsonl
"""
import argparse, json, os, re, time
import numpy as np

from seqcond.torch.generator import TorchGenerator
from train_grpo import load_gsm8k, check_answer, _compute_advantages


def extract_number(text: str) -> float | None:
    """Extrait le meilleur candidat numérique après <|think_end|> (ou dans tout le texte)."""
    parts = text.split("<|think_end|>")
    zone = parts[-1] if len(parts) > 1 else text
    candidates = re.findall(r"-?[\d,]+\.?\d*", zone)
    if not candidates:
        candidates = re.findall(r"-?[\d,]+\.?\d*", text)
    for c in reversed(candidates):  # le dernier nombre est souvent la réponse finale
        try:
            return float(c.replace(",", ""))
        except ValueError:
            continue
    return None


def within_pct(text: str, ground_truth: str, pct: float = 0.10) -> bool:
    """True si la réponse extraite est dans ±pct du ground truth."""
    try:
        gt = float(ground_truth.replace(",", ""))
    except ValueError:
        return False
    if gt == 0:
        return False
    pred = extract_number(text)
    if pred is None:
        return False
    return abs(pred - gt) / abs(gt) <= pct


def score_text(text: str, ground_truth: str) -> float:
    """Reward : 1.0 si correct, 0.5 si dans ±10%, sinon 0."""
    if check_answer(text, ground_truth):
        return 1.0
    if within_pct(text, ground_truth, pct=0.10):
        return 0.5
    return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="./checkpoints/seqcond_lin5.pt")
    p.add_argument("--output",      default="./data/cot_dataset.jsonl")
    p.add_argument("--split",       default="train",   help="train ou test")
    p.add_argument("--max_examples",type=int, default=None, help="limiter le nombre d'exemples")
    p.add_argument("--g",           type=int, default=8,    help="completions par exemple")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--max_tokens",  type=int, default=1000)
    p.add_argument("--batch_size",  type=int, default=8,    help="prompts en parallèle")
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading {args.checkpoint} …")
    gen = TorchGenerator(args.checkpoint, device="cuda")

    examples = load_gsm8k(split=args.split, seed=args.seed)
    if args.max_examples:
        examples = examples[:args.max_examples]
    print(f"  {len(examples)} exemples  G={args.g}  temp={args.temperature}")

    n_written = 0
    n_correct_groups = 0
    n_partial_groups = 0
    n_empty_groups   = 0
    t0 = time.perf_counter()

    with open(args.output, "w") as f:
        for i, ex in enumerate(examples):
            # ── Génération ────────────────────────────────────────────────────
            texts, _ = gen.generate_group(
                ex["prompt"], n=args.g,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                use_synth_template=False,
            )

            # ── Scoring ───────────────────────────────────────────────────────
            rewards = [score_text(t, ex["ground_truth"]) for t in texts]
            has_correct = any(r == 1.0 for r in rewards)

            if has_correct:
                # Garde uniquement les correctes
                kept = [(t, r) for t, r in zip(texts, rewards) if r == 1.0]
                group_type = "correct"
                n_correct_groups += 1
            else:
                # Fallback : ±10%
                kept = [(t, r) for t, r in zip(texts, rewards) if r > 0]
                group_type = "partial"
                if kept:
                    n_partial_groups += 1
                else:
                    n_empty_groups += 1

            if not kept:
                continue

            # ── Avantages dans le groupe retenu ───────────────────────────────
            kept_rewards = [r for _, r in kept]
            advantages   = list(_compute_advantages(kept_rewards, normalize_std=False))

            # ── Écriture JSONL ────────────────────────────────────────────────
            for (text, reward), adv in zip(kept, advantages):
                record = {
                    "query":        ex["question"],
                    "reasoning":    text,
                    "ground_truth": ex["ground_truth"],
                    "reward":       round(reward, 4),
                    "advantage":    round(float(adv), 4),
                    "group_type":   group_type,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

            # ── Progress ──────────────────────────────────────────────────────
            if (i + 1) % 50 == 0 or (i + 1) == len(examples):
                elapsed = time.perf_counter() - t0
                pct_correct = 100 * n_correct_groups / (i + 1)
                pct_partial = 100 * n_partial_groups / (i + 1)
                pct_empty   = 100 * n_empty_groups   / (i + 1)
                print(f"  [{i+1:5d}/{len(examples)}]  "
                      f"correct={pct_correct:.1f}%  partial={pct_partial:.1f}%  "
                      f"empty={pct_empty:.1f}%  "
                      f"written={n_written}  {elapsed:.0f}s")
                f.flush()

    elapsed = time.perf_counter() - t0
    print(f"\nTerminé en {elapsed:.0f}s")
    print(f"  {n_written} entrées écrites → {args.output}")
    print(f"  groupes : correct={n_correct_groups}  partial={n_partial_groups}  "
          f"vides (ignorés)={n_empty_groups}")


if __name__ == "__main__":
    main()

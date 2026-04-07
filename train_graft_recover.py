"""
train_rft.py — One batch: generate → score → gradient step → verify.

Usage:
    KERAS_BACKEND=torch python train_rft.py --checkpoint checkpoints/seqcond_lin5.pt --n 4 --g 16
"""
import argparse, os, random, textwrap
import numpy as np
import torch

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras import ops

from seqcond.torch.generator import TorchGenerator
from convert_torch_to_keras import (
    build_keras_model, convert_weights, load_torch_checkpoint,
)
from train_grpo import (
    load_gsm8k, check_answer,
    _compute_advantages, _repetition_penalty,
    _seq_token_log_probs, sync_keras_to_torch,
)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_train_test(seed=42):
    """Load GSM8K; return (train_examples, test_examples) shuffled with seed."""
    train = load_gsm8k(split="train", seed=seed)
    test  = load_gsm8k(split="test",  seed=seed)
    return train, test


# ── Generation ────────────────────────────────────────────────────────────────

def generate_groups(torch_gen: TorchGenerator, examples, G, max_tokens, temperature):
    """Generate G completions for each example using TorchGenerator.generate_group.

    Returns list of dicts:
        {
            "example":       ex,
            "prompt_tokens": List[int],        # prompt token ids
            "texts":         List[str],         # G decoded completions
            "comp_ids":      List[List[int]],   # G completion token id lists
        }
    """
    results = []
    for ex in examples:
        prompt_tokens = torch_gen.tokenizer([ex["prompt"]])[0]
        texts, comp_ids = torch_gen.generate_group(
            ex["prompt"],
            n=G,
            max_new_tokens=max_tokens,
            temperature=temperature,
            use_synth_template=False,  # prompt already formatted by load_gsm8k
        )
        results.append({
            "example":       ex,
            "prompt_tokens": prompt_tokens,
            "texts":         texts,
            "comp_ids":      comp_ids,
        })
    return results


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_group(group, conciseness_alpha=0.05):
    """Binary reward 1/0, conciseness-scaled for correct completions.

    Correct completions shorter than the group mean get a bonus:
        r *= exp(-alpha * (len - mu) / sigma)
    """
    texts, comp_ids, gt = group["texts"], group["comp_ids"], group["example"]["ground_truth"]
    binary  = [1.0 if check_answer(t, gt) else 0.0 for t in texts]
    rewards = list(binary)

    # Repetition penalty
    for i, ids in enumerate(comp_ids):
        pen = _repetition_penalty(ids)
        if pen:
            rewards[i] += pen

    # Conciseness scaling among correct completions
    correct_lens = [len(comp_ids[i]) for i, b in enumerate(binary) if b > 0]
    if len(correct_lens) >= 2:
        mu    = np.mean(correct_lens)
        sigma = max(np.std(correct_lens), 1.0)
        for i, b in enumerate(binary):
            if b > 0:
                z = (len(comp_ids[i]) - mu) / sigma
                rewards[i] *= np.exp(-conciseness_alpha * z)

    return rewards, binary


# ── Selection ─────────────────────────────────────────────────────────────────

def select_pairs(texts, comp_ids, advantages, binary):
    """Return (positives, negatives) sorted by |advantage| descending.

    positives: list of (idx, text, comp_tokens, adv)
    negatives: list of (idx, text, comp_tokens, adv)   — all, sorted by |adv|
    """
    positives = sorted(
        [(i, texts[i], comp_ids[i], advantages[i])
         for i, b in enumerate(binary) if b > 0],
        key=lambda x: x[3], reverse=True,
    )
    negatives = sorted(
        [(i, texts[i], comp_ids[i], advantages[i])
         for i, a in enumerate(advantages) if a < 0],
        key=lambda x: abs(x[3]), reverse=True,
    )
    return positives, negatives


# ── Log-prob helpers ──────────────────────────────────────────────────────────

def seq_avg_logprob(model, prompt_tokens, comp_tokens):
    """Mean per-token log-prob for comp_tokens (no grad, purely cosmetic)."""
    import torch
    ids = np.array([prompt_tokens + comp_tokens], dtype=np.int32)
    with torch.no_grad():
        token_lps = _seq_token_log_probs(model, ids, len(prompt_tokens))
    return float(ops.mean(token_lps))


# ── Gradient step ─────────────────────────────────────────────────────────────

def _apply_gradient(keras_model, optimizer, scored_groups):
    optimizer.zero_grad()
    n_terms = sum(len(g["positives"]) + len(g["negatives"]) for g in scored_groups)
    
    total_loss_val = 0.0
    
    for g in scored_groups:
        pt = g["prompt_tokens"]
        pl = len(pt)
        for (_, _, comp, adv) in g["positives"] + g["negatives"]:
            if not comp: continue
            
            ids = np.array([pt + comp], dtype=np.int32)
            token_lps = _seq_token_log_probs(keras_model, ids, pl)
            
            # On calcule la perte pour CETTE séquence uniquement
            loss = (-adv * ops.mean(token_lps)) / n_terms
            
            # .backward() ICI libère les activations de cette séquence 
            # mais accumule le GRADIENT dans les paramètres du modèle
            loss.backward() 
            
            total_loss_val += float(loss)

    # Une fois que toutes les séquences ont été traitées :
    optimizer.step() 
    return total_loss_val


def gradient_step(keras_model, optimizer, scored_groups, sft_steps=1):
    """sft_steps gradient updates on the same pos/neg batch.

    CE is recomputed at each step (no IS ratio — fine for on-policy SFT).
    lp_before/after computed once (before first step, after last step).
    Returns lps_before, lps_after — dicts keyed by (group_idx, seq_idx).
    """
    # ── Before log-probs (no grad) ────────────────────────────────────────────
    lps_before = {}
    for gi, g in enumerate(scored_groups):
        pt = g["prompt_tokens"]
        for (idx, _, comp, _) in g["positives"] + g["negatives"]:
            lps_before[(gi, idx)] = seq_avg_logprob(keras_model, pt, comp)

    # ── sft_steps gradient updates ────────────────────────────────────────────
    for step_i in range(sft_steps):
        loss = _apply_gradient(keras_model, optimizer, scored_groups)
        print(f"  gradient step {step_i + 1}/{sft_steps}  loss={loss:.4f}")

    # ── After log-probs (no grad) ─────────────────────────────────────────────
    lps_after = {}
    for gi, g in enumerate(scored_groups):
        pt = g["prompt_tokens"]
        for (idx, _, comp, _) in g["positives"] + g["negatives"]:
            lps_after[(gi, idx)] = seq_avg_logprob(keras_model, pt, comp)

    return lps_before, lps_after


# ── Pretty print ──────────────────────────────────────────────────────────────

W = 88

def box(text, width=W):
    return "\n".join("  " + l for l in textwrap.wrap(text, width - 2))

def separator(char="─"):
    print(char * W)

def lp_tag(lp_b, lp_a, adv):
    """Format log-prob delta with correctness indicator."""
    delta = lp_a - lp_b
    ok = (delta > 0 and adv > 0) or (delta < 0 and adv < 0)
    return f"lp: {lp_b:.3f} → {lp_a:.3f}  Δ={delta:+.3f} {'✓' if ok else '✗'}"

def print_completion(label, text, adv, lp_b=None, lp_a=None, full=True):
    lp_str = f"  {lp_tag(lp_b, lp_a, adv)}" if (lp_b is not None and lp_a is not None) else ""
    print(f"  [{label}  adv={adv:+.3f}]{lp_str}")
    if full:
        print(box(text))
    else:
        parts = text.split("<|think_end|>")
        snippet = ("<think>…</think>\n" + parts[-1].strip()) if len(parts) > 1 else text
        print(box(snippet[:400]))
    print()


def print_results(groups, lps_before=None, lps_after=None):
    lps_before = lps_before or {}
    lps_after  = lps_after  or {}

    for gi, g in enumerate(groups):
        ex       = g["example"]
        binary   = g["binary"]
        pos      = g["positives"]
        neg      = g["negatives"]
        pos_w    = sum(abs(a) for *_, a in pos)
        neg_w    = sum(abs(a) for *_, a in neg)
        n        = len(binary)

        separator("═")
        print(f"Q : {ex['question'][:110].replace(chr(10), ' ')}")
        print(f"GT: {ex['ground_truth']}   "
              f"solve={int(sum(binary))}/{n}  ({100*sum(binary)/n:.0f}%)")
        separator()

        if not pos:
            print("  (no correct completions)\n")
        else:
            print(f"\n  ✓ POSITIVES  n={len(pos)}  Σ|adv|={pos_w:.3f}\n")
            for rank, (idx, text, _, adv) in enumerate(pos, 1):
                lp_b = lps_before.get((gi, idx))
                lp_a = lps_after.get((gi, idx))
                print_completion(
                    f"pos {rank}/{len(pos)}  idx={idx}",
                    text, adv, lp_b, lp_a, full=True,
                )

        if not neg:
            print("  (no negatives)\n")
        else:
            print(f"  ✗ NEGATIVES  n={len(neg)}  Σ|adv|={neg_w:.3f}  "
                  f"(pool={len(neg)})\n")
            for rank, (idx, text, _, adv) in enumerate(neg, 1):
                lp_b = lps_before.get((gi, idx))
                lp_a = lps_after.get((gi, idx))
                print_completion(
                    f"neg {rank}/{len(neg)}  idx={idx}",
                    text, adv, lp_b, lp_a, full=False,
                )


# ── DIAGNOSTIC: reward delta after gradient step ─────────────────────────────
# Regenerates the same prompts with the updated torch_gen and compares mean
# rewards before vs after.  Remove this block once training is validated.

def reward_delta_check(torch_gen, scored_groups, G, max_tokens, temperature):
    """Re-generate the same prompts and compare mean rewards.

    Returns list of (mean_reward_before, mean_reward_after, delta) per group.
    """
    rows = []
    for g in scored_groups:
        ex       = g["example"]
        r_before = np.mean(g["rewards"])

        # Re-generate with updated model (different samples, same temperature)
        texts_new, comp_ids_new = torch_gen.generate_group(
            ex["prompt"], n=G,
            max_new_tokens=max_tokens,
            temperature=temperature,
            use_synth_template=False,
        )
        rewards_new, _ = score_group({
            "texts":    texts_new,
            "comp_ids": comp_ids_new,
            "example":  ex,
        })
        r_after = np.mean(rewards_new)
        rows.append((r_before, r_after, r_after - r_before))

    separator("═")
    print("  DIAGNOSTIC — reward delta after one gradient step")
    separator()
    total_before, total_after = 0.0, 0.0
    for gi, (rb, ra, d) in enumerate(rows):
        ex = scored_groups[gi]["example"]
        ok = "↑" if d > 0 else ("↓" if d < 0 else "=")
        print(f"  [{ok}] group {gi}  r_before={rb:.4f}  r_after={ra:.4f}  Δ={d:+.4f}"
              f"   GT={ex['ground_truth']}")
        total_before += rb
        total_after  += ra
    n = len(rows)
    td = total_after / n - total_before / n
    ok = "↑" if td > 0 else ("↓" if td < 0 else "=")
    separator()
    print(f"  [{ok}] MEAN  r_before={total_before/n:.4f}  r_after={total_after/n:.4f}"
          f"  Δ={td:+.4f}")
    separator("═")
    return rows


# ── Eval ──────────────────────────────────────────────────────────────────────

def eval_pass_at_k(torch_gen: TorchGenerator, keras_model, examples, k,
                   max_tokens, temperature, gen_batch_size=8):
    """pass@k via TorchGenerator.generate_batch, batching gen_batch_size prompts.

    keras_model is moved to CPU before generation (free VRAM) and back after.
    """
    keras_model.cpu()
    torch.cuda.empty_cache()

    n_correct = 0
    prompts = [ex["prompt"] for ex in examples]
    for start in range(0, len(prompts), gen_batch_size):
        batch_prompts = prompts[start : start + gen_batch_size]
        # k passes over the same batch of prompts
        batch_correct = [False] * len(batch_prompts)
        for _ in range(k):
            texts = torch_gen.generate_batch(
                batch_prompts,
                max_new_tokens=max_tokens,
                temperature=temperature,
                use_synth_template=False,
            )
            for j, (text, ex) in enumerate(
                zip(texts, examples[start : start + gen_batch_size])
            ):
                if not batch_correct[j] and check_answer(text, ex["ground_truth"]):
                    batch_correct[j] = True
        n_correct += sum(batch_correct)

    keras_model.cuda()
    return n_correct / len(examples)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="./checkpoints/seqcond_lin5.pt")
    p.add_argument("--n",           type=int,   default=2,    help="prompts per step")
    p.add_argument("--g",           type=int,   default=16,   help="completions per prompt")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_tokens",  type=int,   default=800)
    p.add_argument("--lr",          type=float, default=1e-5)
    p.add_argument("--sft_steps",   type=int,   default=1,    help="gradient steps per batch")
    p.add_argument("--steps",       type=int,   default=500,  help="total training steps")
    p.add_argument("--eval_every",  type=int,   default=20,   help="eval frequency (steps)")
    p.add_argument("--eval_n",      type=int,   default=50,  help="test examples for eval")
    p.add_argument("--eval_k",      type=int,   default=1,    help="completions per problem for pass@k")
    p.add_argument("--eval_batch",  type=int,   default=8,    help="prompts per generation batch during eval")
    p.add_argument("--seed",        type=int,   default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Load models ───────────────────────────────────────────────────────────
    torch_gen  = TorchGenerator(args.checkpoint, device="cuda")
    config     = torch_gen.config
    state_dict = {k: v.detach().cpu().float().numpy()
                  for k, v in torch_gen.model.state_dict().items()}

    keras.mixed_precision.set_global_policy("mixed_float16")
    keras_model = build_keras_model(config)
    convert_weights(config, state_dict, keras_model)
    keras_model.cuda()
    optimizer = torch.optim.AdamW(
        keras_model.parameters(), lr=args.lr, betas=(0.9, 0.99),
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    train, test = load_train_test(seed=args.seed)
    eval_examples = test[:args.eval_n]

    # ── Training loop ─────────────────────────────────────────────────────────
    prev_eval_score = None
    print(f"\n  evaluating pass@{args.eval_k} on {args.eval_n} test examples …")
    score = eval_pass_at_k(
        torch_gen, keras_model, eval_examples,
        k=args.eval_k, max_tokens=args.max_tokens,
        temperature=args.temperature, gen_batch_size=args.eval_batch,
    )
    separator("═")
    print(f"  BASELINE EVAL  pass@{args.eval_k}"
          f"  n={args.eval_n}  score={score:.4f}")
    separator("═")
    print()
    prev_eval_score = score

    for step in range(1, args.steps + 1 - args.n):
        batch = train[step : step + args.n]

        raw_groups = generate_groups(
            torch_gen, batch,
            G=args.g, max_tokens=args.max_tokens, temperature=args.temperature,
        )
        scored_groups = []
        for g in raw_groups:
            rewards, binary = score_group(g)
            advantages = list(_compute_advantages(rewards, normalize_std=False))
            positives, negatives = select_pairs(g["texts"], g["comp_ids"], advantages, binary)
            scored_groups.append({
                "example":       g["example"],
                "prompt_tokens": g["prompt_tokens"],
                "texts":         g["texts"],
                "binary":        binary,
                "rewards":       rewards,
                "advantages":    advantages,
                "positives":     positives,
                "negatives":     negatives,
            })

        # ── Gradient step ─────────────────────────────────────────────────────
        has_signal = any(g["positives"] or g["negatives"] for g in scored_groups)
        loss = 0.0
        if has_signal:
            for _ in range(args.sft_steps):
                loss = _apply_gradient(keras_model, optimizer, scored_groups)
            sync_keras_to_torch(keras_model, torch_gen.model, config)

        # ── Per-step summary ──────────────────────────────────────────────────
        n_comp    = sum(len(g["binary"]) for g in scored_groups)
        n_correct = sum(int(sum(g["binary"])) for g in scored_groups)
        mean_r    = float(np.mean([r for g in scored_groups for r in g["rewards"]]))
        skip_tag  = "  (skipped)" if not has_signal else ""
        print(f"[step {step:4d}/{args.steps}]  "
              f"solve={n_correct}/{n_comp}({100*n_correct/n_comp:.0f}%)  "
              f"r̄={mean_r:.4f}  loss={loss:.5f}{skip_tag}")

        # ── Eval pass@k ───────────────────────────────────────────────────────
        if step % args.eval_every == 0:
            print(f"\n  evaluating pass@{args.eval_k} on {args.eval_n} test examples …")
            score = eval_pass_at_k(
                torch_gen, keras_model, eval_examples,
                k=args.eval_k, max_tokens=args.max_tokens,
                temperature=args.temperature, gen_batch_size=args.eval_batch,
            )
            if prev_eval_score is not None:
                d = score - prev_eval_score
                tag = f"  Δ={d:+.4f} {'↑' if d > 0 else ('↓' if d < 0 else '=')}"
            else:
                tag = "  (baseline)"
            separator("═")
            print(f"  EVAL  step={step}/{args.steps}  pass@{args.eval_k}"
                  f"  n={args.eval_n}  score={score:.4f}{tag}")
            separator("═")
            print()
            prev_eval_score = score


if __name__ == "__main__":
    main()

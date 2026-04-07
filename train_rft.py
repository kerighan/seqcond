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
    build_keras_model, convert_weights,
)
from train_grpo import (
    load_gsm8k, check_answer, _partial_answer_score,
    _compute_advantages, _repetition_penalty,
    _seq_token_log_probs, sync_keras_to_torch,
)
import time


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
    rewards = [b if b > 0 else _partial_answer_score(t, gt)
               for t, b in zip(texts, binary)]

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

    positives: adv > 0  (correct answers + near-correct above mean)
    negatives: adv < 0  (wrong or below-mean answers)

    Using adv > 0 for positives (not just binary=1) ensures that when
    solve=0, partial-credit completions above the group mean still get
    a positive gradient update — avoids one-sided negative-only steps.
    """
    positives = sorted(
        [(i, texts[i], comp_ids[i], advantages[i])
         for i, a in enumerate(advantages) if a > 0],
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

def _apply_gradient_softmax(keras_model, optimizer, scored_groups, softmax_temp=1.0):
    """SFT pur avec pondération softmax des rewards par groupe.

    Pour un groupe avec k positifs et rewards [r1, …, rk] :
        w_i = k * softmax(rewards / T)[i]

    Si tous les rewards sont égaux → w_i = 1 pour tous (SFT pur).
    Si un reward domine      → ce positif reçoit presque tout le gradient.

    Normalisation globale par total_n_pos pour que la magnitude du gradient
    soit indépendante du nombre de positifs par batch.
    """
    optimizer.zero_grad()

    all_items = []
    total_n_pos = 0

    for g in scored_groups:
        pt  = g["prompt_tokens"]
        pos = g["positives"]          # (idx, text, comp, adv)
        if not pos:
            continue

        # raw rewards pour les positifs (pas les avantages, mais softmax est invariant)
        rewards_pos = np.array([g["rewards"][idx] for (idx, _, _, _) in pos], dtype=np.float32)
        n_pos = len(pos)
        total_n_pos += n_pos

        # softmax numériquement stable avec température
        r = rewards_pos / softmax_temp
        r -= r.max()
        exp_r = np.exp(r)
        weights = n_pos * exp_r / exp_r.sum()   # somme = n_pos

        for (idx, _, comp, _), w in zip(pos, weights):
            if comp:
                all_items.append({"ids": pt + comp, "pl": len(pt), "w": float(w)})

    if total_n_pos == 0:
        return 0.0, 0.0

    total_tokens = sum(len(it["ids"]) - it["pl"] for it in all_items)
    total_loss_val = 0.0
    for item in all_items:
        ids       = np.array([item["ids"]], dtype=np.int32)
        token_lps = _seq_token_log_probs(keras_model, ids, item["pl"])
        loss      = (-item["w"] * ops.sum(token_lps) / total_tokens) / total_n_pos
        loss.backward()
        total_loss_val += float(loss.detach())

    trainable = [p for p in keras_model.parameters() if p.requires_grad]
    gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0))
    optimizer.step()
    return total_loss_val, gnorm


_GRAD_CHECKED = False  # print grad diagnostics once

def _apply_gradient(keras_model, optimizer, scored_groups, micro_batch_size=2,
                    neg_weight=1.0, kl_beta=0.0, ref_model=None):
    global _GRAD_CHECKED
    optimizer.zero_grad()

    all_items = []
    for g in scored_groups:
        pt = g["prompt_tokens"]
        for (_, _, comp, adv) in g["positives"] + g["negatives"]:
            if comp:
                w = neg_weight if adv < 0 else 1.0
                all_items.append({"ids": pt + comp, "pl": len(pt), "adv": adv, "w": w})

    # keep only items that contribute gradient (w > 0)
    all_items = [it for it in all_items if it["w"] > 0]
    n_terms = len(all_items)
    if n_terms == 0: return 0.0, 0.0

    total_tokens = sum(len(it["ids"]) - it["pl"] for it in all_items)
    total_loss_val = 0.0
    for i in range(0, n_terms, micro_batch_size):
        batch_loss = 0
        for item in all_items[i : i + micro_batch_size]:
            ids = np.array([item["ids"]], dtype=np.int32)
            token_lps = _seq_token_log_probs(keras_model, ids, item["pl"])
            w = item["w"]
            loss = (-w * item["adv"] * ops.sum(token_lps) / total_tokens) / n_terms
            if kl_beta > 0 and ref_model is not None:
                with torch.no_grad():
                    ref_lps = _seq_token_log_probs(ref_model, ids, item["pl"])
                kl = ops.sum(token_lps.float() - ref_lps.float()) / total_tokens
                loss = loss + (kl_beta * kl) / n_terms
            batch_loss     += loss
            total_loss_val += float(loss.detach())
        if isinstance(batch_loss, int):
            continue
        batch_loss.backward()

    trainable = [p for p in keras_model.parameters() if p.requires_grad]
    gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0))

    if not _GRAD_CHECKED:
        _GRAD_CHECKED = True
        has_grad  = [p for p in trainable if p.grad is not None]
        nonzero   = [p for p in has_grad  if p.grad.abs().max() > 0]
        ref_param = trainable[0]
        val_before = ref_param.data.float().mean().item()
        optimizer.step()
        val_after  = ref_param.data.float().mean().item()
        print(f"\n  [GRAD CHECK] trainable={len(trainable)} | has_grad={len(has_grad)} | nonzero_grad={len(nonzero)}")
        print(f"  [GRAD CHECK] param[0] mean: {val_before:.6f} → {val_after:.6f}  (changed={val_before != val_after})\n")
    else:
        optimizer.step()

    return total_loss_val, gnorm


def _apply_gradient_2(keras_model, optimizer, scored_groups, **kwargs):
    """One gradient update on all groups.  CE is recomputed from scratch.

    Loss: mean_over_sequences( -adv * mean_token_log_prob )
      adv > 0 → SFT (push log-prob up)
      adv < 0 → gradient ascent (push log-prob down)
    No padding: input_ids = prompt + comp (exact length); only completion
    tokens contribute via the _seq_token_log_probs slice.
    """
    optimizer.zero_grad()
    all_comps = []
    for g in scored_groups:
        pt = g["prompt_tokens"]
        pl = len(pt)
        for (_, _, comp, adv) in g["positives"] + g["negatives"]:
            if not comp:
                continue
            all_comps.append({"ids": pt + comp, "pl": pl, "adv": adv})
    n_terms = len(all_comps)
    if n_terms == 0:
        optimizer.step()
        return 0.0
    total_tokens = sum(len(it["ids"]) - it["pl"] for it in all_comps)
    total_loss = torch.tensor(0.0, device="cuda")
    for item in all_comps:
        ids = np.array([item["ids"]], dtype=np.int32)
        token_lps = _seq_token_log_probs(keras_model, ids, item["pl"])
        total_loss += (-item["adv"] * ops.sum(token_lps) / total_tokens)
    (total_loss / n_terms).backward()
    optimizer.step()
    return float(total_loss / n_terms)


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


# ── Checkpoint ───────────────────────────────────────────────────────────────

_CACHE_PREFIXES = (
    '_conv_kernel_t', '_decay_slopes_cached', '_phase_scale_b',
    '_score_bias_b', '_score_scale_b', '_theta_cached', '_w_int_cached',
)

def save_checkpoint(torch_gen: TorchGenerator, config, save_path, step):
    """Save directly from torch_gen weights — avoids lossy Keras→pkl→pt round-trip.

    Strips seq-len-dependent cached buffers that corrupt generation if reloaded.
    """
    sd = {k: v.cpu() for k, v in torch_gen.model.state_dict().items()
          if not any(k.endswith(sfx) for sfx in _CACHE_PREFIXES)}
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({"state_dict": sd, "config": config}, save_path)
    print(f"  checkpoint saved → {save_path}  ({len(sd)} tensors, step={step})")


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
    # Warm up CUDA kernels so the next training step isn't 3× slower
    dummy = np.ones((1, 4), dtype=np.int32)
    with torch.no_grad():
        keras_model(dummy, training=False)
    return n_correct / len(examples)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="./checkpoints/seqcond_lin5.pt")
    p.add_argument("--n",           type=int,   default=4,    help="prompts per step")
    p.add_argument("--g",           type=int,   default=16,   help="completions per prompt")
    p.add_argument("--temperature",      type=float, default=0.7)
    p.add_argument("--eval_temperature", type=float, default=None, help="eval temperature (default: same as --temperature)")
    p.add_argument("--max_tokens",  type=int,   default=1000)
    p.add_argument("--lr",          type=float, default=1e-5)
    p.add_argument("--sft_steps",   type=int,   default=1,    help="gradient steps per batch")
    p.add_argument("--neg_weight",        type=float, default=2,     help="scale factor for negative gradient ascent (1.0 = symmetric)")
    p.add_argument("--conciseness_alpha", type=float, default=0.02,  help="force du bonus/malus de concision (0=désactivé, 0.05=fort)")
    p.add_argument("--kl_beta",           type=float, default=0.0,   help="KL penalty weight against reference policy (0 = disabled)")
    p.add_argument("--softmax_weighted",  action="store_true",        help="pondération softmax des rewards par groupe (remplace neg_weight)")
    p.add_argument("--softmax_temp",      type=float, default=1.0,   help="température du softmax (haute=uniforme, basse=winner-take-all)")
    p.add_argument("--float32",          action="store_true",        help="use float32 compute instead of mixed float16 (more stable, more memory)")
    p.add_argument("--transformer_only", action="store_true",        help="freeze seqcond blocks, train only transformer blocks")
    p.add_argument("--epochs",      type=float, default=1.0,  help="epochs over train set (float ok)")
    p.add_argument("--eval_every",  type=int,   default=20,   help="eval frequency (steps)")
    p.add_argument("--eval_n",      type=int,   default=50,  help="test examples for eval")
    p.add_argument("--eval_k",      type=int,   default=1,    help="completions per problem for pass@k")
    p.add_argument("--eval_batch",  type=int,   default=8,    help="prompts per generation batch during eval")
    p.add_argument("--save_path",   type=str,   default=None, help="override default save dir (checkpoints/graft/)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--overfit_n",   type=int,   default=None, help="overfit sanity check: train+eval on the same N train examples")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.save_path is None:
        base = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.save_path = os.path.join("checkpoints", "graft", f"{base}_graft.pt")

    # ── Load models ───────────────────────────────────────────────────────────
    torch_gen  = TorchGenerator(args.checkpoint, device="cuda")
    config     = torch_gen.config
    state_dict = {k: v.detach().cpu().float().numpy()
                  for k, v in torch_gen.model.state_dict().items()}

    if not args.float32:
        keras.mixed_precision.set_global_policy("mixed_bfloat16")
    keras_model = build_keras_model(config)
    convert_weights(config, state_dict, keras_model)
    keras_model.cuda()

    # ── Reference model for KL penalty (frozen, kept on CPU) ──────────────────
    ref_model = None
    if args.kl_beta > 0:
        ref_model = build_keras_model(config)
        convert_weights(config, state_dict, ref_model)
        ref_model.cpu()
        ref_model.cuda()
        for p in ref_model.parameters():
            p.requires_grad = False
        print(f"  KL penalty enabled  β={args.kl_beta}")

    if args.transformer_only:
        for btype, block in zip(keras_model.block_types, keras_model.blocks_list):
            if btype != "transformer":
                for p in block.parameters():
                    p.requires_grad = False
        for p in keras_model.token_embedding.parameters():
            p.requires_grad = False
        n_transformer = sum(1 for t in keras_model.block_types if t == "transformer")
        n_total       = len(keras_model.block_types)
        print(f"  trainable: transformer blocks only ({n_transformer}/{n_total} blocks, embeddings frozen)")

    trainable_params = [p for p in keras_model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total     = sum(p.numel() for p in keras_model.parameters())
    print(f"  params: {n_trainable:,} trainable / {n_total:,} total")
    if n_trainable == 0:
        raise RuntimeError("No trainable parameters! Keras weights may not have requires_grad=True.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.99))

    # ── Data ──────────────────────────────────────────────────────────────────
    train, test = load_train_test(seed=args.seed)
    if args.overfit_n is not None:
        train = train[:args.overfit_n]
        eval_examples = train
        print(f"  [OVERFIT MODE] training + eval on the same {len(train)} examples")
    else:
        eval_examples = test[:args.eval_n]

    # ── Baseline eval ─────────────────────────────────────────────────────────
    eval_temp = args.eval_temperature if args.eval_temperature is not None else args.temperature
    prev_eval_score = None
    n_eval_label = len(eval_examples)
    print(f"\n  Baseline eval pass@{args.eval_k} on {n_eval_label} examples …")
    score = eval_pass_at_k(
        torch_gen, keras_model, eval_examples,
        k=args.eval_k, max_tokens=args.max_tokens,
        temperature=eval_temp, gen_batch_size=args.eval_batch,
    )
    separator("═")
    print(f"  BASELINE EVAL  pass@{args.eval_k}"
          f"  n={n_eval_label}  score={score:.4f}")
    separator("═")
    print()
    prev_eval_score = score

    # ── Training loop ─────────────────────────────────────────────────────────
    steps_per_epoch = len(train) // args.n
    total_steps     = max(1, int(steps_per_epoch * args.epochs))
    print(f"Train: {len(train)} examples  →  {steps_per_epoch} steps/epoch  ×  {args.epochs} = {total_steps} steps\n")

    train_buf  = []   # refilled + reshuffled at the start of each epoch
    step       = 0
    last_step_time = time.perf_counter()

    # ── Inter-eval accumulators ───────────────────────────────────────────────
    _acc = dict(n_correct=0, n_comp=0, reward_sum=0.0, reward_n=0,
                loss_sum=0.0, gnorm_sum=0.0, grad_steps=0, steps=0)

    while step < total_steps:
        if not train_buf:
            epoch_data = list(train)
            random.shuffle(epoch_data)
            train_buf  = [epoch_data[i : i + args.n]
                          for i in range(0, len(epoch_data) - args.n + 1, args.n)]

        batch = train_buf.pop(0)
        step += 1

        raw_groups = generate_groups(
            torch_gen, batch,
            G=args.g, max_tokens=args.max_tokens, temperature=args.temperature,
        )
        scored_groups = []
        for g in raw_groups:
            rewards, binary = score_group(g, conciseness_alpha=args.conciseness_alpha)
            advantages = list(_compute_advantages(rewards, normalize_std=True))
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
        if args.softmax_weighted:
            has_signal = any(g["positives"] for g in scored_groups)
        loss = 0.0
        gnorm = 0.0
        n_pos = sum(len(g["positives"]) for g in scored_groups)
        n_neg = sum(len(g["negatives"]) for g in scored_groups)
        if has_signal:
            for _ in range(args.sft_steps):
                if args.softmax_weighted:
                    loss, gnorm = _apply_gradient_softmax(
                        keras_model, optimizer, scored_groups,
                        softmax_temp=args.softmax_temp,
                    )
                    continue
                loss, gnorm = _apply_gradient(keras_model, optimizer, scored_groups,
                                              neg_weight=args.neg_weight,
                                              kl_beta=args.kl_beta,
                                              ref_model=ref_model)
            sync_keras_to_torch(keras_model, torch_gen.model, config)

        # ── Per-step summary ──────────────────────────────────────────────────
        n_comp    = sum(len(g["binary"]) for g in scored_groups)
        n_correct = sum(int(sum(g["binary"])) for g in scored_groups)
        mean_r    = float(np.mean([r for g in scored_groups for r in g["rewards"]]))
        skip_tag  = "  (skipped)" if not has_signal else ""
        step_duration = time.perf_counter() - last_step_time
        last_step_time = time.perf_counter()
        w = len(str(n_comp))
        print(f"[step {step:4d}/{total_steps}]  "
              f"solve={n_correct:{w}d}/{n_comp}({100*n_correct/n_comp:3.0f}%)  "
              f"r̄={mean_r:.4f}  n+/n-={n_pos}/{n_neg}  "
              f"loss={loss:.5f}  gnorm={gnorm:.3f}{skip_tag}  {step_duration:.1f}s")

        # accumulate for inter-eval summary
        _acc["n_correct"]  += n_correct
        _acc["n_comp"]     += n_comp
        _acc["reward_sum"] += mean_r
        _acc["reward_n"]   += 1
        _acc["steps"]      += 1
        if has_signal:
            _acc["loss_sum"]   += loss
            _acc["gnorm_sum"]  += gnorm
            _acc["grad_steps"] += 1

        # ── Eval pass@k ───────────────────────────────────────────────────────
        if step % args.eval_every == 0:
            # print inter-eval training summary
            if _acc["steps"] > 0:
                avg_solve = _acc["n_correct"] / max(_acc["n_comp"], 1)
                avg_r     = _acc["reward_sum"] / _acc["reward_n"]
                gs        = _acc["grad_steps"]
                avg_loss  = _acc["loss_sum"]  / gs if gs else float("nan")
                avg_gnorm = _acc["gnorm_sum"] / gs if gs else float("nan")
                separator()
                print(f"  last {_acc['steps']} steps:  "
                      f"train_solve={100*avg_solve:.1f}%  "
                      f"r̄={avg_r:.4f}  "
                      f"loss̄={avg_loss:.5f}  "
                      f"gnorm̄={avg_gnorm:.3f}")
                separator()
            _acc.update(n_correct=0, n_comp=0, reward_sum=0.0, reward_n=0,
                        loss_sum=0.0, gnorm_sum=0.0, grad_steps=0, steps=0)
            print(f"\n  evaluating pass@{args.eval_k} on {n_eval_label} examples …")
            score = eval_pass_at_k(
                torch_gen, keras_model, eval_examples,
                k=args.eval_k, max_tokens=args.max_tokens,
                temperature=eval_temp, gen_batch_size=args.eval_batch,
            )
            if prev_eval_score is not None:
                d = score - prev_eval_score
                tag = f"  Δ={d:+.4f} {'↑' if d > 0 else ('↓' if d < 0 else '=')}"
            else:
                tag = "  (baseline)"
            separator("═")
            print(f"  EVAL  step={step}/{total_steps}  pass@{args.eval_k}"
                  f"  n={n_eval_label}  score={score:.4f}{tag}")
            separator("═")
            print()
            prev_eval_score = score

            if args.save_path:
                base, ext = os.path.splitext(args.save_path)
                path = f"{base}_step{step:04d}{ext}"
                save_checkpoint(torch_gen, config, path, step)


if __name__ == "__main__":
    main()

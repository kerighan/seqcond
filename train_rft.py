"""
train_rft.py — One batch: generate → score → gradient step → verify.

Usage:
    python train_rft.py --checkpoint checkpoints/seqcond_lin5.pt --n 4 --g 16
"""
import argparse, copy, math, os, random, re, textwrap
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F

from seqcond.torch.generator import TorchGenerator
from train_grpo import (
    load_gsm8k, check_answer, _partial_answer_score,
    _extract_numeric_answer,
    _compute_advantages, _repetition_penalty,
)
from collect_cot import (
    _extract_math_answer, _math_answers_equal, _extract_answer_after_thinking,
)
import time


# ── Data ──────────────────────────────────────────────────────────────────────

def load_train_test(seed=42, dataset="gsm8k", local_file=None):
    """Load train/test examples; return (train_examples, test_examples) shuffled with seed."""
    if dataset == "local_math":
        import json
        with open(local_file) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        random.shuffle(rows)
        examples = []
        for row in rows:
            query = row["query"]
            gt    = str(row.get("ground_truth", row.get("answer", "")))
            examples.append({"prompt": query, "question": query, "ground_truth": gt})
        split = int(len(examples) * 0.9)
        return examples[:split], examples[split:]
    train = load_gsm8k(split="train", seed=seed)
    test  = load_gsm8k(split="test",  seed=seed)
    return train, test


# ── Generation ────────────────────────────────────────────────────────────────

def generate_groups(torch_gen: TorchGenerator, examples, G, max_tokens, temperature,
                    use_synth_template=False):
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
            use_synth_template=use_synth_template,
        )
        results.append({
            "example":       ex,
            "prompt_tokens": prompt_tokens,
            "texts":         texts,
            "comp_ids":      comp_ids,
        })
    return results


# ── Scoring ───────────────────────────────────────────────────────────────────

_THINK_END_TOKEN = "<|think_end|>"
_THINK_FORMAT_RE = re.compile(r"(?s)^(.*?)<\|think_end\|>\s*(.+?)\s*$")


def _extract_structured_answer(text):
    stripped = text.strip()
    if stripped.count(_THINK_END_TOKEN) != 1:
        return None
    match = _THINK_FORMAT_RE.fullmatch(stripped)
    if match is None:
        return None
    answer = match.group(2).strip()
    return answer if answer else None


def _format_reward(text):
    if _extract_structured_answer(text) is not None:
        return 3.0
    return 0.5 if text.count(_THINK_END_TOKEN) == 1 else -1.0


def _answer_reward(text, ground_truth, r_max=5.0, r_floor=0.1, sigma=0.5):
    """Continuous Gaussian reward based on numeric distance.

    - Exact match           → r_max
    - Numeric guess found   → r_max * exp(-(log(guess/gt))² / (2σ²))
    - No number extracted   → r_floor
    Never negative — negatives come only from advantage mean-centering.
    """
    # Exact match shortcut
    if check_answer(text, ground_truth):
        return r_max

    # Try to extract a numeric guess
    answer = _extract_structured_answer(text)
    if answer is not None:
        guess_str = _extract_numeric_answer(answer, answer_last=True)
    else:
        guess_str = _extract_numeric_answer(text)

    if guess_str is None:
        return r_floor

    try:
        guess_val = float(guess_str)
        true_val  = float(ground_truth)
    except ValueError:
        return r_floor

    if true_val == 0.0:
        # Can't compute log-ratio; use absolute distance fallback
        dist = abs(guess_val)
        return r_max * math.exp(-dist * dist / 2.0) if dist < 10.0 else r_floor

    if guess_val == 0.0:
        return r_floor

    # Gaussian on log-ratio: symmetric (×2 and ÷2 same distance)
    log_ratio = math.log(abs(guess_val / true_val))
    reward = r_max * math.exp(-log_ratio * log_ratio / (2.0 * sigma * sigma))

    # Sign mismatch penalty (halve reward if signs differ)
    if (guess_val > 0) != (true_val > 0):
        reward *= 0.5

    return max(reward, r_floor)

def _is_correct(text, gt, dataset="gsm8k"):
    if dataset == "local_math":
        answer_text = _extract_answer_after_thinking(text)
        predicted   = _extract_math_answer(answer_text)
        return _math_answers_equal(predicted, gt)
    return check_answer(text, gt)


def score_group(group, conciseness_alpha=0.05):
    """Binary reward 1/0, conciseness-scaled for correct completions.

    Correct completions shorter than the group mean get a bonus:
        r *= exp(-alpha * (len - mu) / sigma)
    """
    texts, comp_ids, gt = group["texts"], group["comp_ids"], group["example"]["ground_truth"]
    dataset = group["example"].get("dataset", "gsm8k")
    binary = [1.0 if _is_correct(t, gt, dataset) else 0.0 for t in texts]
    rewards = [_format_reward(t) + _answer_reward(t, gt) for t in texts]

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

def _model_device(model):
    return next(model.parameters()).device


def _autocast_context(use_mixed_bfloat16):
    if use_mixed_bfloat16 and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _seq_token_log_probs(model, input_ids, prompt_len, use_mixed_bfloat16=False):
    device = _model_device(model)
    if torch.is_tensor(input_ids):
        ids = input_ids.to(device=device, dtype=torch.long)
    else:
        ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    with _autocast_context(use_mixed_bfloat16):
        logits = model(ids)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    shift = log_probs[0, prompt_len - 1 : -1, :]
    targets = ids[0, prompt_len:].unsqueeze(-1)
    return torch.gather(shift, dim=-1, index=targets).squeeze(-1)

def seq_avg_logprob(model, prompt_tokens, comp_tokens, use_mixed_bfloat16=False):
    """Mean per-token log-prob for comp_tokens (no grad, purely cosmetic)."""
    ids = torch.tensor([prompt_tokens + comp_tokens], dtype=torch.long, device=_model_device(model))
    with torch.no_grad():
        token_lps = _seq_token_log_probs(
            model, ids, len(prompt_tokens), use_mixed_bfloat16=use_mixed_bfloat16
        )
    return float(torch.mean(token_lps))


def _cache_old_policy_lps(model, scored_groups, use_mixed_bfloat16=False):
    for g in scored_groups:
        pt = g["prompt_tokens"]
        old_mean_lps = {}
        for idx, comp in enumerate(g["comp_ids"]):
            if comp:
                old_mean_lps[idx] = seq_avg_logprob(
                    model, pt, comp, use_mixed_bfloat16=use_mixed_bfloat16
                )
        g["old_mean_lps"] = old_mean_lps


def _set_trainable_parameters(model, target):
    if target == "all":
        for param in model.parameters():
            param.requires_grad = True
        return []

    for param in model.parameters():
        param.requires_grad = False

    unfrozen_names = []
    for idx, (btype, block) in enumerate(zip(model.block_types, model.blocks)):
        block_name = f"block{idx:02d}"
        if target in ("transformer", "seqcond") and btype == target:
            for param in block.parameters():
                param.requires_grad = True
            unfrozen_names.append(f"{btype}:{block_name}")
        elif target == "mlp" or target == f"{btype}-mlp":
            if btype == "transformer":
                for sub in (block.norm2, block.ff_in, block.ff_out):
                    for param in sub.parameters():
                        param.requires_grad = True
                unfrozen_names.append(f"transformer-mlp:{block_name}")
            elif btype == "seqcond":
                attn = block.attn
                for sub in (attn.gate_proj, attn.out_proj, attn.gated_norm):
                    for param in sub.parameters():
                        param.requires_grad = True
                attn.W_readout.requires_grad = True
                unfrozen_names.append(f"seqcond-mlp:{block_name}")
    return unfrozen_names


# ── Gradient step ─────────────────────────────────────────────────────────────

def _apply_gradient_softmax(model, optimizer, scored_groups, softmax_temp=1.0,
                            use_mixed_bfloat16=False):
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
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for item in all_items:
        ids = torch.tensor([item["ids"]], dtype=torch.long, device=_model_device(model))
        token_lps = _seq_token_log_probs(
            model, ids, item["pl"], use_mixed_bfloat16=use_mixed_bfloat16
        )
        loss = (-item["w"] * torch.sum(token_lps) / total_tokens) / total_n_pos
        loss.backward()
        total_loss_val += float(loss.detach())

    trainable = [p for p in model.parameters() if p.requires_grad]
    gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0))
    optimizer.step()
    return total_loss_val, gnorm


_GRAD_CHECKED = False  # print grad diagnostics once


def _apply_gradient(model, optimizer, scored_groups, micro_batch_size=2,
                    neg_scale=0.0, kl_beta=0.0, ref_model=None,
                    use_mixed_bfloat16=False, on_policy=False):
    global _GRAD_CHECKED
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]

    # Build item lists
    pos_items, neg_items = [], []
    for g in scored_groups:
        pt = g["prompt_tokens"]
        for (idx, _, comp, adv) in g["positives"] + g["negatives"]:
            if not comp:
                continue
            if adv > 0:
                pos_items.append({
                    "ids": pt + comp,
                    "pl": len(pt),
                    "adv": adv,
                    "old_mean_lp": g.get("old_mean_lps", {}).get(idx),
                })
            elif adv < 0 and neg_scale > 0:
                neg_items.append({
                    "ids": pt + comp,
                    "pl": len(pt),
                    "adv": adv,
                    "old_mean_lp": g.get("old_mean_lps", {}).get(idx),
                })

    if not pos_items and not neg_items:
        return 0.0, 0.0, 0.0

    all_items = pos_items + neg_items
    n_terms = len(all_items)
    total_tokens = sum(len(it["ids"]) - it["pl"] for it in all_items)

    # ── Estimate effective neg weight via micro-batched grad norms ────────────
    # neg_scale controls negative gradient magnitude relative to positive:
    #   neg_scale=0   → no negatives
    #   neg_scale=0.5 → neg gradient = 50% of pos gradient magnitude
    #   neg_scale=1.0 → balanced (equal magnitudes)
    effective_neg_weight = 0.0
    if neg_scale > 0 and neg_items and pos_items:
        def _micro_grad_norm(items):
            optimizer.zero_grad(set_to_none=True)
            for i in range(0, len(items), micro_batch_size):
                batch_loss = None
                for item in items[i : i + micro_batch_size]:
                    ids = torch.tensor([item["ids"]], dtype=torch.long,
                                       device=_model_device(model))
                    token_lps = _seq_token_log_probs(
                        model, ids, item["pl"], use_mixed_bfloat16=use_mixed_bfloat16
                    )
                    token_weight = (len(item["ids"]) - item["pl"]) / total_tokens
                    if on_policy and item["old_mean_lp"] is not None:
                        cur_mean_lp = torch.mean(token_lps)
                        log_ratio = cur_mean_lp - float(item["old_mean_lp"])
                        ratio = torch.exp(torch.clamp(log_ratio, min=-0.2, max=0.2))
                        item_loss = (-item["adv"] * ratio * token_weight) / n_terms
                    else:
                        item_loss = (-item["adv"] * torch.sum(token_lps) / total_tokens) / n_terms
                    batch_loss = item_loss if batch_loss is None else batch_loss + item_loss
                if batch_loss is not None:
                    batch_loss.backward()
            return sum(float(torch.sum(p.grad.float() ** 2))
                       for p in trainable if p.grad is not None) ** 0.5

        pos_norm = _micro_grad_norm(pos_items)
        neg_norm = _micro_grad_norm(neg_items)
        if pos_norm > 1e-8 and neg_norm > 1e-8:
            effective_neg_weight = min(neg_scale * pos_norm / neg_norm, 10.0)

    # ── Final gradient step ───────────────────────────────────────────────────
    optimizer.zero_grad(set_to_none=True)
    total_loss_val = 0.0
    for i in range(0, n_terms, micro_batch_size):
        batch_loss = None
        for item in all_items[i : i + micro_batch_size]:
            ids = torch.tensor([item["ids"]], dtype=torch.long, device=_model_device(model))
            token_lps = _seq_token_log_probs(
                model, ids, item["pl"], use_mixed_bfloat16=use_mixed_bfloat16
            )
            w = effective_neg_weight if item["adv"] < 0 else 1.0
            if on_policy and item["old_mean_lp"] is not None:
                cur_mean_lp = torch.mean(token_lps)
                token_weight = (len(item["ids"]) - item["pl"]) / total_tokens
                log_ratio = cur_mean_lp - float(item["old_mean_lp"])
                ratio = torch.exp(torch.clamp(log_ratio, min=-0.2, max=0.2))
                loss = (-w * item["adv"] * ratio * token_weight) / n_terms
            else:
                loss = (-w * item["adv"] * torch.sum(token_lps) / total_tokens) / n_terms
            if kl_beta > 0 and ref_model is not None:
                with torch.no_grad():
                    ref_lps = _seq_token_log_probs(
                        ref_model, ids, item["pl"], use_mixed_bfloat16=use_mixed_bfloat16
                    )
                log_ratio = token_lps.float() - ref_lps.float()
                kl_k3 = torch.exp(log_ratio) - 1 - log_ratio  # K3: unbiased, low-variance
                loss = loss + (kl_beta * torch.sum(kl_k3) / total_tokens) / n_terms
            batch_loss = loss if batch_loss is None else batch_loss + loss
            total_loss_val += float(loss.detach())
        if batch_loss is not None:
            batch_loss.backward()

    gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0))

    if not _GRAD_CHECKED:
        _GRAD_CHECKED = True
        has_grad = [p for p in trainable if p.grad is not None]
        nonzero  = [p for p in has_grad  if p.grad.abs().max() > 0]
        ref_param = trainable[0]
        val_before = ref_param.data.float().mean().item()
        optimizer.step()
        val_after  = ref_param.data.float().mean().item()
        print(f"\n  [GRAD CHECK] trainable={len(trainable)} | has_grad={len(has_grad)} | nonzero_grad={len(nonzero)}")
        print(f"  [GRAD CHECK] param[0] mean: {val_before:.6f} → {val_after:.6f}  (changed={val_before != val_after})\n")
    else:
        optimizer.step()

    return total_loss_val, gnorm, effective_neg_weight


def gradient_step(model, optimizer, scored_groups, sft_steps=1, use_mixed_bfloat16=False):
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
            lps_before[(gi, idx)] = seq_avg_logprob(
                model, pt, comp, use_mixed_bfloat16=use_mixed_bfloat16
            )

    # ── sft_steps gradient updates ────────────────────────────────────────────
    for step_i in range(sft_steps):
        loss = _apply_gradient(
            model, optimizer, scored_groups, use_mixed_bfloat16=use_mixed_bfloat16
        )
        print(f"  gradient step {step_i + 1}/{sft_steps}  loss={loss:.4f}")

    # ── After log-probs (no grad) ─────────────────────────────────────────────
    lps_after = {}
    for gi, g in enumerate(scored_groups):
        pt = g["prompt_tokens"]
        for (idx, _, comp, _) in g["positives"] + g["negatives"]:
            lps_after[(gi, idx)] = seq_avg_logprob(
                model, pt, comp, use_mixed_bfloat16=use_mixed_bfloat16
            )

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


def refresh_reference_model(ref_model, model):
    sd = {k: v for k, v in model.state_dict().items()
          if not any(k.endswith(sfx) for sfx in _CACHE_PREFIXES)}
    ref_model.load_state_dict(sd, strict=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False


def snapshot_training_state(model, optimizer):
    model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    for state in optimizer_state.get("state", {}).values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.detach().cpu().clone()
    return model_state, optimizer_state


def _optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def restore_training_state(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    _optimizer_to_device(optimizer, _model_device(model))


def eval_score_sigma(score, n_examples):
    if n_examples <= 0:
        return 0.0
    p = min(max(float(score), 0.0), 1.0)
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / n_examples))


# ── Eval ──────────────────────────────────────────────────────────────────────

def eval_pass_at_k(torch_gen: TorchGenerator, examples, k,
                   max_tokens, temperature, gen_batch_size=8, use_synth_template=False):
    """pass@k via TorchGenerator.generate_batch, batching gen_batch_size prompts.
    """
    model = torch_gen.model
    was_training = model.training
    model.eval()

    n_correct = 0
    prompts = [ex["prompt"] for ex in examples]
    for start in range(0, len(prompts), gen_batch_size):
        batch_prompts = prompts[start : start + gen_batch_size]
        batch_exs     = examples[start : start + gen_batch_size]
        batch_correct = [False] * len(batch_prompts)
        for _ in range(k):
            texts = torch_gen.generate_batch(
                batch_prompts,
                max_new_tokens=max_tokens,
                temperature=temperature,
                use_synth_template=use_synth_template,
            )
            for j, (text, ex) in enumerate(zip(texts, batch_exs)):
                if not batch_correct[j] and _is_correct(
                    text, ex["ground_truth"], ex.get("dataset", "gsm8k")
                ):
                    batch_correct[j] = True
        n_correct += sum(batch_correct)

    if was_training:
        model.train()
    return n_correct / len(examples)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="./checkpoints/seqcond_lin5.pt")
    p.add_argument("--dataset",      default="gsm8k", choices=["gsm8k", "local_math"],
                   help="dataset source (gsm8k | local_math)")
    p.add_argument("--local_file",   default=None,
                   help="chemin JSONL local {query, ground_truth|answer} (requis pour local_math)")
    p.add_argument("--n",           type=int,   default=4,    help="prompts per step")
    p.add_argument("--g",           type=int,   default=16,   help="completions per prompt")
    p.add_argument("--temperature",      type=float, default=0.7)
    p.add_argument("--eval_temperature", type=float, default=None, help="eval temperature (default: same as --temperature)")
    p.add_argument("--max_tokens",  type=int,   default=1000)
    p.add_argument("--lr",          type=float, default=1e-5)
    p.add_argument("--sft_steps",        type=int,   default=1,    help="gradient steps per batch")
    p.add_argument("--micro_batch_size", type=int,   default=24,    help="sequences per backward call (lower = less GPU memory)")
    p.add_argument("--neg_scale",         type=float, default=0.0,   help="negative gradient magnitude relative to positive (0=off, 0.5=half, 1=balanced)")
    p.add_argument("--conciseness_alpha", type=float, default=0.02,  help="force du bonus/malus de concision (0=désactivé, 0.05=fort)")
    p.add_argument("--kl_beta",           type=float, default=0.0,   help="KL penalty weight against reference policy (0 = disabled)")
    p.add_argument("--softmax_weighted",  action="store_true",        help="pondération softmax des rewards par groupe (remplace neg_scale)")
    p.add_argument("--softmax_temp",      type=float, default=1.0,   help="température du softmax (haute=uniforme, basse=winner-take-all)")
    p.add_argument("--float32",          action="store_true",        help="use float32 compute instead of mixed float16 (more stable, more memory)")
    p.add_argument("--on_policy",        action="store_true",        help="apply an old-policy importance ratio on repeated updates over the same sampled batch")
    p.add_argument("--refresh_kl_ref_on_best_eval", action="store_true", help="refresh the KL reference model whenever eval reaches a new best score")
    p.add_argument("--rollback_on_eval_drop", action="store_true", help="restore the best model if eval falls too far below the best score")
    p.add_argument("--rollback_sigma", type=float, default=2.0, help="rollback threshold in binomial sigma units below the best eval score")
    p.add_argument("--train",             default="all",              help="all | transformer | seqcond | mlp | transformer-mlp | seqcond-mlp")
    p.add_argument("--transformer_only", action="store_true",        help="freeze seqcond blocks, train only transformer blocks")
    p.add_argument("--epochs",      type=float, default=1.0,  help="epochs over train set (float ok)")
    p.add_argument("--eval_every",  type=int,   default=20,   help="eval frequency (steps)")
    p.add_argument("--eval_n",      type=int,   default=50,  help="test examples for eval")
    p.add_argument("--eval_k",      type=int,   default=1,    help="completions per problem for pass@k")
    p.add_argument("--eval_batch",  type=int,   default=30,    help="prompts per generation batch during eval")
    p.add_argument("--save_path",   type=str,   default=None, help="override default save dir (checkpoints/graft/)")
    p.add_argument("--save_best",   action="store_true",     help="save the current best eval model to a dedicated *_best.pt checkpoint")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--overfit_n",   type=int,   default=None, help="overfit sanity check: train+eval on the same N train examples")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.save_path is None:
        base = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.save_path = os.path.join("checkpoints", "graft", f"{base}_graft.pt")
    save_base, save_ext = os.path.splitext(args.save_path)
    best_save_path = f"{save_base}_best{save_ext}"

    # ── Load models ───────────────────────────────────────────────────────────
    torch_gen = TorchGenerator(args.checkpoint, device="cuda")
    config = torch_gen.config
    model = torch_gen.model
    use_mixed_bfloat16 = not args.float32
    precision = "mixed_bfloat16 autocast" if use_mixed_bfloat16 else "float32"
    print(f"  using native Torch model ({precision})")

    train_target = "transformer" if args.transformer_only else args.train

    # ── Reference model for KL penalty (frozen, kept on CPU) ──────────────────
    ref_model = None
    if args.kl_beta > 0:
        ref_model = copy.deepcopy(model).cuda().eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        print(f"  KL penalty enabled  β={args.kl_beta}")

    unfrozen_names = _set_trainable_parameters(model, train_target)
    if train_target != "all":
        print(f"  [freeze] --train {train_target}: unfroze {len(unfrozen_names)} block groups, embeddings frozen")
        for name in unfrozen_names:
            print(f"    ✓ {name}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_trainable:,} trainable / {n_total:,} total")
    if n_trainable == 0:
        raise RuntimeError("No trainable parameters! Torch weights may not have requires_grad=True.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.99))

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.dataset == "local_math" and not args.local_file:
        raise ValueError("--local_file requis pour --dataset local_math")
    use_synth_template = (args.dataset == "local_math")
    train, test = load_train_test(seed=args.seed, dataset=args.dataset, local_file=args.local_file)
    for ex in train: ex.setdefault("dataset", args.dataset)
    for ex in test:  ex.setdefault("dataset", args.dataset)
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
        torch_gen, eval_examples,
        k=args.eval_k, max_tokens=args.max_tokens,
        temperature=eval_temp, gen_batch_size=args.eval_batch,
        use_synth_template=use_synth_template,
    )
    separator("═")
    print(f"  BASELINE EVAL  pass@{args.eval_k}"
          f"  n={n_eval_label}  score={score:.4f}")
    separator("═")
    print()
    prev_eval_score = score
    best_eval_score = score
    best_model_state, best_optimizer_state = snapshot_training_state(model, optimizer)
    if args.save_best:
        save_checkpoint(torch_gen, config, best_save_path, 0)

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
            use_synth_template=use_synth_template,
        )
        scored_groups = []
        for g in raw_groups:
            rewards, binary = score_group(g, conciseness_alpha=args.conciseness_alpha)
            advantages = list(_compute_advantages(rewards, normalize_std=True))
            positives, negatives = select_pairs(g["texts"], g["comp_ids"], advantages, binary)
            scored_groups.append({
                "example":       g["example"],
                "prompt_tokens": g["prompt_tokens"],
                "comp_ids":      g["comp_ids"],
                "texts":         g["texts"],
                "binary":        binary,
                "rewards":       rewards,
                "advantages":    advantages,
                "positives":     positives,
                "negatives":     negatives,
            })
        if args.on_policy and not args.softmax_weighted:
            _cache_old_policy_lps(
                model, scored_groups, use_mixed_bfloat16=use_mixed_bfloat16
            )

        # ── Gradient step ─────────────────────────────────────────────────────
        has_signal = any(g["positives"] or g["negatives"] for g in scored_groups)
        if args.softmax_weighted:
            has_signal = any(g["positives"] for g in scored_groups)
        loss = 0.0
        gnorm = 0.0
        effective_neg_weight = args.neg_scale
        n_pos = sum(len(g["positives"]) for g in scored_groups)
        n_neg = sum(len(g["negatives"]) for g in scored_groups)
        if has_signal:
            for _ in range(args.sft_steps):
                if args.softmax_weighted:
                    loss, gnorm = _apply_gradient_softmax(
                        model, optimizer, scored_groups,
                        softmax_temp=args.softmax_temp,
                        use_mixed_bfloat16=use_mixed_bfloat16,
                    )
                    continue
                loss, gnorm, effective_neg_weight = _apply_gradient(
                    model, optimizer, scored_groups,
                    micro_batch_size=args.micro_batch_size,
                    neg_scale=args.neg_scale,
                    kl_beta=args.kl_beta,
                    ref_model=ref_model,
                    use_mixed_bfloat16=use_mixed_bfloat16,
                    on_policy=args.on_policy,
                )

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
              f"loss={loss:.5f}  gnorm={gnorm:.3f}  negw={effective_neg_weight:.4f}{skip_tag}  {step_duration:.1f}s")

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
                torch_gen, eval_examples,
                k=args.eval_k, max_tokens=args.max_tokens,
                temperature=eval_temp, gen_batch_size=args.eval_batch,
                use_synth_template=use_synth_template,
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
            if score > best_eval_score:
                best_eval_score = score
                best_model_state, best_optimizer_state = snapshot_training_state(model, optimizer)
                if args.refresh_kl_ref_on_best_eval and ref_model is not None:
                    refresh_reference_model(ref_model, model)
                    print(f"  KL reference updated from best eval model  score={score:.4f}")
                    print()
                if args.save_best:
                    save_checkpoint(torch_gen, config, best_save_path, step)
            else:
                if args.rollback_on_eval_drop:
                    sigma = eval_score_sigma(best_eval_score, n_eval_label)
                    rollback_threshold = best_eval_score - args.rollback_sigma * sigma
                    if score < rollback_threshold:
                        restore_training_state(model, optimizer, best_model_state, best_optimizer_state)
                        if args.refresh_kl_ref_on_best_eval and ref_model is not None:
                            refresh_reference_model(ref_model, model)
                        print(
                            f"  rollback triggered: score={score:.4f} < threshold={rollback_threshold:.4f} "
                            f"(best={best_eval_score:.4f}, σ={sigma:.4f}, z={args.rollback_sigma:.2f})"
                        )
                        score = best_eval_score
            print()
            prev_eval_score = score

            if args.save_path:
                base, ext = os.path.splitext(args.save_path)
                path = f"{base}_step{step:04d}{ext}"
                save_checkpoint(torch_gen, config, path, step)


if __name__ == "__main__":
    main()

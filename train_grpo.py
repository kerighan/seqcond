"""
GRPO fine-tuning for SeqCond — single PyTorch model (no Keras).

Quick start:
    python train_grpo.py --checkpoint checkpoints/seqcond_lin5.pt

Two functions to modify:
    score_output()  — what counts as a good response  (reward design)
    grpo_step()     — how rewards drive the update     (algorithm)
"""

import argparse
import copy
import math
import os
import random
import re
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from seqcond.dataset import Tokenizer
from seqcond.torch.generator import TorchGenerator


_CACHE_PREFIXES = (
    '_conv_kernel_t', '_decay_slopes_cached', '_phase_scale_b',
    '_score_bias_b', '_score_scale_b', '_theta_cached', '_w_int_cached',
)


def save_ckpt(torch_gen, config, path, step):
    sd = {k: v.cpu() for k, v in torch_gen.model.state_dict().items()
          if not any(k.endswith(s) for s in _CACHE_PREFIXES)}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": sd, "config": config, "step": step}, path)
    print(f"  checkpoint → {path}  ({len(sd)} tensors)")




# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────


def load_gsm8k(split="train", seed=42, max_examples=None):
    """Load GSM8K and return list of dicts with question/ground_truth/prompt."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    examples = []
    for ex in ds:
        m = re.search(r"####\s*(.+)", ex["answer"])
        if not m:
            continue
        gt = m.group(1).strip().replace(",", "")
        examples.append(
            {
                "question": ex["question"],
                "ground_truth": gt,
                "prompt": (
                    "<|im_start|>user\n"
                    + ex["question"]
                    + "\n<|im_end|><|im_start|>assistant\n<|think_start|>"
                ),
            }
        )
    random.Random(seed).shuffle(examples)
    return examples[:max_examples] if max_examples else examples


def load_local_math(path, seed=42, max_examples=None):
    """Load a local JSONL with {query, ground_truth|answer} fields."""
    import json
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    random.Random(seed).shuffle(rows)
    if max_examples:
        rows = rows[:max_examples]
    examples = []
    for row in rows:
        query = row["query"]
        gt = str(row.get("ground_truth", row.get("answer", "")))
        examples.append({
            "question": query,
            "ground_truth": gt,
            "prompt": (
                "<|im_start|>user\n"
                + query
                + "\n<|im_end|><|im_start|>assistant\n<|think_start|>"
            ),
        })
    print(f"  local_math: {len(examples)} exemples chargés depuis {path}")
    return examples


_GSM8K_NUM_PATTERN = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_GSM8K_NUM_RE = re.compile(_GSM8K_NUM_PATTERN)
_GSM8K_ANSWER_PATTERNS = [
    re.compile(rf"####\s*({_GSM8K_NUM_PATTERN})"),
    re.compile(
        rf"(?:final\s+answer|answer|solution|result|total)\s*(?:is|=|:)\s*\$?\s*({_GSM8K_NUM_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:therefore|thus|so)\s*(?:the\s+)?(?:answer|solution|result)?\s*(?:is|=|:)?\s*\$?\s*({_GSM8K_NUM_PATTERN})",
        re.IGNORECASE,
    ),
]


def _normalize_numeric_string(raw: str):
    value = raw.strip().replace(",", "")
    return value.rstrip(".")


def _numeric_search_zones(text: str):
    stripped = text.strip()
    zones = []
    parts = stripped.split("<|think_end|>")
    if len(parts) > 1:
        answer_zone = parts[-1].strip()
        if answer_zone:
            zones.append(answer_zone)
        return zones
    if stripped:
        zones.append(stripped)
    return zones


def _extract_answer_after_thinking(text: str) -> str:
    if "<|think_end|>" in text:
        text = text.split("<|think_end|>")[-1]
    elif "</think>" in text:
        text = text.split("</think>")[-1]
    elif "<|think_start|>" in text:
        before = text.split("<|think_start|>")[0].strip()
        if before:
            text = before
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    return text.strip()


def _extract_numeric_candidates(text: str):
    candidates = []
    seen = set()
    for zone in _numeric_search_zones(text):
        if "\\boxed{" in zone:
            boxed_zone = zone.split("\\boxed{")[-1].split("}", 1)[0]
            boxed_match = _GSM8K_NUM_RE.search(boxed_zone)
            if boxed_match is not None:
                value = _normalize_numeric_string(boxed_match.group(0))
                if value and value not in seen:
                    seen.add(value)
                    candidates.append(value)
        for pattern in _GSM8K_ANSWER_PATTERNS:
            match = pattern.search(zone)
            if match is not None:
                value = _normalize_numeric_string(match.group(1))
                if value and value not in seen:
                    seen.add(value)
                    candidates.append(value)
        for match in _GSM8K_NUM_RE.finditer(zone):
            value = _normalize_numeric_string(match.group(0))
            if value and value not in seen:
                seen.add(value)
                candidates.append(value)
    return candidates


def _extract_numeric_answer(text: str, answer_last=None):
    for zone in _numeric_search_zones(text):
        if "\\boxed{" in zone:
            boxed_zone = zone.split("\\boxed{")[-1].split("}", 1)[0]
            boxed_match = _GSM8K_NUM_RE.search(boxed_zone)
            if boxed_match is not None:
                return _normalize_numeric_string(boxed_match.group(0))
        for pattern in _GSM8K_ANSWER_PATTERNS:
            match = pattern.search(zone)
            if match is not None:
                return _normalize_numeric_string(match.group(1))
    candidates = _extract_numeric_candidates(text)
    if not candidates:
        return None
    if answer_last is None:
        idx = -1 if "<|think_end|>" in text else 0
    else:
        idx = -1 if answer_last else 0
    return candidates[idx]


def _answer_zone(text: str) -> str:
    """Return the answer zone: text after the last <|think_end|>, or full text if absent."""
    parts = text.split("<|think_end|>")
    if len(parts) > 1:
        zone = parts[-1].strip()
        if zone:
            return zone
    return text


def check_answer(text: str, ground_truth: str) -> bool:
    answer_text = _extract_answer_after_thinking(text)
    extracted = _extract_numeric_answer(answer_text)
    if extracted is None:
        return False
    try:
        return float(extracted) == float(ground_truth)
    except ValueError:
        return False


def _partial_answer_score(text: str, ground_truth: str) -> float:
    """Return a partial reward in [0, 1) for near-correct numerical answers.

    Only called when check_answer() already returned False (wrong answer).
    Extracts the best candidate number from the completion and compares it to
    ground_truth numerically.

    Rewards:
        0.5   within  5%  (likely a rounding / unit error)
        0.2   within 20%  (right order of magnitude, rough attempt)
        0.0   otherwise
    """
    try:
        gt = float(ground_truth.replace(",", ""))
    except ValueError:
        return 0.0
    if gt == 0:
        return 0.0

    answer_text = _extract_answer_after_thinking(text)
    raw = _extract_numeric_answer(answer_text)
    if raw is None:
        return 0.0
    try:
        val = float(raw)
    except ValueError:
        return 0.0
    ratio = val / gt
    if 0.95 <= ratio <= 1.05:
        return 0.5
    if 0.80 <= ratio <= 1.20:
        return 0.2
    return 0.0


# ═════════════════════════════════════════════════════════════════════════════
# ★  REWARD FUNCTION  —  modify this to change what the model is rewarded for
# ═════════════════════════════════════════════════════════════════════════════


def score_output(
    question: str,
    completions: List[str],
    ground_truth: str,
    completions_tokens: Optional[List[List[int]]] = None,
    conciseness_alpha: float = 0.05,
    return_components: bool = False,
):
    """Score a group of completions for one question.

    Base rewards:
        1.0   correct answer (exact match)
        0.5   within  5% of correct  (rounding / unit error)
        0.2   within 20% of correct  (right ballpark)
        0.0   wrong answer

    binary (exact match only) drives solve_rate and GRPO-LEAD weighting.
    Partial scores provide gradient signal even when solve_rate=0.

    Conciseness (GRPO-LEAD style, applied when completions_tokens provided):
        Correct answers shorter than group-correct mean get a bonus,
        longer ones get a penalty.  r_correct *= exp(-alpha * (len - mu) / sigma)
    """
    binary = []
    partial = []
    rewards = []
    for c in completions:
        correct = 1.0 if check_answer(c, ground_truth) else 0.0
        part = 0.0 if correct else _partial_answer_score(c, ground_truth)
        binary.append(correct)
        partial.append(part)
        rewards.append(correct + part)

    # Conciseness scaling for correct answers (GRPO-LEAD)
    if completions_tokens is not None and conciseness_alpha > 0:
        correct_lens = [
            len(completions_tokens[i])
            for i in range(len(completions))
            if binary[i] > 0
        ]
        if len(correct_lens) >= 2:
            mu = np.mean(correct_lens)
            sigma = max(np.std(correct_lens), 1.0)
            for i in range(len(rewards)):
                if binary[i] > 0:
                    z = (len(completions_tokens[i]) - mu) / sigma
                    # exp(-alpha * z): shorter → bonus, longer → penalty
                    rewards[i] = rewards[i] * math.exp(-conciseness_alpha * z)

    if return_components:
        return {
            "rewards": rewards,
            "binary": binary,
            "partial": partial,
        }
    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# GRPO internals
# ─────────────────────────────────────────────────────────────────────────────


def _seq_token_log_probs(model, input_ids_t, prompt_len):
    """Per-token log probs for the completion portion of input_ids (PyTorch tensor)."""
    logits = model(input_ids_t)          # (1, T, vocab)
    log_probs = F.log_softmax(logits, dim=-1)
    shift = log_probs[0, prompt_len - 1:-1, :]          # (comp_len, vocab)
    targets = input_ids_t[0, prompt_len:].long().unsqueeze(-1)  # (comp_len, 1)
    return shift.gather(-1, targets).squeeze(-1)         # (comp_len,)


def _seq_log_prob(model, input_ids_t, prompt_len):
    """Sum of per-token log probs for the completion portion."""
    return _seq_token_log_probs(model, input_ids_t, prompt_len).sum()


def _repetition_penalty(
    token_ids: List[int], n: int = 4, threshold: float = 0.5,
    think_end_id: Optional[int] = None,
) -> float:
    """Penalize completions with excessive n-gram repetition (degenerate loops).

    Checks the ENTIRE completion for repetition (loops often happen during
    thinking, before <|think_end|> is ever produced).
    Returns 0.0 if unique_ratio >= threshold (healthy text),
    linearly ramps to -1.0 as unique_ratio approaches 0.
    Short completions (< 2*n tokens) are never penalized.
    """
    if len(token_ids) < 2 * n:
        return 0.0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    unique_ratio = len(set(ngrams)) / len(ngrams)
    if unique_ratio >= threshold:
        return 0.0
    return -(1.0 - unique_ratio / threshold)


def _compute_advantages(rewards, normalize_std: bool = True):
    """Group-relative advantages: (r - mean) / (std + eps), or just (r - mean) with normalize_std=False (Dr. GRPO)."""
    r = np.array(rewards, dtype=np.float32)
    centered = r - r.mean()
    if not normalize_std:
        return centered
    std = r.std()
    return np.zeros_like(r) if std < 1e-8 else centered / std


def _filter_group_by_total_length(
    prompt_tokens: List[int],
    texts: List[str],
    completions_tokens: List[List[int]],
    max_total_tokens: Optional[int],
):
    if not max_total_tokens:
        return texts, completions_tokens, 0
    kept_texts = []
    kept_tokens = []
    skipped = 0
    prompt_len = len(prompt_tokens)
    for text, comp in zip(texts, completions_tokens):
        if prompt_len + len(comp) > max_total_tokens:
            skipped += 1
            continue
        kept_texts.append(text)
        kept_tokens.append(comp)
    return kept_texts, kept_tokens, skipped


def _ids_tensor(prompt_tokens, comp, device):
    """Build a (1, T) long tensor from prompt + completion token lists."""
    return torch.tensor([prompt_tokens + comp], dtype=torch.long, device=device)


def _compute_old_log_probs(model, prompt_tokens, completions_tokens, device):
    """Compute detached per-token log probs for all completions (π_old)."""
    prompt_len = len(prompt_tokens)
    old_lps = []
    with torch.no_grad():
        for comp in completions_tokens:
            if len(comp) == 0:
                old_lps.append(None)
                continue
            ids = _ids_tensor(prompt_tokens, comp, device)
            token_lps = _seq_token_log_probs(model, ids, prompt_len)
            old_lps.append(token_lps.detach().cpu().numpy().copy())
    return old_lps


def _accumulate_loss_term(acc, term):
    if term is None:
        return acc
    return term if acc is None else acc + term


def _grad_norm_sq(loss, params):
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    total = None
    for grad in grads:
        if grad is None:
            continue
        cur = grad.detach().pow(2).sum()
        total = cur if total is None else total + cur
    if total is None:
        return 0.0
    return float(total.item())


def _apply_balanced_backward(pos_loss, neg_loss, params, grad_scale: float, neg_scale: float):
    if pos_loss is None and neg_loss is None:
        return 1.0, 0.0, 0.0
    if neg_scale <= 0 or pos_loss is None or neg_loss is None:
        total_loss = _accumulate_loss_term(pos_loss, neg_loss)
        (total_loss * grad_scale).backward()
        return 1.0, 0.0, 0.0

    pos_norm_sq = _grad_norm_sq(pos_loss, params)
    neg_norm_sq = _grad_norm_sq(neg_loss, params)
    neg_mult = 1.0
    if pos_norm_sq > 0.0 and neg_norm_sq > pos_norm_sq * (neg_scale ** 2):
        neg_mult = (math.sqrt(pos_norm_sq) * neg_scale) / math.sqrt(neg_norm_sq)

    total_loss = pos_loss + neg_loss * neg_mult
    (total_loss * grad_scale).backward()
    return neg_mult, math.sqrt(pos_norm_sq), math.sqrt(neg_norm_sq)


# ═════════════════════════════════════════════════════════════════════════════
# ★  GRPO FORMULA  —  modify this to change how rewards drive the update
# ═════════════════════════════════════════════════════════════════════════════


def grpo_step_ppo(
    model,
    prompt_tokens: List[int],
    completions_tokens: List[List[int]],
    advantages: np.ndarray,
    old_token_log_probs: List,
    *,
    clip_eps_low: float = 0.2,
    clip_eps_high: float = 0.28,
    beta: float = 0.0,
    ref_model=None,
    grad_scale: float = 1.0,
    llds_lambda: float = 0.0,
    neg_scale: float = 0.0,
    device: str = "cuda",
) -> float:
    """PPO-clip GRPO (DAPO asymmetric clipping). Pure PyTorch."""
    prompt_len = len(prompt_tokens)
    total_tokens = sum(len(c) for c in completions_tokens if len(c) > 0)
    if total_tokens == 0:
        return 0.0

    ref_avg_lps = []
    if beta > 0:
        ref = ref_model if ref_model is not None else model
        with torch.no_grad():
            for comp in completions_tokens:
                if len(comp) == 0:
                    ref_avg_lps.append(0.0)
                    continue
                ids = _ids_tensor(prompt_tokens, comp, device)
                ref_avg_lps.append(float(_seq_log_prob(ref, ids, prompt_len).item()) / len(comp))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    pos_loss = None
    neg_loss = None
    total_loss = 0.0
    for i, comp in enumerate(completions_tokens):
        if len(comp) == 0 or advantages[i] == 0 or old_token_log_probs[i] is None:
            continue
        ids = _ids_tensor(prompt_tokens, comp, device)
        cur_token_lps = _seq_token_log_probs(model, ids, prompt_len)
        old_lps_t = torch.tensor(old_token_log_probs[i], device=device)

        log_ratio = cur_token_lps - old_lps_t
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_high)
        adv_i = float(advantages[i])
        ppo_obj = torch.minimum(ratio * adv_i, clipped_ratio * adv_i)
        token_weight = len(comp) / total_tokens
        loss_i = -ppo_obj.mean() * token_weight

        if beta > 0:
            kl = cur_token_lps.sum() / len(comp) - ref_avg_lps[i]
            loss_i = loss_i + beta * kl * token_weight

        if adv_i >= 0:
            pos_loss = _accumulate_loss_term(pos_loss, loss_i)
        else:
            neg_loss = _accumulate_loss_term(neg_loss, loss_i)
        total_loss += loss_i.item()

    _apply_balanced_backward(pos_loss, neg_loss, trainable_params, grad_scale, neg_scale)

    if llds_lambda > 0:
        for i, comp in enumerate(completions_tokens):
            if len(comp) == 0 or advantages[i] < 0 or old_token_log_probs[i] is None:
                continue
            ids = _ids_tensor(prompt_tokens, comp, device)
            cur_token_lps = _seq_token_log_probs(model, ids, prompt_len)
            old_lps_t = torch.tensor(old_token_log_probs[i], device=device)
            if cur_token_lps.sum().item() >= old_lps_t.sum().item():
                continue
            drop_masked = torch.clamp(old_lps_t - cur_token_lps, min=0.0)
            token_weight = len(comp) / total_tokens
            llds_loss = llds_lambda * drop_masked.mean() * token_weight
            (llds_loss * grad_scale).backward()
            total_loss += llds_loss.item()

    return total_loss


def grpo_step(
    model,
    prompt_tokens: List[int],
    completions_tokens: List[List[int]],
    advantages: np.ndarray,
    *,
    beta: float = 0.1,
    ref_model=None,
    grad_scale: float = 1.0,
    llds_lambda: float = 0.0,
    neg_scale: float = 0.0,
    device: str = "cuda",
) -> float:
    """REINFORCE GRPO with KL penalty. Pure PyTorch."""
    prompt_len = len(prompt_tokens)
    total_tokens = sum(len(c) for c in completions_tokens if len(c) > 0)
    if total_tokens == 0:
        return 0.0

    ref = ref_model if ref_model is not None else model
    ref_avg_lps = []
    with torch.no_grad():
        for comp in completions_tokens:
            if len(comp) == 0:
                ref_avg_lps.append(0.0)
                continue
            ids = _ids_tensor(prompt_tokens, comp, device)
            ref_avg_lps.append(float(_seq_log_prob(ref, ids, prompt_len).item()) / len(comp))

    old_token_lps_list = []
    if llds_lambda > 0:
        with torch.no_grad():
            for comp in completions_tokens:
                if len(comp) == 0:
                    old_token_lps_list.append(None)
                    continue
                ids = _ids_tensor(prompt_tokens, comp, device)
                old_token_lps_list.append(_seq_token_log_probs(model, ids, prompt_len).detach().cpu().numpy().copy())

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    pos_loss = None
    neg_loss = None
    total_loss = 0.0
    for i, comp in enumerate(completions_tokens):
        if len(comp) == 0 or advantages[i] == 0:
            continue
        ids = _ids_tensor(prompt_tokens, comp, device)
        policy_lp = _seq_log_prob(model, ids, prompt_len)
        avg_lp = policy_lp / len(comp)
        kl = avg_lp - ref_avg_lps[i]
        token_weight = len(comp) / total_tokens
        loss_i = (-advantages[i] * avg_lp + beta * kl) * token_weight
        if advantages[i] >= 0:
            pos_loss = _accumulate_loss_term(pos_loss, loss_i)
        else:
            neg_loss = _accumulate_loss_term(neg_loss, loss_i)
        total_loss += loss_i.item()

    _apply_balanced_backward(pos_loss, neg_loss, trainable_params, grad_scale, neg_scale)

    if llds_lambda > 0:
        for i, comp in enumerate(completions_tokens):
            if len(comp) == 0 or advantages[i] < 0 or old_token_lps_list[i] is None:
                continue
            ids = _ids_tensor(prompt_tokens, comp, device)
            cur_token_lps = _seq_token_log_probs(model, ids, prompt_len)
            old_lps_t = torch.tensor(old_token_lps_list[i], device=device)
            if cur_token_lps.sum().item() >= old_lps_t.sum().item():
                continue
            drop_masked = torch.clamp(old_lps_t - cur_token_lps, min=0.0)
            token_weight = len(comp) / total_tokens
            llds_loss = llds_lambda * drop_masked.mean() * token_weight
            (llds_loss * grad_scale).backward()
            total_loss += llds_loss.item()

    return total_loss


def gmpo_step(
    model,
    prompt_tokens: List[int],
    completions_tokens: List[List[int]],
    advantages: np.ndarray,
    *,
    clip_eps: float = 0.4,
    beta: float = 0.05,
    ref_model=None,
    grad_scale: float = 1.0,
    neg_scale: float = 0.0,
    device: str = "cuda",
) -> float:
    """GMPO: geometric-mean policy optimization. Pure PyTorch."""
    prompt_len = len(prompt_tokens)
    total_tokens = sum(len(c) for c in completions_tokens if len(c) > 0)
    if total_tokens == 0:
        return 0.0

    ref = ref_model if ref_model is not None else model
    old_token_lps = []
    ref_avg_lps = []
    with torch.no_grad():
        for comp in completions_tokens:
            if len(comp) == 0:
                old_token_lps.append(None)
                ref_avg_lps.append(0.0)
                continue
            ids = _ids_tensor(prompt_tokens, comp, device)
            old_token_lps.append(_seq_token_log_probs(model, ids, prompt_len).detach())
            ref_avg_lps.append(float(_seq_log_prob(ref, ids, prompt_len).item()) / len(comp))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    pos_loss = None
    neg_loss = None
    total_loss = 0.0
    for i, comp in enumerate(completions_tokens):
        if len(comp) == 0 or advantages[i] == 0:
            continue
        ids = _ids_tensor(prompt_tokens, comp, device)
        new_token_lps = _seq_token_log_probs(model, ids, prompt_len)
        old_lps = old_token_lps[i]

        adv = float(advantages[i])
        sign_a = 1.0 if adv > 0 else -1.0
        token_log_ratios = new_token_lps - old_lps
        signed_log_ratios = sign_a * token_log_ratios
        pessimistic = torch.minimum(signed_log_ratios, torch.tensor(clip_eps, device=device))
        seq_ratio = torch.exp(pessimistic.mean())
        kl = new_token_lps.mean() - ref_avg_lps[i]

        token_weight = len(comp) / total_tokens
        loss_i = (-adv * seq_ratio + beta * kl) * token_weight
        if adv >= 0:
            pos_loss = _accumulate_loss_term(pos_loss, loss_i)
        else:
            neg_loss = _accumulate_loss_term(neg_loss, loss_i)
        total_loss += loss_i.item()

    _apply_balanced_backward(pos_loss, neg_loss, trainable_params, grad_scale, neg_scale)
    return total_loss


def gmpo_step_ppo(
    model,
    prompt_tokens: List[int],
    completions_tokens: List[List[int]],
    advantages: np.ndarray,
    old_token_log_probs: List,
    *,
    clip_eps: float = 0.4,
    beta: float = 0.0,
    ref_model=None,
    grad_scale: float = 1.0,
    neg_scale: float = 0.0,
    device: str = "cuda",
) -> float:
    """Multi-epoch GMPO with stored π_old. Pure PyTorch."""
    prompt_len = len(prompt_tokens)
    total_tokens = sum(len(c) for c in completions_tokens if len(c) > 0)
    if total_tokens == 0:
        return 0.0

    ref_avg_lps = []
    if beta > 0:
        ref = ref_model if ref_model is not None else model
        with torch.no_grad():
            for comp in completions_tokens:
                if len(comp) == 0:
                    ref_avg_lps.append(0.0)
                    continue
                ids = _ids_tensor(prompt_tokens, comp, device)
                ref_avg_lps.append(float(_seq_log_prob(ref, ids, prompt_len).item()) / len(comp))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    pos_loss = None
    neg_loss = None
    total_loss = 0.0
    for i, comp in enumerate(completions_tokens):
        if len(comp) == 0 or advantages[i] == 0 or old_token_log_probs[i] is None:
            continue
        ids = _ids_tensor(prompt_tokens, comp, device)
        new_token_lps = _seq_token_log_probs(model, ids, prompt_len)
        old_lps_t = torch.tensor(old_token_log_probs[i], device=device)

        adv = float(advantages[i])
        sign_a = 1.0 if adv > 0 else -1.0
        token_log_ratios = new_token_lps - old_lps_t
        signed_log_ratios = sign_a * token_log_ratios
        pessimistic = torch.minimum(signed_log_ratios, torch.tensor(clip_eps, device=device))
        seq_ratio = torch.exp(pessimistic.mean())
        token_weight = len(comp) / total_tokens
        loss_i = -adv * seq_ratio * token_weight

        if beta > 0:
            kl = new_token_lps.mean() - ref_avg_lps[i]
            loss_i = loss_i + beta * kl * token_weight

        if adv >= 0:
            pos_loss = _accumulate_loss_term(pos_loss, loss_i)
        else:
            neg_loss = _accumulate_loss_term(neg_loss, loss_i)
        total_loss += loss_i.item()

    _apply_balanced_backward(pos_loss, neg_loss, trainable_params, grad_scale, neg_scale)
    return total_loss


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────


def load_model(checkpoint_path: str):
    """Load a single TorchGenerator (model + config). Returns (model, config, torch_gen)
    where model IS torch_gen.model — no Keras, no sync needed."""
    torch_gen = TorchGenerator(checkpoint_path, device="cuda")
    config = torch_gen.config
    n = sum(p.numel() for p in torch_gen.model.parameters())
    print(f"Loaded {checkpoint_path}  ({n:,} params, PyTorch)")
    return torch_gen.model, config, torch_gen


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(
    torch_gen,
    examples,
    max_examples=100,
    num_completions=1,
    max_new_tokens=512,
    temperature=0.0,
    rep_penalty: float = 1.0,
    gen_batch_size=4,
    eval_batch_size=None,
    step=None,
    log_path=None,
):
    """Evaluate pass@k accuracy on the first max_examples problems.

    All k*N prompts are flattened into a single queue and dispatched in batches
    of `eval_batch_size` (falls back to `gen_batch_size`). This is much faster
    than looping pass-by-pass because every generate_batch call uses the full
    GPU capacity regardless of how many examples remain in the current pass.
    """
    examples = examples[:max_examples]
    t0 = time.time()
    was_training = torch_gen.model.training
    torch_gen.model.eval()

    eval_bs = eval_batch_size or gen_batch_size
    N = len(examples)
    flat_prompts = [ex["prompt"] for ex in examples for _ in range(num_completions)]
    flat_idx = [(i, k) for i in range(N) for k in range(num_completions)]

    all_comps = [[None] * num_completions for _ in range(N)]
    for start in range(0, len(flat_prompts), eval_bs):
        chunk_prompts = flat_prompts[start:start + eval_bs]
        chunk_idx = flat_idx[start:start + eval_bs]
        texts = torch_gen.generate_batch(
            chunk_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=10,
            use_synth_template=False,
        )
        for (i, k), c in zip(chunk_idx, texts):
            all_comps[i][k] = c

    if was_training:
        torch_gen.model.train()

    torch.cuda.empty_cache()
    correct = sum(
        int(any(check_answer(c, ex["ground_truth"]) for c in comps))
        for ex, comps in zip(examples, all_comps)
    )
    acc = correct / len(examples)
    elapsed = time.time() - t0
    step_str = f"step={step}  " if step is not None else ""
    log_line = (
        f"{step_str}pass@{num_completions}: {100*acc:.1f}%  "
        f"correct={correct}/{len(examples)}  ({elapsed:.0f}s)"
    )
    print(f"\n  {log_line}\n")
    if log_path:
        with open(log_path, "a") as _lf:
            _lf.write(log_line + "\n")
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────


def _save_stepped(torch_gen, config, save_path, step):
    """Save a stepped checkpoint next to save_path (e.g. foo_step00064.pt)."""
    base, ext = os.path.splitext(save_path)
    stepped = f"{base}_step{step:05d}{ext}"
    try:
        save_ckpt(torch_gen, config, stepped, step)
    except OSError as e:
        print(f"WARNING: could not save stepped checkpoint ({e})")


def train_grpo(
    model,
    config,
    examples,
    *,
    eval_examples=None,
    torch_gen=None,
    use_gmpo: bool = False,
    num_completions: int = 6,
    max_new_tokens: int = 512,
    temperature: float = 0.5,
    rep_penalty: float = 1.0,
    gen_batch_size: int = 4,
    eval_batch_size: int = 0,
    beta: float = 0.05,
    lr: float = 5e-5,
    optimizer_name: str = "adamw",
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    train_layers: int = 3,
    warmup_steps: int = 20,
    min_completion_tokens: int = 5,
    num_steps: int = 250,
    log_every: int = 1,
    eval_every: int = 50,
    max_eval: int = 100,
    eval_num_completions: int = 1,
    eval_temperature: float = 0.0,
    save_every: int = 0,
    save_path: str = None,
    grad_accum_steps: int = 4,
    seed: int = 42,
    use_dr_grpo: bool = True,
    clip_adv: float = 5.0,
    reshuffle_layers_every: int = 4,
    lr_decay: str = "linear",
    lr_final_ratio: float = 0.0,
    ppo_epochs: int = 1,
    clip_eps_low: float = 0.2,
    clip_eps_high: float = 0.28,
    gmpo_clip_eps: float = 0.4,
    llds_lambda: float = 0.0,
    min_active_steps: int = 1,
    neg_scale: float = 0.0,
):
    device = next(model.parameters()).device

    tokenizer = Tokenizer()
    np.random.seed(seed)
    random.seed(seed)

    n_blocks = len(model.blocks)
    train_layers = min(train_layers, n_blocks)

    all_params = list(model.parameters())
    optimizer = (
        torch.optim.AdamW(all_params, lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
        if optimizer_name == "adamw"
        else torch.optim.SGD(all_params, lr=lr, momentum=0.0, weight_decay=weight_decay)
    )

    max_total_tokens = int(config.get("maxlen") or 0)

    # Frozen reference model for KL penalty (deepcopy of initial PyTorch model)
    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    # SeqCondAttention lazily caches tensors derived from trainable params
    # (_theta_cached, _w_int_cached, _decay_slopes_cached, _anchor_slopes_cached,
    #  _score_scale_b, _score_bias_b, _phase_scale_b, _conv_kernel_t). These are
    # NEVER recomputed once set, so after an optimizer.step() they become stale
    # while step() (used by torch_gen.generate_batch) keeps using them. Must
    # invalidate after every param update to keep generation consistent with
    # training forward path — otherwise the checkpoint saved to disk (which
    # omits these buffers) will be decoded with freshly recomputed caches and
    # diverge from the live training-eval behavior.
    _CACHE_ATTRS = (
        "_theta_cached", "_w_int_cached", "_decay_slopes_cached",
        "_anchor_slopes_cached", "_score_scale_b", "_score_bias_b",
        "_phase_scale_b", "_conv_kernel_t",
    )

    def _invalidate_seqcond_caches(m):
        for sub in m.modules():
            for a in _CACHE_ATTRS:
                if hasattr(sub, a) and getattr(sub, a) is not None:
                    setattr(sub, a, None)

    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")
    if lr_decay not in {"constant", "linear"}:
        raise ValueError(f"lr_decay must be 'constant' or 'linear', got {lr_decay}")
    if not 0.0 <= lr_final_ratio <= 1.0:
        raise ValueError(
            f"lr_final_ratio must be between 0 and 1, got {lr_final_ratio}"
        )

    objective_name = "GMPO" if use_gmpo else "GRPO"

    if ppo_epochs > 1:
        _eps_str = f"gmpo_ε={gmpo_clip_eps}" if use_gmpo else f"ε=[{clip_eps_low},{clip_eps_high}]"
        ppo_tag = f"  ppo_epochs={ppo_epochs} {_eps_str}"
    else:
        ppo_tag = ""
    print(
        f"\n── {objective_name}  {num_steps} steps  G={num_completions}  "
        f"lr={lr}  β={beta}  train_layers={train_layers}/{n_blocks}  "
        f"warmup={warmup_steps}  decay={lr_decay}:{lr_final_ratio:.2f}  "
        f"accum={grad_accum_steps}  dr_grpo={use_dr_grpo}  llds_λ={llds_lambda}{ppo_tag} ──\n"
    )

    t0 = time.time()
    run_reward = run_loss = run_correct = run_skipped = 0.0
    run_count = run_adv_abs = 0.0
    pending_grad_steps = 0
    active_grad_steps = 0  # steps that actually contributed a gradient in current window
    optimizer_updates = 0
    ppo_batch = []
    optimizer.zero_grad()

    def _ppo_replay_step(_pt, _ct, _adv, _olp, _grad_scale):
        """Dispatch a single PPO replay group to GMPO or GRPO objective."""
        if use_gmpo:
            return gmpo_step_ppo(
                model, _pt, _ct, _adv, _olp,
                clip_eps=gmpo_clip_eps,
                beta=beta, ref_model=ref_model,
                grad_scale=_grad_scale,
                neg_scale=neg_scale,
                device=str(device),
            )
        else:
            return grpo_step_ppo(
                model, _pt, _ct, _adv, _olp,
                clip_eps_low=clip_eps_low,
                clip_eps_high=clip_eps_high,
                beta=beta, ref_model=ref_model,
                grad_scale=_grad_scale,
                llds_lambda=llds_lambda,
                neg_scale=neg_scale,
                device=str(device),
            )

    def _set_active_trainable_blocks(block_indices):
        for p in model.parameters():
            p.requires_grad = False
        for idx in block_indices:
            for p in model.blocks[idx].parameters():
                p.requires_grad = True
        # Embeddings FROZEN — tied to lm_head (weight tying), training them
        # causes progressive regression (confirmed: grad_norm escalation + eval drop).

    def _set_optimizer_lr(current_step):
        current_lr = lr
        if warmup_steps > 0 and current_step <= warmup_steps:
            current_lr = lr * (current_step / warmup_steps)
        elif lr_decay == "linear":
            decay_start = warmup_steps + 1 if warmup_steps > 0 else 1
            if num_steps > decay_start:
                progress = (current_step - decay_start) / (num_steps - decay_start)
                progress = min(max(progress, 0.0), 1.0)
                current_lr = lr * (1.0 - (1.0 - lr_final_ratio) * progress)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr
        return current_lr

    active_block_indices = sorted(random.sample(range(n_blocks), train_layers))
    _set_active_trainable_blocks(active_block_indices)

    for step in range(1, num_steps + 1):
        ex = random.choice(examples)
        prompt_tokens = tokenizer([ex["prompt"]])[0]

        # Generate using the single PyTorch model (via TorchGenerator).
        # IMPORTANT: request return_token_ids=True to get the *actual* tokens
        # the model produced. Decoding to text and re-encoding is NOT identity
        # with tiktoken (special tokens, whitespace, multi-byte boundaries), so
        # using re-encoded ids would feed GRPO gradient on sequences that
        # differ from what the model actually sampled — producing an incorrect
        # log-prob / ratio / advantage mapping and silently corrupting the
        # policy update.
        was_training = model.training
        model.eval()
        texts, ids = torch_gen.generate_batch(
            [ex["prompt"]] * num_completions,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_synth_template=False,
            return_token_ids=True,
        )
        if was_training:
            model.train()

        # Filter out degenerate completions (too short to be meaningful)
        valid = [len(c) >= min_completion_tokens for c in ids]
        texts_f = [t for t, v in zip(texts, valid) if v]
        ids_f = [c for c, v in zip(ids, valid) if v]
        if not texts_f:
            run_skipped += 1
            continue
        texts_f, ids_f, skipped_too_long = _filter_group_by_total_length(
            prompt_tokens,
            texts_f,
            ids_f,
            max_total_tokens,
        )
        if skipped_too_long:
            print(
                f"  skipped {skipped_too_long} completions exceeding model maxlen={max_total_tokens}"
            )
            run_skipped += skipped_too_long
        if not texts_f:
            print("  skipped step: all sampled completions exceeded model maxlen")
            continue

        # Score
        score_info = score_output(
            ex["question"],
            texts_f,
            ex["ground_truth"],
            completions_tokens=ids_f,
            return_components=True,
        )
        rewards = score_info["rewards"]
        binary = score_info["binary"]

        # Repetition penalty (degenerate loops anywhere in the completion)
        for _i in range(len(ids_f)):
            _rep = _repetition_penalty(ids_f[_i])
            if _rep != 0.0:
                rewards[_i] = rewards[_i] + _rep

        # Skip gradient when there's no contrastive signal.
        # Don't skip if repetition penalties create variance (even at solve_rate=0%).
        solve_rate = sum(binary) / len(binary)
        has_rep_variance = len(set(round(r, 4) for r in rewards)) > 1
        skip_gradient = (solve_rate == 1.0) or (solve_rate == 0.0 and not has_rep_variance)

        # Always count toward the accumulation window (even skipped steps).
        # This ensures optimizer.step fires every grad_accum_steps steps.
        pending_grad_steps += 1

        if skip_gradient:
            run_reward += sum(rewards)
            run_count += len(rewards)
            run_correct += int(sum(binary))
            run_skipped += 1
        else:
            advantages = _compute_advantages(rewards, normalize_std=not use_dr_grpo)
            # GRPO-LEAD asymmetric advantage re-weighting by ρ_G (group solve-rate)
            # w_i = ω(ρ_G) if A_i ≥ 0, w_i = ω(1-ρ_G) if A_i < 0
            _k1, _k2, _k3, _rho0 = 0.4, 1.1, 10.0, 0.75
            _w_pos = _k1 + (_k2 - _k1) / (1.0 + math.exp(-_k3 * (solve_rate - _rho0)))
            _w_neg = _k1 + (_k2 - _k1) / (1.0 + math.exp(-_k3 * ((1.0 - solve_rate) - _rho0)))
            _weights = np.where(advantages >= 0, _w_pos, _w_neg)
            advantages = advantages * _weights
            if clip_adv > 0:
                advantages = np.clip(advantages, -clip_adv, clip_adv)
            run_reward += sum(rewards)
            run_count += len(rewards)
            run_correct += int(sum(binary))
            run_adv_abs += float(np.mean(np.abs(advantages)))

            if np.all(advantages == 0):
                run_skipped += 1
            else:
                # Policy update — scale by actual contributing steps
                _actual_grad_scale = 1.0 / grad_accum_steps
                if ppo_epochs > 1:
                    # Multi-epoch PPO: store group + π_old for later
                    old_lps = _compute_old_log_probs(model, prompt_tokens, ids_f, device)
                    ppo_batch.append((prompt_tokens, ids_f, advantages, old_lps))
                    active_grad_steps += 1
                    # Approximate loss for logging using π_old
                    _approx = sum(
                        -advantages[j] * float(np.mean(old_lps[j]))
                        for j in range(len(ids_f))
                        if old_lps[j] is not None and advantages[j] != 0
                    ) / max(len(ids_f), 1)
                    run_loss += _approx
                elif use_gmpo:
                    loss = gmpo_step(
                        model,
                        prompt_tokens,
                        ids_f,
                        advantages,
                        beta=beta,
                        ref_model=ref_model,
                        grad_scale=_actual_grad_scale,
                        neg_scale=neg_scale,
                        device=str(device),
                    )
                    run_loss += loss
                    active_grad_steps += 1
                else:
                    loss = grpo_step(
                        model,
                        prompt_tokens,
                        ids_f,
                        advantages,
                        beta=beta,
                        ref_model=ref_model,
                        grad_scale=_actual_grad_scale,
                        llds_lambda=llds_lambda,
                        neg_scale=neg_scale,
                        device=str(device),
                    )
                    run_loss += loss
                    active_grad_steps += 1

            if pending_grad_steps >= grad_accum_steps:
                if active_grad_steps < min_active_steps:
                    # Not enough real gradient signal — discard accumulated gradients
                    print(
                        f"  [skip update: only {active_grad_steps}/{grad_accum_steps} active steps "
                        f"(min={min_active_steps}) — resetting]"
                    )
                    optimizer.zero_grad()
                    ppo_batch = []
                    pending_grad_steps = 0
                    active_grad_steps = 0
                elif ppo_epochs > 1 and ppo_batch:
                    # ── Multi-epoch PPO/GMPO update ──
                    for _epoch in range(ppo_epochs):
                        optimizer.zero_grad()
                        for _pt, _ct, _adv, _olp in ppo_batch:
                            _ppo_replay_step(_pt, _ct, _adv, _olp, 1.0 / len(ppo_batch))
                        _set_optimizer_lr(step)
                        pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=max_grad_norm
                        )
                        n_grads = sum(1 for p in model.parameters() if p.grad is not None)
                        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
                        optimizer.step()
                        _invalidate_seqcond_caches(model)
                        print(
                            f"  [optimizer.step | train_step={step} | "
                            f"ppo_epoch={_epoch+1}/{ppo_epochs} | "
                            f"grad_norm={pre_clip_norm:.4f} | "
                            f"n_grads={n_grads}/{n_trainable} | "
                            f"active={active_grad_steps}/{grad_accum_steps}]"
                        )
                    ppo_batch = []
                    optimizer.zero_grad()
                    pending_grad_steps = 0
                    active_grad_steps = 0
                    optimizer_updates += 1
                    if optimizer_updates % reshuffle_layers_every == 0:
                        active_block_indices = sorted(
                            random.sample(range(n_blocks), train_layers)
                        )
                        _set_active_trainable_blocks(active_block_indices)
                        print(f"  [reshuffle layers → {active_block_indices}]")
                else:
                    # ── Single-epoch REINFORCE update ──
                    _set_optimizer_lr(step)
                    pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=max_grad_norm
                    )
                    n_grads = sum(1 for p in model.parameters() if p.grad is not None)
                    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
                    optimizer.step()
                    _invalidate_seqcond_caches(model)
                    print(
                        f"  [optimizer.step | train_step={step} | "
                        f"grad_norm={pre_clip_norm:.4f} | "
                        f"n_grads={n_grads}/{n_trainable} | "
                        f"active={active_grad_steps}/{grad_accum_steps}]"
                    )
                    optimizer.zero_grad()
                    pending_grad_steps = 0
                    active_grad_steps = 0
                    optimizer_updates += 1
                    if optimizer_updates % reshuffle_layers_every == 0:
                        active_block_indices = sorted(
                            random.sample(range(n_blocks), train_layers)
                        )
                        _set_active_trainable_blocks(active_block_indices)
                        print(f"  [reshuffle layers → {active_block_indices}]")

        # Log
        if step % log_every == 0:
            avg_r = run_reward / max(run_count, 1.0)
            avg_adv = run_adv_abs / max(log_every, 1)
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            eta = elapsed / step * (num_steps - step)
            print(
                f"  Step {step:4d}/{num_steps} | loss={run_loss:.4f} | "
                f"reward_avg={avg_r:.3f} | adv_abs={avg_adv:.3f} | "
                f"correct={int(run_correct)}/{int(run_count)} | skip={int(run_skipped)} | "
                f"lr={current_lr:.2e} | "
                f"ETA {int(eta//60):02d}:{int(eta%60):02d}"
            )
            run_reward = run_loss = run_correct = run_skipped = 0.0
            run_count = run_adv_abs = 0.0

        # Flush pending gradients before eval
        if eval_every > 0 and step % eval_every == 0 and pending_grad_steps > 0:
            if active_grad_steps < min_active_steps:
                print(f"  [skip flush/eval: only {active_grad_steps} active steps — resetting]")
                optimizer.zero_grad()
                ppo_batch = []
            elif ppo_epochs > 1 and ppo_batch:
                for _epoch in range(ppo_epochs):
                    optimizer.zero_grad()
                    for _pt, _ct, _adv, _olp in ppo_batch:
                        _ppo_replay_step(_pt, _ct, _adv, _olp, 1.0 / len(ppo_batch))
                    _set_optimizer_lr(step)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()
                    _invalidate_seqcond_caches(model)
                    print(f"  [optimizer.step | train_step={step} | ppo_epoch={_epoch+1}/{ppo_epochs} | flush=eval]")
                ppo_batch = []
            else:
                _set_optimizer_lr(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                _invalidate_seqcond_caches(model)
                print(f"  [optimizer.step | train_step={step} | flush=eval]")
            optimizer.zero_grad()
            pending_grad_steps = 0
            active_grad_steps = 0
            active_block_indices = sorted(random.sample(range(n_blocks), train_layers))
            _set_active_trainable_blocks(active_block_indices)

        # Periodic eval
        if eval_every > 0 and step % eval_every == 0:
            log_path = save_path.replace(".pt", "_eval.log") if save_path else None
            ds = eval_examples if eval_examples is not None else examples
            # Configured eval (typically pass@K sampled at eval_temperature)
            evaluate(
                torch_gen,
                ds,
                max_examples=max_eval,
                num_completions=eval_num_completions,
                max_new_tokens=max_new_tokens,
                temperature=eval_temperature,
                rep_penalty=rep_penalty,
                gen_batch_size=gen_batch_size,
                eval_batch_size=eval_batch_size or None,
                step=step,
                log_path=log_path,
            )
            # Secondary greedy eval: pass@1 at T=0 (mode of the policy).
            # Useful to detect cases where sampled pass@K improves while the
            # greedy mode collapses (policy narrowing / deteriorating).
            if eval_num_completions != 1 or eval_temperature != 0.0:
                evaluate(
                    torch_gen,
                    ds,
                    max_examples=max_eval,
                    num_completions=1,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    rep_penalty=rep_penalty,
                    gen_batch_size=gen_batch_size,
                    eval_batch_size=eval_batch_size or None,
                    step=step,
                    log_path=log_path,
                )

            # Round-trip debug moved to sanity_check_reload.py

        # Flush pending gradients before backup
        if (
            save_every > 0
            and step % save_every == 0
            and save_path
            and pending_grad_steps > 0
        ):
            if active_grad_steps < min_active_steps:
                print(f"  [skip flush/save: only {active_grad_steps} active steps — resetting]")
                optimizer.zero_grad()
                ppo_batch = []
            elif ppo_epochs > 1 and ppo_batch:
                for _epoch in range(ppo_epochs):
                    optimizer.zero_grad()
                    for _pt, _ct, _adv, _olp in ppo_batch:
                        _ppo_replay_step(_pt, _ct, _adv, _olp, 1.0 / len(ppo_batch))
                    _set_optimizer_lr(step)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()
                    _invalidate_seqcond_caches(model)
                    print(f"  [optimizer.step | train_step={step} | ppo_epoch={_epoch+1}/{ppo_epochs} | flush=save]")
                ppo_batch = []
            else:
                _set_optimizer_lr(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                _invalidate_seqcond_caches(model)
                print(f"  [optimizer.step | train_step={step} | flush=save]")
            optimizer.zero_grad()
            pending_grad_steps = 0
            active_grad_steps = 0
            active_block_indices = sorted(random.sample(range(n_blocks), train_layers))
            _set_active_trainable_blocks(active_block_indices)

        # Periodic local checkpoint save
        if save_every > 0 and step % save_every == 0 and save_path:
            _save_stepped(torch_gen, config, save_path, step)

    # Flush any remaining accumulated gradients
    if pending_grad_steps > 0 and active_grad_steps >= min_active_steps:
        if ppo_epochs > 1 and ppo_batch:
            for _epoch in range(ppo_epochs):
                optimizer.zero_grad()
                for _pt, _ct, _adv, _olp in ppo_batch:
                    _ppo_replay_step(_pt, _ct, _adv, _olp, 1.0 / len(ppo_batch))
                _set_optimizer_lr(num_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                _invalidate_seqcond_caches(model)
            ppo_batch = []
        else:
            _set_optimizer_lr(num_steps)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            _invalidate_seqcond_caches(model)
        optimizer.zero_grad()

    print(f"\n── {objective_name} complete ({time.time()-t0:.0f}s) ──\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="GRPO fine-tuning for SeqCond on GSM8K")
    p.add_argument("--checkpoint", required=True, help="PyTorch .pt checkpoint")
    p.add_argument(
        "--save", default=None, help="Output .pt path (default: <base>_grpo.pt)"
    )
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--local_file", default=None,
                   help="Chemin vers un JSONL local {query, ground_truth} (remplace GSM8K)")
    p.add_argument("--skip_baseline", action="store_true")

    # Generation
    p.add_argument("--num_completions", type=int, default=6)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--rep_penalty", type=float, default=1.0)
    p.add_argument("--gen_batch_size", type=int, default=4)
    p.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="Batch size for eval generation. All k*N pass@k prompts are "
        "flattened into one queue and dispatched in chunks of this size. "
        "0 = fall back to --gen_batch_size.",
    )

    # Training
    p.add_argument("--num_steps", type=int, default=250)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--train_layers", type=int, default=3)
    p.add_argument("--optimizer", default="adamw", choices=["sgd", "adamw"])
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--max_eval", type=int, default=100)
    p.add_argument("--eval_num_completions", type=int, default=1)
    p.add_argument("--eval_temperature", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument(
        "--lr_decay",
        default="linear",
        choices=["constant", "linear"],
    )
    p.add_argument("--lr_final_ratio", type=float, default=0.0)
    p.add_argument(
        "--grad_accum_steps",
        type=int,
        default=4,
        help="Accumulate gradients over N steps before optimizer update",
    )
    p.add_argument(
        "--save_every",
        "--save-gcp",  # legacy alias
        type=int,
        default=0,
        dest="save_every",
        help="Save a stepped checkpoint every N steps to disk (0=disabled). "
        "Writes <save_path>_step<NNNNN>.pt next to the final save path.",
    )

    # GMPO
    p.add_argument(
        "--use-gmpo",
        action="store_true",
        default=True,
        help="Use GMPO objective instead of the default GRPO objective",
    )
    p.add_argument(
        "--no-gmpo",
        action="store_false",
        dest="use_gmpo",
        help="Disable GMPO and use the original GRPO-style policy objective",
    )

    # Dr. GRPO
    p.add_argument(
        "--use-dr-grpo",
        action="store_true",
        default=True,
        dest="use_dr_grpo",
        help="Remove std from advantage denominator (Dr. GRPO) to fix difficulty bias",
    )
    p.add_argument(
        "--no-dr-grpo",
        action="store_false",
        dest="use_dr_grpo",
        help="Use standard (r - mean) / std advantage normalization",
    )

    # Advantage clipping
    p.add_argument(
        "--clip-adv",
        type=float,
        default=5.0,
        dest="clip_adv",
        help="Clip advantages to [-x, x] after normalization (0=disabled)",
    )

    # PPO multi-epoch
    p.add_argument(
        "--ppo_epochs",
        type=int,
        default=1,
        help="Number of PPO epochs per batch (1=REINFORCE, >1=PPO-clip)",
    )
    p.add_argument(
        "--clip_eps_low",
        type=float,
        default=0.2,
        help="PPO clip lower bound: ratio >= 1-eps_low",
    )
    p.add_argument(
        "--clip_eps_high",
        type=float,
        default=0.28,
        help="PPO clip upper bound: ratio <= 1+eps_high (DAPO asymmetric)",
    )
    p.add_argument(
        "--gmpo_clip_eps",
        type=float,
        default=0.4,
        help="GMPO symmetric clip epsilon on per-token log-ratios (used when --use-gmpo + ppo_epochs>1)",
    )

    # LLDS regularization (Lazy Likelihood-Displacement Stabilization)
    p.add_argument(
        "--llds_lambda",
        type=float,
        default=0.0,
        help="LLDS regularization weight (0=disabled). Prevents likelihood collapse by penalizing "
             "token-level log-prob drops for non-negative advantage completions. Paper: arxiv 2512.04220",
    )
    p.add_argument(
        "--min_active_steps",
        type=int,
        default=1,
        help="Minimum number of steps with real gradient signal in an accumulation window. "
             "If fewer, the window is discarded without applying the optimizer update.",
    )
    p.add_argument(
        "--neg_scale",
        type=float,
        default=0.0,
        help="If >0, downscale negative-advantage gradients so their norm is at most "
             "neg_scale times the positive-gradient norm within each GRPO/PPO group. "
             "No effect when negative gradients are already weaker.",
    )

    # Layer rotation
    p.add_argument(
        "--reshuffle_layers_every",
        type=int,
        default=4,
        help="Reshuffle which layers are trained every N optimizer updates.",
    )

    args = p.parse_args()

    model, config, torch_gen = load_model(args.checkpoint)
    if args.local_file:
        train_examples = load_local_math(args.local_file, seed=42, max_examples=args.max_examples)
        eval_examples  = load_gsm8k(split="test", seed=42, max_examples=args.max_eval)
    else:
        train_examples = load_gsm8k(split="train", seed=42, max_examples=args.max_examples)
        eval_examples  = load_gsm8k(split="test", seed=42, max_examples=args.max_eval)
    print(f"Dataset: {len(train_examples)} train / {len(eval_examples)} eval (test split)")

    if not args.skip_baseline:
        print("\n── Baseline eval ──")
        evaluate(
            torch_gen,
            eval_examples,
            max_examples=args.max_eval,
            num_completions=args.eval_num_completions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.eval_temperature,
            rep_penalty=args.rep_penalty,
            gen_batch_size=args.gen_batch_size,
            eval_batch_size=args.eval_batch_size or None,
        )
        if args.eval_num_completions != 1 or args.eval_temperature != 0.0:
            evaluate(
                torch_gen,
                eval_examples,
                max_examples=args.max_eval,
                num_completions=1,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                rep_penalty=args.rep_penalty,
                gen_batch_size=args.gen_batch_size,
                eval_batch_size=args.eval_batch_size or None,
            )
        # Round-trip debug moved to sanity_check_reload.py

    save_path = args.save or os.path.join(
        "checkpoints",
        os.path.splitext(os.path.basename(args.checkpoint))[0] + "_grpo.pt",
    )

    train_grpo(
        model,
        config,
        train_examples,
        eval_examples=eval_examples,
        torch_gen=torch_gen,
        use_gmpo=args.use_gmpo,
        num_completions=args.num_completions,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        rep_penalty=args.rep_penalty,
        gen_batch_size=args.gen_batch_size,
        eval_batch_size=args.eval_batch_size,
        beta=args.beta,
        lr=args.lr,
        optimizer_name=args.optimizer,
        train_layers=args.train_layers,
        num_steps=args.num_steps,
        eval_every=args.eval_every,
        max_eval=args.max_eval,
        eval_num_completions=args.eval_num_completions,
        eval_temperature=args.eval_temperature,
        save_every=args.save_every,
        save_path=save_path,
        grad_accum_steps=args.grad_accum_steps,
        use_dr_grpo=args.use_dr_grpo,
        clip_adv=args.clip_adv,
        reshuffle_layers_every=args.reshuffle_layers_every,
        lr_decay=args.lr_decay,
        lr_final_ratio=args.lr_final_ratio,
        warmup_steps=args.warmup_steps,
        ppo_epochs=args.ppo_epochs,
        clip_eps_low=args.clip_eps_low,
        clip_eps_high=args.clip_eps_high,
        gmpo_clip_eps=args.gmpo_clip_eps,
        llds_lambda=args.llds_lambda,
        min_active_steps=args.min_active_steps,
        neg_scale=args.neg_scale,
    )

    try:
        save_ckpt(torch_gen, config, save_path, 0)
        print(f"Saved: {save_path}")
    except OSError as e:
        print(f"WARNING: could not save ({e})")


if __name__ == "__main__":
    main()

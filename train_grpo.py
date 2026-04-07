"""
GRPO fine-tuning for SeqCond on GSM8K.

Quick start:
    KERAS_BACKEND=torch python train_grpo.py --checkpoint checkpoints/seqcond_lin5.pt

Colab:
    !pip install keras datasets openai  # once
    %env KERAS_BACKEND=torch
    !python train_grpo.py --checkpoint ... --openai_api_key sk-...

Two functions to modify:
    score_output()  — what counts as a good response  (reward design)
    grpo_step()     — how rewards drive the update     (algorithm)
"""

import argparse
import math
import os
import random
import re
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import ops

from convert_torch_to_keras import (
    build_keras_model,
    convert_weights,
    get_config_value,
    keras_pkl_to_torch_pt,
    load_torch_checkpoint,
    save_keras_checkpoint,
)
from seqcond.dataset import Tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Fast PyTorch generation model (Triton-accelerated)
# ─────────────────────────────────────────────────────────────────────────────


def load_torch_gen_model(config, checkpoint_path):
    """Load a pure PyTorch SeqCondModel for fast Triton-accelerated generation."""
    import torch
    from seqcond.torch.model import SeqCondModel

    torch_model = SeqCondModel(**config).cpu().eval()
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    torch_model.load_state_dict(data["state_dict"], strict=False)
    n = sum(p.numel() for p in torch_model.parameters())
    print(f"Loaded PyTorch gen model ({n:,} params, Triton-accelerated)")
    return torch_model


def sync_keras_to_torch(keras_model, torch_model, config):
    """Copy weights from Keras model to PyTorch model (in-place).

    Since KERAS_BACKEND=torch, Keras weights are torch tensors.
    We reverse the mapping from convert_torch_to_keras.convert_weights.
    """
    import torch

    # Build Keras weight lookup: short path -> tensor
    keras_w = {}
    for w in keras_model.weights:
        parts = w.path.split("/")
        short = "/".join(parts[1:]) if len(parts) > 1 else w.path
        keras_w[short] = w

    # Build block map
    num_layers = get_config_value(config, "num_layers")
    seqcond_ratio = get_config_value(config, "seqcond_ratio", 3)
    transformer_idx = seqcond_idx = 0
    block_map = []
    for i in range(num_layers):
        if (i + 1) % (seqcond_ratio + 1) == 0:
            block_map.append((i, "transformer", f"transformer_block_{transformer_idx}"))
            transformer_idx += 1
        else:
            block_map.append((i, "seqcond", f"seqcond_block_{seqcond_idx}"))
            seqcond_idx += 1

    state_dict = {}

    def get(short):
        if short not in keras_w:
            return None
        return keras_w[short].value  # returns underlying torch tensor

    # Embedding
    state_dict["embedding.weight"] = get("token_embedding/embeddings")

    # Final norm
    v = get("final_norm/scale")
    if v is not None:
        state_dict["final_norm.scale"] = v

    # Blocks
    for torch_i, btype, kname in block_map:
        tp = f"blocks.{torch_i}."
        if btype == "transformer":
            state_dict[tp + "norm1.scale"] = get(f"{kname}/norm1/scale")
            state_dict[tp + "norm2.scale"] = get(f"{kname}/norm2/scale")
            state_dict[tp + "attn.q_proj.weight"] = get(f"{kname}/attn/q_proj/kernel").T
            state_dict[tp + "attn.k_proj.weight"] = get(f"{kname}/attn/k_proj/kernel").T
            state_dict[tp + "attn.v_proj.weight"] = get(f"{kname}/attn/v_proj/kernel").T
            state_dict[tp + "attn.out_proj.weight"] = get(
                f"{kname}/attn/out_proj/kernel"
            ).T
            state_dict[tp + "ff_in.weight"] = get(f"{kname}/ff_in/kernel").T
            state_dict[tp + "ff_in.bias"] = get(f"{kname}/ff_in/bias")
            state_dict[tp + "ff_out.weight"] = get(f"{kname}/ff_out/kernel").T
            state_dict[tp + "ff_out.bias"] = get(f"{kname}/ff_out/bias")
        else:  # seqcond
            state_dict[tp + "norm.scale"] = get(f"{kname}/pre_norm/scale")
            state_dict[tp + "attn.in_proj.weight"] = get(
                f"{kname}/attn/in_proj/kernel"
            ).T
            conv = get(f"{kname}/attn/conv/kernel")
            state_dict[tp + "attn.conv_weight"] = conv.permute(2, 1, 0).contiguous()
            state_dict[tp + "attn.gate_proj.weight"] = get(
                f"{kname}/attn/gate_proj/kernel"
            ).T
            state_dict[tp + "attn.out_proj.weight"] = get(
                f"{kname}/attn/out_proj/kernel"
            ).T
            for raw_key in [
                "theta_d_raw",
                "theta_raw",
                "w_int_raw",
                "decay_slopes",
                "anchor_slopes",
                "score_scale",
                "score_bias",
                "phase_scale",
            ]:
                val = get(f"{kname}/attn/{raw_key}")
                if val is not None:
                    state_dict[tp + f"attn.{raw_key}"] = val
            state_dict[tp + "attn.gated_norm.weight"] = get(
                f"{kname}/attn/gated_norm_weight"
            )
            state_dict[tp + "attn.W_readout"] = get(f"{kname}/attn/W_readout")

    # Filter None and load
    state_dict = {k: v for k, v in state_dict.items() if v is not None}
    torch_model.load_state_dict(state_dict, strict=False)


def generate_completions_torch(
    torch_model,
    tokenizer,
    prompt: str,
    num_completions: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    rep_penalty: float = 1.0,
    gen_batch_size: int = 4,
    return_tokens: bool = False,
    keep_on_cuda: bool = False,
):
    """Fast generation using PyTorch model with Triton kernels.

    Moves model to CUDA for generation, then back to CPU to free VRAM.
    If keep_on_cuda=True, assumes model is already on CUDA and does not move it.
    """
    import torch

    eos_id = tokenizer.encode("<|im_end|>")[0]
    prompt_toks = tokenizer([prompt])[0]
    all_texts, all_ids = [], []

    if not keep_on_cuda:
        torch_model.cuda()
    input_ids = torch.tensor([prompt_toks], device="cuda")

    with torch.no_grad():
        for start in range(0, num_completions, gen_batch_size):
            B = min(gen_batch_size, num_completions - start)

            # Prefill once
            logits, states = torch_model.prefill(input_ids)
            logits = logits.squeeze(1)  # (1, vocab)

            # Tile for batch
            if B > 1:
                logits = logits.repeat(B, 1)
                states = [
                    tuple(s.repeat(B, *([1] * (s.ndim - 1))) for s in state)
                    for state in states
                ]

            generated = [[] for _ in range(B)]
            finished = [False] * B
            active_map = list(range(B))
            token_buf = torch.zeros((B, 1), dtype=torch.long, device="cuda")
            current_seq_len = len(prompt_toks)

            for _ in range(max_new_tokens):
                B_cur = len(active_map)
                logits_np = logits.cpu().float().numpy()
                if rep_penalty != 1.0:
                    for ci in range(B_cur):
                        oi = active_map[ci]
                        for tid in set(generated[oi]):
                            logits_np[ci, tid] = (
                                logits_np[ci, tid] / rep_penalty
                                if logits_np[ci, tid] > 0
                                else logits_np[ci, tid] * rep_penalty
                            )
                toks = _sample_batch(logits_np, temperature, top_k, top_p)
                newly_done = set()
                for ci in range(B_cur):
                    oi = active_map[ci]
                    tok = int(toks[ci])
                    generated[oi].append(tok)
                    if tok == eos_id:
                        finished[oi] = True
                        newly_done.add(ci)
                    else:
                        token_buf[ci, 0] = tok
                if all(finished):
                    break
                if newly_done:
                    keep = [ci for ci in range(B_cur) if ci not in newly_done]
                    if not keep:
                        break
                    keep_idx = torch.tensor(keep, device="cuda")
                    token_buf = token_buf[keep_idx].contiguous()
                    states = [
                        tuple(s[keep_idx].contiguous() for s in state)
                        for state in states
                    ]
                    active_map = [active_map[ci] for ci in keep]
                current_seq_len += 1
                logits, states = torch_model.step(
                    token_buf, states, seq_len=current_seq_len, use_triton=True
                )


            for i in range(B):
                ids = generated[i]
                if ids and ids[-1] == eos_id:
                    ids = ids[:-1]
                all_ids.append(list(ids))
                try:
                    all_texts.append(tokenizer.decode(ids))
                except Exception:
                    all_texts.append("")

    # Move gen model back to CPU and free VRAM for training
    if not keep_on_cuda:
        torch_model.cpu()
        torch.cuda.empty_cache()

    return (all_texts, all_ids) if return_tokens else all_texts


# ─────────────────────────────────────────────────────────────────────────────
# Backend helpers
# ─────────────────────────────────────────────────────────────────────────────


def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.array(x)


@contextmanager
def inference_mode():
    if keras.backend.backend() == "torch":
        import torch

        with torch.no_grad():
            yield
    else:
        yield


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


def check_answer(text: str, ground_truth: str) -> bool:
    """Return True if text contains the correct numerical answer."""
    m = re.search(r"####\s*(.+)", text)
    if m:
        return m.group(1).strip().replace(",", "") == ground_truth
    parts = text.split("<|think_end|>")
    after = parts[-1].strip() if len(parts) > 1 else text.strip()
    for num in re.findall(r"-?[\d,]+\.?\d*", after):
        if num.replace(",", "").rstrip(".") == ground_truth:
            return True
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

    # Prefer numbers appearing after <|think_end|>; fall back to full text
    parts = text.split("<|think_end|>")
    search_zone = parts[-1] if len(parts) > 1 else text
    candidates = re.findall(r"-?[\d,]+\.?\d*", search_zone)
    if not candidates:
        candidates = re.findall(r"-?[\d,]+\.?\d*", text)

    best = 0.0
    for raw in candidates:
        try:
            val = float(raw.replace(",", "").rstrip("."))
        except ValueError:
            continue
        ratio = val / gt
        if 0.95 <= ratio <= 1.05:
            best = max(best, 0.5)
        elif 0.80 <= ratio <= 1.20:
            best = max(best, 0.2)
    return best


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


def _tile_states(states, n):
    return [
        tuple(ops.tile(s, (n,) + (1,) * (s.ndim - 1)) for s in state)
        for state in states
    ]


def _sample_batch(logits_np, temperature=0.7, top_k=50, top_p=0.95):
    """Sample one token per row from (B, vocab) logits."""
    B = logits_np.shape[0]
    tokens = np.empty(B, dtype=np.int64)
    for i in range(B):
        row = logits_np[i].astype(np.float64)
        if temperature <= 0.0:
            tokens[i] = np.argmax(row)
            continue
        row = row / temperature
        topk_idx = (
            np.argpartition(row, -top_k)[-top_k:]
            if 0 < top_k < len(row)
            else np.arange(len(row))
        )
        vals = row[topk_idx]
        vals -= np.max(vals)
        probs = np.exp(vals)
        probs /= probs.sum()
        if top_p < 1.0:
            order = np.argsort(probs)[::-1]
            cut = np.searchsorted(np.cumsum(probs[order]), top_p) + 1
            nucleus = order[:cut]
            p = probs[nucleus]
            p /= p.sum()
            tokens[i] = topk_idx[np.random.choice(nucleus, p=p)]
        else:
            tokens[i] = topk_idx[np.random.choice(len(probs), p=probs)]
    return tokens


def generate_completions(
    model,
    tokenizer,
    prompt: str,
    num_completions: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    rep_penalty: float = 1.0,
    gen_batch_size: int = 4,
    return_tokens: bool = False,
):
    """Generate num_completions responses for prompt. Prefills once, tiles states."""
    eos_id = tokenizer.encode("<|im_end|>")[0]
    prompt_toks = tokenizer([prompt])[0]
    all_texts, all_ids = [], []

    with inference_mode():
        for start in range(0, num_completions, gen_batch_size):
            B = min(gen_batch_size, num_completions - start)
            states = model.init_state(batch_size=1)
            for t in prompt_toks:
                logits, states = model.step(np.array([[t]], dtype=np.int32), states)
            if B > 1:
                states = _tile_states(states, B)
            logits_np = np.tile(to_numpy(logits), (B, 1))

            generated = [[] for _ in range(B)]
            finished = [False] * B
            buf = np.zeros((B, 1), dtype=np.int32)

            for _ in range(max_new_tokens):
                if rep_penalty != 1.0:
                    for i in range(B):
                        for tid in set(generated[i]):
                            logits_np[i, tid] = (
                                logits_np[i, tid] / rep_penalty
                                if logits_np[i, tid] > 0
                                else logits_np[i, tid] * rep_penalty
                            )
                toks = _sample_batch(logits_np, temperature, top_k, top_p)
                for i in range(B):
                    if not finished[i]:
                        generated[i].append(int(toks[i]))
                        finished[i] = toks[i] == eos_id
                        buf[i, 0] = toks[i]
                    else:
                        buf[i, 0] = eos_id
                if all(finished):
                    break
                logits, states = model.step(buf, states)
                logits_np = to_numpy(logits)

            for i in range(B):
                ids = generated[i]
                if ids and ids[-1] == eos_id:
                    ids = ids[:-1]
                all_ids.append(list(ids))
                try:
                    all_texts.append(tokenizer.decode(ids))
                except Exception:
                    all_texts.append("")

    return (all_texts, all_ids) if return_tokens else all_texts


# ─────────────────────────────────────────────────────────────────────────────
# GRPO internals
# ─────────────────────────────────────────────────────────────────────────────


def _seq_log_prob(model, input_ids, prompt_len):
    """Sum of per-token log probs for the completion portion of input_ids."""
    return ops.sum(_seq_token_log_probs(model, input_ids, prompt_len))


def _seq_token_log_probs(model, input_ids, prompt_len):
    """Per-token log probs for the completion portion of input_ids."""
    logits = model(input_ids)
    log_probs = ops.log_softmax(logits, axis=-1)
    shift = log_probs[0, prompt_len - 1 : -1, :]
    targets = ops.expand_dims(ops.cast(input_ids[0, prompt_len:], "int32"), -1)
    return ops.squeeze(ops.take_along_axis(shift, targets, axis=-1), axis=-1)


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


def _compute_old_log_probs(model, prompt_tokens, completions_tokens):
    """Compute detached per-token log probs for all completions (π_old)."""
    prompt_len = len(prompt_tokens)
    old_lps = []
    with inference_mode():
        for comp in completions_tokens:
            if len(comp) == 0:
                old_lps.append(None)
                continue
            ids = np.array([prompt_tokens + comp], dtype=np.int32)
            token_lps = _seq_token_log_probs(model, ids, prompt_len)
            old_lps.append(to_numpy(token_lps).copy())
    return old_lps


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
) -> float:
    """PPO-clip GRPO: clipped surrogate objective with importance sampling.

    Uses stored π_old to compute the ratio π/π_old per token, with asymmetric
    clipping [1-ε_low, 1+ε_high] (DAPO-style).  Supports multiple epochs
    over the same batch of completions.

    LLDS: For completions with advantage >= 0, if total log-likelihood decreased
    vs π_old, penalize tokens whose individual log-prob dropped.
    """
    prompt_len = len(prompt_tokens)
    total_tokens = sum(len(c) for c in completions_tokens if len(c) > 0)
    if total_tokens == 0:
        return 0.0

    # Reference log probs for KL (optional, frozen model)
    ref_avg_lps = []
    if beta > 0:
        ref = ref_model if ref_model is not None else model
        with inference_mode():
            for comp in completions_tokens:
                if len(comp) == 0:
                    ref_avg_lps.append(0.0)
                    continue
                ids = np.array([prompt_tokens + comp], dtype=np.int32)
                ref_avg_lps.append(float(_seq_log_prob(ref, ids, prompt_len)) / len(comp))

    total_loss = 0.0
    for i, comp in enumerate(completions_tokens):
        if len(comp) == 0 or advantages[i] == 0 or old_token_log_probs[i] is None:
            continue
        ids = np.array([prompt_tokens + comp], dtype=np.int32)
        # Current per-token log probs (with gradient)
        cur_token_lps = _seq_token_log_probs(model, ids, prompt_len)
        # Old per-token log probs (detached numpy)
        old_lps_t = ops.convert_to_tensor(old_token_log_probs[i])

        # Importance sampling ratio per token
        log_ratio = cur_token_lps - old_lps_t
        ratio = ops.exp(log_ratio)

        # Clipped ratio (asymmetric: DAPO)
        clipped_ratio = ops.clip(ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_high)

        # PPO surrogate: min(ratio*A, clip(ratio)*A)
        adv_i = float(advantages[i])
        surr1 = ratio * adv_i
        surr2 = clipped_ratio * adv_i
        # min handles both signs of advantage correctly
        ppo_obj = ops.minimum(surr1, surr2)

        # Token-level mean, then weight by completion length
        token_weight = len(comp) / total_tokens
        loss_i = -ops.mean(ppo_obj) * token_weight

        # KL penalty (optional)
        if beta > 0:
            avg_lp = ops.sum(cur_token_lps) / len(comp)
            kl = avg_lp - ref_avg_lps[i]
            loss_i = loss_i + beta * kl * token_weight

        (loss_i * grad_scale).backward()
        total_loss += float(to_numpy(loss_i))

    # LLDS regularization for PPO path
    if llds_lambda > 0:
        for i, comp in enumerate(completions_tokens):
            if len(comp) == 0 or advantages[i] < 0 or old_token_log_probs[i] is None:
                continue
            ids = np.array([prompt_tokens + comp], dtype=np.int32)
            cur_token_lps = _seq_token_log_probs(model, ids, prompt_len)
            old_lps_t = ops.convert_to_tensor(old_token_log_probs[i])
            # Response-level gate: only activate if total likelihood decreased
            cur_total = ops.sum(cur_token_lps)
            old_total = float(np.sum(old_token_log_probs[i]))
            if float(to_numpy(cur_total)) >= old_total:
                continue
            # Token-level: penalize only tokens whose log-prob dropped
            drop = old_lps_t - cur_token_lps
            drop_masked = ops.maximum(drop, 0.0)
            token_weight = len(comp) / total_tokens
            llds_loss = llds_lambda * ops.mean(drop_masked) * token_weight
            (llds_loss * grad_scale).backward()
            total_loss += float(to_numpy(llds_loss))

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
) -> float:
    """Compute GRPO loss and accumulate gradients for one group of completions.

    Token-level normalization: each completion is weighted by its token count so
    each token equal weight regardless of sequence length.

    For each completion i with non-zero advantage:
        loss_i = -advantage_i * sum_log_prob_i  +  β * KL(π || π_ref) * len(comp_i)
    Total loss is divided by total_tokens (sum of all completion lengths).

    LLDS (Lazy Likelihood-Displacement Stabilization):
        For completions with advantage >= 0 whose total log-likelihood decreased
        vs the old policy, penalize individual tokens whose log-prob dropped.
    """
    prompt_len = len(prompt_tokens)
    total_tokens = sum(len(c) for c in completions_tokens if len(c) > 0)
    if total_tokens == 0:
        return 0.0

    # Reference log probs (frozen initial model, or current snapshot if no ref)
    ref = ref_model if ref_model is not None else model
    ref_avg_lps = []
    with inference_mode():
        for comp in completions_tokens:
            if len(comp) == 0:
                ref_avg_lps.append(0.0)
                continue
            ids = np.array([prompt_tokens + comp], dtype=np.int32)
            ref_avg_lps.append(float(_seq_log_prob(ref, ids, prompt_len)) / len(comp))

    # Old token-level log probs for LLDS (detached snapshot of current policy)
    old_token_lps_list = []
    if llds_lambda > 0:
        with inference_mode():
            for comp in completions_tokens:
                if len(comp) == 0:
                    old_token_lps_list.append(None)
                    continue
                ids = np.array([prompt_tokens + comp], dtype=np.int32)
                old_token_lps_list.append(to_numpy(_seq_token_log_probs(model, ids, prompt_len)).copy())

    # Policy gradient with token-level normalization
    total_loss = 0.0
    for i, comp in enumerate(completions_tokens):
        if len(comp) == 0 or advantages[i] == 0:
            continue
        ids = np.array([prompt_tokens + comp], dtype=np.int32)
        policy_lp = _seq_log_prob(model, ids, prompt_len)  # sum over tokens
        avg_lp = policy_lp / len(comp)
        kl = avg_lp - ref_avg_lps[i]
        # Token-level normalization: weight each completion by its token count
        token_weight = len(comp) / total_tokens
        loss_i = (-advantages[i] * avg_lp + beta * kl) * token_weight
        (loss_i * grad_scale).backward()
        total_loss += float(to_numpy(loss_i))

    # LLDS regularization: prevent likelihood collapse for non-negative advantage completions
    if llds_lambda > 0:
        for i, comp in enumerate(completions_tokens):
            if len(comp) == 0 or advantages[i] < 0 or old_token_lps_list[i] is None:
                continue
            ids = np.array([prompt_tokens + comp], dtype=np.int32)
            cur_token_lps = _seq_token_log_probs(model, ids, prompt_len)
            old_lps_t = ops.convert_to_tensor(old_token_lps_list[i])
            # Response-level gate: only activate if total likelihood decreased
            cur_total = ops.sum(cur_token_lps)
            old_total = float(np.sum(old_token_lps_list[i]))
            if float(to_numpy(cur_total)) >= old_total:
                continue
            # Token-level: penalize only tokens whose log-prob dropped
            drop = old_lps_t - cur_token_lps  # positive where likelihood decreased
            drop_masked = ops.maximum(drop, 0.0)  # only penalize decreases
            token_weight = len(comp) / total_tokens
            llds_loss = llds_lambda * ops.mean(drop_masked) * token_weight
            (llds_loss * grad_scale).backward()
            total_loss += float(to_numpy(llds_loss))

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
) -> float:
    """Compute GMPO loss (true geometric-mean policy optimization).

    Each token's log importance-ratio is clipped independently to [-clip_eps, +clip_eps]
    BEFORE taking the mean (= geometric mean in linear space). This prevents outlier
    tokens from dominating the sequence ratio, which was the bug in the previous
    sequence-level clipping approach. Token-level normalization is used (same as
    grpo_step) so every token contributes equally regardless of sequence length.
    """
    prompt_len = len(prompt_tokens)
    total_tokens = sum(len(c) for c in completions_tokens if len(c) > 0)
    if total_tokens == 0:
        return 0.0

    ref = ref_model if ref_model is not None else model
    old_token_lps = []
    ref_avg_lps = []
    with inference_mode():
        for comp in completions_tokens:
            if len(comp) == 0:
                old_token_lps.append(None)
                ref_avg_lps.append(0.0)
                continue
            ids = np.array([prompt_tokens + comp], dtype=np.int32)
            old_token_lps.append(_seq_token_log_probs(model, ids, prompt_len))
            ref_avg_lps.append(float(_seq_log_prob(ref, ids, prompt_len)) / len(comp))

    total_loss = 0.0
    for i, comp in enumerate(completions_tokens):
        if len(comp) == 0 or advantages[i] == 0:
            continue

        ids = np.array([prompt_tokens + comp], dtype=np.int32)
        new_token_lps = _seq_token_log_probs(model, ids, prompt_len)
        old_lps = old_token_lps[i]

        adv = float(advantages[i])
        sign_a = 1.0 if adv > 0 else -1.0
        # GMPO (paper): per-token r_t^sgn(A), pessimistic clip (cap above only)
        # In log-space: s·ℓ_t where s=sgn(A), ℓ_t = log(π_new/π_old)
        # min(r^s, clip(r^s, e^-ε, e^+ε)) in log = min(s·ℓ_t, +ε)  (no floor)
        token_log_ratios = new_token_lps - old_lps          # ℓ_t
        signed_log_ratios = sign_a * token_log_ratios        # s·ℓ_t
        pessimistic = ops.minimum(signed_log_ratios, clip_eps)  # min(s·ℓ_t, ε)
        mean_pessimistic = ops.mean(pessimistic)
        seq_ratio = ops.exp(mean_pessimistic)                # geometric mean
        avg_policy_lp = ops.mean(new_token_lps)
        kl = avg_policy_lp - ref_avg_lps[i]

        # Token-level normalization: weight by this completion's share of total tokens
        token_weight = len(comp) / total_tokens
        loss_i = (-adv * seq_ratio + beta * kl) * token_weight
        (loss_i * grad_scale).backward()
        total_loss += float(to_numpy(loss_i))

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
) -> float:
    """Multi-epoch GMPO: geometric-mean policy optimization with stored π_old.

    Like gmpo_step but uses pre-computed old_token_log_probs instead of
    computing π_old inline.  Each token's log importance-ratio is clipped
    symmetrically to [-clip_eps, +clip_eps] BEFORE taking the mean
    (= geometric mean in linear space).
    """
    prompt_len = len(prompt_tokens)
    total_tokens = sum(len(c) for c in completions_tokens if len(c) > 0)
    if total_tokens == 0:
        return 0.0

    ref_avg_lps = []
    if beta > 0:
        ref = ref_model if ref_model is not None else model
        with inference_mode():
            for comp in completions_tokens:
                if len(comp) == 0:
                    ref_avg_lps.append(0.0)
                    continue
                ids = np.array([prompt_tokens + comp], dtype=np.int32)
                ref_avg_lps.append(float(_seq_log_prob(ref, ids, prompt_len)) / len(comp))

    total_loss = 0.0
    for i, comp in enumerate(completions_tokens):
        if len(comp) == 0 or advantages[i] == 0 or old_token_log_probs[i] is None:
            continue
        ids = np.array([prompt_tokens + comp], dtype=np.int32)
        new_token_lps = _seq_token_log_probs(model, ids, prompt_len)
        old_lps_t = ops.convert_to_tensor(old_token_log_probs[i])

        adv = float(advantages[i])
        sign_a = 1.0 if adv > 0 else -1.0
        # GMPO (paper): s·ℓ_t, pessimistic clip = min(s·ℓ_t, +ε)
        token_log_ratios = new_token_lps - old_lps_t
        signed_log_ratios = sign_a * token_log_ratios
        pessimistic = ops.minimum(signed_log_ratios, clip_eps)
        mean_pessimistic = ops.mean(pessimistic)
        seq_ratio = ops.exp(mean_pessimistic)

        token_weight = len(comp) / total_tokens
        loss_i = -adv * seq_ratio * token_weight

        if beta > 0:
            avg_policy_lp = ops.mean(new_token_lps)
            kl = avg_policy_lp - ref_avg_lps[i]
            loss_i = loss_i + beta * kl * token_weight

        (loss_i * grad_scale).backward()
        total_loss += float(to_numpy(loss_i))

    return total_loss


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────


def load_model(checkpoint_path: str):
    """Load Keras SeqCond model + PyTorch gen model from a .pt checkpoint."""
    config, state_dict = load_torch_checkpoint(checkpoint_path)
    model = build_keras_model(config)
    convert_weights(config, state_dict, model)
    n_params = model.count_params()
    print(
        f"Loaded {checkpoint_path}  ({n_params:,} params, backend={keras.backend.backend()})"
    )
    torch_gen = load_torch_gen_model(config, checkpoint_path)
    return model, config, torch_gen


# Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def _generate_eval_batch(
    torch_model,
    tokenizer,
    prompts,
    max_new_tokens=512,
    temperature=0.0,
    rep_penalty: float = 1.0,
):
    """Generate one completion per prompt, batched across different prompts.

    Assumes model is already on CUDA. Returns list of completion strings.
    """
    import torch

    B = len(prompts)
    if B == 0:
        return []
    eos_id = tokenizer.encode("<|im_end|>")[0]

    # Tokenize all prompts
    all_toks = [tokenizer([p])[0] for p in prompts]
    prompt_lens = [len(t) for t in all_toks]

    # Prefill each prompt individually (different lengths, no padding noise)
    all_logits, all_states = [], []
    with torch.no_grad():
        for toks in all_toks:
            input_ids = torch.tensor([toks], device="cuda")
            logits_i, states_i = torch_model.prefill(input_ids)
            all_logits.append(logits_i.squeeze(1))
            all_states.append(states_i)

    # Stack into batched tensors
    logits = torch.cat(all_logits, dim=0)
    num_blocks = len(all_states[0])
    states = []
    for block_idx in range(num_blocks):
        block_state = tuple(
            torch.cat([s[block_idx][t] for s in all_states], dim=0)
            for t in range(len(all_states[0][block_idx]))
        )
        states.append(block_state)

    # Decode with batch compaction
    generated = [[] for _ in range(B)]
    finished = [False] * B
    active_map = list(range(B))
    token_buf = torch.zeros((B, 1), dtype=torch.long, device="cuda")
    current_seq_len = max(prompt_lens)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            B_cur = len(active_map)
            logits_np = logits.cpu().float().numpy()

            if rep_penalty != 1.0:
                for ci in range(B_cur):
                    oi = active_map[ci]
                    for tid in set(generated[oi]):
                        logits_np[ci, tid] = (
                            logits_np[ci, tid] / rep_penalty
                            if logits_np[ci, tid] > 0
                            else logits_np[ci, tid] * rep_penalty
                        )

            toks = _sample_batch(logits_np, temperature, 50, 0.95)
            newly_done = set()
            for ci in range(B_cur):
                oi = active_map[ci]
                tok = int(toks[ci])
                generated[oi].append(tok)
                if tok == eos_id:
                    finished[oi] = True
                    newly_done.add(ci)
                else:
                    token_buf[ci, 0] = tok
            if all(finished):
                break
            if newly_done:
                keep = [ci for ci in range(B_cur) if ci not in newly_done]
                if not keep:
                    break
                keep_idx = torch.tensor(keep, device="cuda")
                token_buf = token_buf[keep_idx].contiguous()
                states = [
                    tuple(s[keep_idx].contiguous() for s in state)
                    for state in states
                ]
                active_map = [active_map[ci] for ci in keep]
            current_seq_len += 1
            logits, states = torch_model.step(
                token_buf, states, seq_len=current_seq_len, use_triton=True
            )

    results = []
    for i in range(B):
        ids = generated[i]
        if ids and ids[-1] == eos_id:
            ids = ids[:-1]
        try:
            results.append(tokenizer.decode(ids))
        except Exception:
            results.append("")
    return results


def evaluate(
    torch_gen,
    examples,
    max_examples=100,
    num_completions=1,
    max_new_tokens=512,
    temperature=0.0,
    rep_penalty: float = 1.0,
    gen_batch_size=4,
    step=None,
    log_path=None,
):
    """Evaluate pass@k accuracy on the first max_examples problems."""
    import torch

    tokenizer = Tokenizer()
    examples = examples[:max_examples]
    t0 = time.time()

    # ── Phase 1: Generate all completions (fast, batched, GPU) ──
    torch_gen.cuda()
    # Fast batched path: run num_completions passes over all problems
    all_comps = [[] for _ in range(len(examples))]
    for _pass in range(num_completions):
        for batch_start in range(0, len(examples), gen_batch_size):
            prompts = [
                ex["prompt"]
                for ex in examples[batch_start : batch_start + gen_batch_size]
            ]
            batch_comps = _generate_eval_batch(
                torch_gen,
                tokenizer,
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                rep_penalty=rep_penalty,
            )
            for j, c in enumerate(batch_comps):
                all_comps[batch_start + j].append(c)

    torch.cuda.empty_cache()
    gen_time = time.time() - t0
    total_gens = len(examples) * num_completions
    print(f"  Generated {total_gens} eval completions ({len(examples)} problems × {num_completions}) in {gen_time:.1f}s")

    # ── Phase 2: Check answers ──
    correct = 0
    for i, (ex, comps) in enumerate(zip(examples, all_comps)):
        ok = any(check_answer(c, ex["ground_truth"]) for c in comps)
        correct += int(ok)

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


def _save_and_upload_gcp(model, config, save_path, step, torch_gen=None):
    import subprocess, torch as _torch

    gcs_bucket = "gs://telekinesis-43/checkpoints"

    # Save directly from torch_gen state_dict to avoid lossy Keras→pkl→pt round-trip
    # (SeqCond recurrence is chaotic: even 1e-4 weight diffs change generation)
    # IMPORTANT: filter out internal cached buffers (_conv_kernel_t, _theta_cached, etc.)
    # These depend on seq_len of last forward call and corrupt generation if reloaded.
    if torch_gen is not None and save_path.endswith(".pt"):
        _CACHE_PREFIXES = ('_conv_kernel_t', '_decay_slopes_cached', '_phase_scale_b',
                           '_score_bias_b', '_score_scale_b', '_theta_cached', '_w_int_cached')
        sd = {k: v.cpu() for k, v in torch_gen.state_dict().items()
              if not any(k.endswith(sfx) for sfx in _CACHE_PREFIXES)}
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        _torch.save({"state_dict": sd, "config": config}, save_path)
        print(f"PyTorch checkpoint saved (direct): {save_path} ({len(sd)} tensors)")
    else:
        pkl_path = save_path[:-3] + ".pkl" if save_path.endswith(".pt") else save_path
        save_keras_checkpoint(model, config, pkl_path)
        if save_path.endswith(".pt"):
            keras_pkl_to_torch_pt(pkl_path, save_path)

    try:
        filename = os.path.basename(save_path)
        base, ext = os.path.splitext(filename)
        gcs_filename = f"{base}_step{step}{ext}"
        gcs_path = f"{gcs_bucket}/{gcs_filename}"

        print(f"  Uploading to {gcs_path}...", end=" ", flush=True)
        for cmd in [
            ["gcloud", "storage", "cp", save_path, gcs_path],
            ["gsutil", "cp", save_path, gcs_path],
        ]:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                print(f"✓ ({cmd[0]})")
                break
            err = result.stderr.strip() or result.stdout.strip() or "unknown error"
            print(f"\n    {cmd[0]} failed: {err[:500]}")
        else:
            print(f"  ✗ upload failed for {gcs_path}")
    except Exception as e:
        print(f"✗ ({str(e)[:500]})")


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
    save_gcp_every: int = 0,
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
):
    import torch

    # Match pre-training dtype: bf16 compute, fp32 weights (same as train_jax.py)
    keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"Mixed precision: {keras.mixed_precision.global_policy().name}")

    tokenizer = Tokenizer()
    np.random.seed(seed)
    random.seed(seed)

    n_blocks = len(model.blocks_list)
    train_layers = min(train_layers, n_blocks)

    all_params = list(model.parameters())
    optimizer = (
        torch.optim.AdamW(all_params, lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
        if optimizer_name == "adamw"
        else torch.optim.SGD(all_params, lr=lr, momentum=0.0, weight_decay=weight_decay)
    )

    use_fast_gen = torch_gen is not None
    max_total_tokens = int(config.get("maxlen") or 0)

    # Frozen reference model for KL penalty (anchors to initial policy)
    ref_model = build_keras_model(config)
    ref_model(np.zeros((1, 1), dtype=np.int32))  # ensure built
    for ref_w, src_w in zip(ref_model.weights, model.weights):
        ref_w.assign(src_w)
    for p in ref_model.parameters():
        p.requires_grad = False

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
            )
        else:
            return grpo_step_ppo(
                model, _pt, _ct, _adv, _olp,
                clip_eps_low=clip_eps_low,
                clip_eps_high=clip_eps_high,
                beta=beta, ref_model=ref_model,
                grad_scale=_grad_scale,
                llds_lambda=llds_lambda,
            )

    def _set_active_trainable_blocks(block_indices):
        for p in model.parameters():
            p.requires_grad = False
        for idx in block_indices:
            for p in model.blocks_list[idx].parameters():
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

    if use_fast_gen:
        sync_keras_to_torch(model, torch_gen, config)
        torch_gen.cuda()

    for step in range(1, num_steps + 1):
        ex = random.choice(examples)
        prompt_tokens = tokenizer([ex["prompt"]])[0]

        # Generate (fast path with Triton, or fallback to Keras)
        if use_fast_gen:
            texts, ids = generate_completions_torch(
                torch_gen,
                tokenizer,
                ex["prompt"],
                num_completions=num_completions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                rep_penalty=rep_penalty,
                gen_batch_size=gen_batch_size,
                return_tokens=True,
                keep_on_cuda=True,
            )
        else:
            texts, ids = generate_completions(
                model,
                tokenizer,
                ex["prompt"],
                num_completions=num_completions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                rep_penalty=rep_penalty,
                gen_batch_size=gen_batch_size,
                return_tokens=True,
            )

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
                    old_lps = _compute_old_log_probs(model, prompt_tokens, ids_f)
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
                        print(
                            f"  [optimizer.step | train_step={step} | "
                            f"ppo_epoch={_epoch+1}/{ppo_epochs} | "
                            f"grad_norm={pre_clip_norm:.4f} | "
                            f"n_grads={n_grads}/{n_trainable} | "
                            f"active={active_grad_steps}/{grad_accum_steps}]"
                        )
                    ppo_batch = []
                    if use_fast_gen:
                        sync_keras_to_torch(model, torch_gen, config)
                        print(f"  [sync weights → torch gen model at step {step}]")
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
                    print(
                        f"  [optimizer.step | train_step={step} | "
                        f"grad_norm={pre_clip_norm:.4f} | "
                        f"n_grads={n_grads}/{n_trainable} | "
                        f"active={active_grad_steps}/{grad_accum_steps}]"
                    )
                    if use_fast_gen:
                        sync_keras_to_torch(model, torch_gen, config)
                        print(f"  [sync weights → torch gen model at step {step}]")
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
                    print(f"  [optimizer.step | train_step={step} | ppo_epoch={_epoch+1}/{ppo_epochs} | flush=eval]")
                ppo_batch = []
                if use_fast_gen:
                    sync_keras_to_torch(model, torch_gen, config)
                    print(f"  [sync weights → torch gen model at step {step} | flush=eval]")
            else:
                _set_optimizer_lr(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                print(f"  [optimizer.step | train_step={step} | flush=eval]")
                if use_fast_gen:
                    sync_keras_to_torch(model, torch_gen, config)
                    print(f"  [sync weights → torch gen model at step {step} | flush=eval]")
            optimizer.zero_grad()
            pending_grad_steps = 0
            active_grad_steps = 0
            active_block_indices = sorted(random.sample(range(n_blocks), train_layers))
            _set_active_trainable_blocks(active_block_indices)

        # Periodic eval
        if eval_every > 0 and step % eval_every == 0:
            log_path = save_path.replace(".pt", "_eval.log") if save_path else None
            evaluate(
                torch_gen if use_fast_gen else model,
                eval_examples if eval_examples is not None else examples,
                max_examples=max_eval,
                num_completions=eval_num_completions,
                max_new_tokens=max_new_tokens,
                temperature=eval_temperature,
                rep_penalty=rep_penalty,
                gen_batch_size=gen_batch_size,
                step=step,
                log_path=log_path,
            )

            # Round-trip debug moved to sanity_check_reload.py

        # Flush pending gradients before backup
        if (
            save_gcp_every > 0
            and step % save_gcp_every == 0
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
                    print(f"  [optimizer.step | train_step={step} | ppo_epoch={_epoch+1}/{ppo_epochs} | flush=save]")
                ppo_batch = []
                if use_fast_gen:
                    sync_keras_to_torch(model, torch_gen, config)
                    print(f"  [sync weights → torch gen model at step {step} | flush=save]")
            else:
                _set_optimizer_lr(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                print(f"  [optimizer.step | train_step={step} | flush=save]")
                if use_fast_gen:
                    sync_keras_to_torch(model, torch_gen, config)
                    print(f"  [sync weights → torch gen model at step {step} | flush=save]")
            optimizer.zero_grad()
            pending_grad_steps = 0
            active_grad_steps = 0
            active_block_indices = sorted(random.sample(range(n_blocks), train_layers))
            _set_active_trainable_blocks(active_block_indices)

        # Periodic GCP backup
        if save_gcp_every > 0 and step % save_gcp_every == 0 and save_path:
            _save_and_upload_gcp(model, config, save_path, step, torch_gen=torch_gen if use_fast_gen else None)

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
            ppo_batch = []
        else:
            _set_optimizer_lr(num_steps)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
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
    p.add_argument("--skip_baseline", action="store_true")

    # Generation
    p.add_argument("--num_completions", type=int, default=6)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--rep_penalty", type=float, default=1.0)
    p.add_argument("--gen_batch_size", type=int, default=4)

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
        "--save-gcp",
        type=int,
        default=0,
        dest="save_gcp_every",
        help="Save checkpoint to GCS every N steps (0=disabled). "
        "Uploads to gs://telekinesis-43/checkpoints/",
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

    # Layer rotation
    p.add_argument(
        "--reshuffle_layers_every",
        type=int,
        default=4,
        help="Reshuffle which layers are trained every N optimizer updates.",
    )

    args = p.parse_args()

    model, config, torch_gen = load_model(args.checkpoint)
    train_examples = load_gsm8k(split="train", seed=42, max_examples=args.max_examples)
    # Eval on test split (sample up to max_eval)
    eval_examples = load_gsm8k(split="test", seed=42, max_examples=args.max_eval)
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
        beta=args.beta,
        lr=args.lr,
        optimizer_name=args.optimizer,
        train_layers=args.train_layers,
        num_steps=args.num_steps,
        eval_every=args.eval_every,
        max_eval=args.max_eval,
        eval_num_completions=args.eval_num_completions,
        eval_temperature=args.eval_temperature,
        save_gcp_every=args.save_gcp_every,
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
    )

    # Save final checkpoint (direct from torch_gen to avoid lossy round-trip)
    try:
        import torch as _torch
        if torch_gen is not None and save_path.endswith(".pt"):
            _CACHE_PREFIXES = ('_conv_kernel_t', '_decay_slopes_cached', '_phase_scale_b',
                               '_score_bias_b', '_score_scale_b', '_theta_cached', '_w_int_cached')
            sd = {k: v.cpu() for k, v in torch_gen.state_dict().items()
                  if not any(k.endswith(sfx) for sfx in _CACHE_PREFIXES)}
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            _torch.save({"state_dict": sd, "config": config}, save_path)
        else:
            pkl_path = save_path[:-3] + ".pkl" if save_path.endswith(".pt") else save_path
            save_keras_checkpoint(model, config, pkl_path)
            if save_path.endswith(".pt"):
                keras_pkl_to_torch_pt(pkl_path, save_path)
        print(f"Saved: {save_path}")
    except OSError as e:
        print(f"WARNING: could not save ({e})")


if __name__ == "__main__":
    main()

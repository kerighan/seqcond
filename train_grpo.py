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
import asyncio
import copy
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
    rep_penalty: float = 1.1,
    gen_batch_size: int = 4,
    return_tokens: bool = False,
):
    """Fast generation using PyTorch model with Triton kernels.

    Moves model to CUDA for generation, then back to CPU to free VRAM.
    """
    import torch

    eos_id = tokenizer.encode("<|im_end|>")[0]
    prompt_toks = tokenizer([prompt])[0]
    all_texts, all_ids = [], []

    # Move gen model to GPU
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

            logits_np = logits.cpu().float().numpy()
            generated = [[] for _ in range(B)]
            finished = [False] * B
            token_buf = torch.zeros((B, 1), dtype=torch.long, device="cuda")

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
                        token_buf[i, 0] = toks[i]
                    else:
                        token_buf[i, 0] = eos_id
                if all(finished):
                    break
                logits, states = torch_model.step(token_buf, states, use_triton=True)
                logits_np = logits.cpu().float().numpy()

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


# ─────────────────────────────────────────────────────────────────────────────
# LLM reasoning scorer  (private — called by score_output)
# ─────────────────────────────────────────────────────────────────────────────

_LLM_SYSTEM = """\
You are evaluating the reasoning quality of a math student's solution.
Score the QUALITY OF REASONING 0-100, independent of whether the final answer is correct.
90-100: sound approach, clear logic. 60-89: mostly correct with minor errors.
30-59: partial understanding, key mistakes. 1-29: mostly wrong. 0: gibberish/off-topic.
Reply with ONLY a single integer 0-100.
Give extra points to a chain of thought that actually corrects itself.
"""


async def _score_one(client, question, response, semaphore):
    async with semaphore:
        try:
            msg = await client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Problem: {question}\n\nStudent response:\n{response}\n\nScore (0-100):",
                    },
                ],
                max_tokens=8,
                temperature=0.0,
            )
            raw = msg.choices[0].message.content.strip()
            return min(max(float(re.search(r"\d+", raw).group()), 0.0), 100.0)
        except Exception:
            return 0.0


def _llm_bonuses(question, completions, api_key, max_concurrent=8):
    """Return list of LLM reasoning bonuses in [0, 0.5]."""
    from openai import AsyncOpenAI

    async def _run():
        client = AsyncOpenAI(api_key=api_key)
        sem = asyncio.Semaphore(max_concurrent)
        scores = await asyncio.gather(
            *[_score_one(client, question, r, sem) for r in completions]
        )
        await client.close()
        return scores

    return [s / 200.0 for s in asyncio.run(_run())]  # [0,100] → [0,0.5]


# ═════════════════════════════════════════════════════════════════════════════
# ★  REWARD FUNCTION  —  modify this to change what the model is rewarded for
# ═════════════════════════════════════════════════════════════════════════════


def score_output(
    question: str,
    completions: List[str],
    ground_truth: str,
    api_key: str = None,
    return_components: bool = False,
):
    """Score a group of completions for one question.

    Returns a reward per completion:
        1.2   correct answer + used <|think_end|> format
        1.0   correct answer, no CoT separator
        0.2   wrong answer but used <|think_end|> format  ← keeps format alive
        0.0   wrong answer, no format
        +0.5  bonus for good reasoning quality (needs api_key / OPENAI_API_KEY)

    Modify this function to change what the model learns to do.
    Example ideas:
        - reward brevity (shorter correct answers score higher)
        - reward self-correction in chain of thought
        - drop ground_truth entirely and rely 100% on LLM score (RLAIF)
        - remove format_bonus if you don't care about the CoT format
    """
    FORMAT_BONUS = 0.2  # reward for using <|think_end|> separator

    binary = []
    format_bonus = []
    rewards = []
    for c in completions:
        correct = 1.0 if check_answer(c, ground_truth) else 0.0
        fmt = FORMAT_BONUS if "<|think_end|>" in c else 0.0
        binary.append(correct)
        format_bonus.append(fmt)
        rewards.append(correct + fmt)

    llm_bonus = [0.0] * len(completions)
    if not api_key:
        if return_components:
            return {
                "rewards": rewards,
                "binary": binary,
                "format_bonus": format_bonus,
                "llm_bonus": llm_bonus,
            }
        return rewards
    llm_bonus = _llm_bonuses(question, completions, api_key)
    rewards = [r + bon for r, bon in zip(rewards, llm_bonus)]
    if return_components:
        return {
            "rewards": rewards,
            "binary": binary,
            "format_bonus": format_bonus,
            "llm_bonus": llm_bonus,
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
    rep_penalty: float = 1.1,
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


def _compute_advantages(rewards):
    """Group-relative advantages: (r - mean) / (std + eps)."""
    r = np.array(rewards, dtype=np.float32)
    std = r.std()
    return np.zeros_like(r) if std < 1e-8 else (r - r.mean()) / std


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


# ═════════════════════════════════════════════════════════════════════════════
# ★  GRPO FORMULA  —  modify this to change how rewards drive the update
# ═════════════════════════════════════════════════════════════════════════════


def grpo_step(
    model,
    prompt_tokens: List[int],
    completions_tokens: List[List[int]],
    advantages: np.ndarray,
    *,
    beta: float = 0.1,
    ref_model=None,
    grad_scale: float = 1.0,
) -> float:
    """Compute GRPO loss and accumulate gradients for one group of completions.

    For each completion i with non-zero advantage:
        loss_i = -advantage_i * avg_log_prob_i  +  β * KL(π || π_ref)

    where KL is approximated as (log π - log π_ref) per token, averaged.

    Returns total accumulated loss (float, for logging only).

    Modify this function to experiment with the RL algorithm, e.g.:
        - remove KL penalty (set beta=0 or delete the kl term)
        - use clipped surrogate (PPO-style): clip advantage * ratio
        - weight by completion length
        - add entropy bonus
    """
    prompt_len = len(prompt_tokens)

    # Reference log probs (frozen initial model, or current snapshot if no ref)
    ref = ref_model if ref_model is not None else model
    ref_lps = []
    with inference_mode():
        for comp in completions_tokens:
            if len(comp) == 0:
                ref_lps.append(0.0)
                continue
            ids = np.array([prompt_tokens + comp], dtype=np.int32)
            ref_lps.append(float(_seq_log_prob(ref, ids, prompt_len)) / len(comp))

    # Policy gradient with grad accumulation
    total_loss = 0.0
    n = len(completions_tokens)
    for i, comp in enumerate(completions_tokens):
        if len(comp) == 0 or advantages[i] == 0:
            continue
        ids = np.array([prompt_tokens + comp], dtype=np.int32)
        policy_lp = _seq_log_prob(model, ids, prompt_len)
        avg_lp = policy_lp / len(comp)
        kl = avg_lp - ref_lps[i]
        loss_i = (-advantages[i] * avg_lp + beta * kl) / n
        (loss_i * grad_scale).backward()
        total_loss += float(to_numpy(loss_i))

    return total_loss


def gmpo_step(
    model,
    prompt_tokens: List[int],
    completions_tokens: List[List[int]],
    advantages: np.ndarray,
    *,
    clip_eps: float = 0.4,
    grad_scale: float = 1.0,
) -> float:
    """Compute a GMPO/GSPO-style loss in log-space.

    We keep the GMPO geometric-mean sequence ratio, but apply clipping at the
    sequence level in the GSPO spirit:
    - compute token log-probs under an "old" snapshot (no grad)
    - average token log-ratios across the completion (length-normalized)
    - clip the signed sequence log-ratio once per response
    - exponentiate back to a sequence importance ratio
    """
    prompt_len = len(prompt_tokens)

    old_token_lps = []
    with inference_mode():
        for comp in completions_tokens:
            if len(comp) == 0:
                old_token_lps.append(None)
                continue
            ids = np.array([prompt_tokens + comp], dtype=np.int32)
            old_token_lps.append(_seq_token_log_probs(model, ids, prompt_len))

    total_loss = 0.0
    n = len(completions_tokens)
    for i, comp in enumerate(completions_tokens):
        if len(comp) == 0 or advantages[i] == 0:
            continue

        ids = np.array([prompt_tokens + comp], dtype=np.int32)
        new_token_lps = _seq_token_log_probs(model, ids, prompt_len)
        old_lps = old_token_lps[i]

        adv = float(advantages[i])
        sign = 1.0 if adv > 0 else -1.0
        seq_log_ratio = ops.mean(new_token_lps - old_lps)
        signed_seq_log_ratio = sign * seq_log_ratio
        clipped_signed_seq_log_ratio = ops.clip(
            signed_seq_log_ratio, -clip_eps, clip_eps
        )
        min_signed_seq_log_ratio = ops.minimum(
            signed_seq_log_ratio, clipped_signed_seq_log_ratio
        )
        clipped_seq_log_ratio = sign * min_signed_seq_log_ratio
        seq_ratio = ops.exp(clipped_seq_log_ratio)

        loss_i = (-adv * seq_ratio) / n
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


def evaluate(
    torch_gen,
    examples,
    max_examples=100,
    num_completions=1,
    max_new_tokens=512,
    temperature=0.0,
    rep_penalty=1.1,
    gen_batch_size=4,
):
    """Evaluate pass@k accuracy on the first max_examples problems."""
    tokenizer = Tokenizer()
    examples = examples[:max_examples]
    correct = 0
    t0 = time.time()
    for i, ex in enumerate(examples):
        comps = generate_completions_torch(
            torch_gen,
            tokenizer,
            ex["prompt"],
            num_completions=num_completions,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            rep_penalty=rep_penalty,
            gen_batch_size=gen_batch_size,
        )
        ok = any(check_answer(c, ex["ground_truth"]) for c in comps)
        correct += int(ok)
        print(
            f"  [{i+1}/{len(examples)}] {'✓' if ok else '✗'}  "
            f"pass@{num_completions}={100*correct/(i+1):.1f}%  "
            f"gt={ex['ground_truth']}"
        )
    acc = correct / len(examples)
    print(f"\n  pass@{num_completions}: {100*acc:.1f}%  ({time.time()-t0:.0f}s)\n")
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────


def _save_and_upload_gcp(model, config, save_path, step):
    import subprocess

    gcs_bucket = "gs://telekinesis-43/checkpoints"

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
        result = subprocess.run(
            ["gsutil", "cp", save_path, gcs_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            print("✓")
        else:
            err = result.stderr.strip() or result.stdout.strip() or "unknown error"
            print(f"✗ ({err[:120]})")
    except Exception as e:
        print(f"✗ ({str(e)[:120]})")


def train_grpo(
    model,
    config,
    examples,
    *,
    torch_gen=None,
    sync_every: int = 20,
    use_gmpo: bool = False,
    num_completions: int = 6,
    max_new_tokens: int = 512,
    temperature: float = 0.5,
    rep_penalty: float = 1.1,
    gen_batch_size: int = 4,
    beta: float = 0.04,
    lr: float = 5e-5,
    optimizer_name: str = "adamw",
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    train_layers: int = 3,
    warmup_steps: int = 20,
    min_completion_tokens: int = 5,
    llm_api_key: str = None,
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
):
    import torch

    tokenizer = Tokenizer()
    np.random.seed(seed)
    random.seed(seed)

    n_blocks = len(model.blocks_list)
    train_layers = min(train_layers, n_blocks)

    all_params = list(model.parameters())
    optimizer = (
        torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
        if optimizer_name == "adamw"
        else torch.optim.SGD(all_params, lr=lr, momentum=0.0, weight_decay=weight_decay)
    )

    use_fast_gen = torch_gen is not None
    max_total_tokens = int(config.get("maxlen") or 0)

    # Frozen reference model for KL penalty (anchors to initial policy)
    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad = False

    if grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")

    objective_name = "GMPO" if use_gmpo else "GRPO"

    print(
        f"\n── {objective_name}  {num_steps} steps  G={num_completions}  "
        f"lr={lr}  β={beta}  train_layers={train_layers}/{n_blocks}  "
        f"warmup={warmup_steps}  accum={grad_accum_steps}"
        f"{'  sync_every=' + str(sync_every) if use_fast_gen else ''} ──\n"
    )

    t0 = time.time()
    run_reward = run_loss = run_correct = run_skipped = 0.0
    run_count = run_adv_abs = 0.0
    pending_grad_steps = 0
    optimizer.zero_grad()

    if save_gcp_every > 0 and save_path:
        print("  [step 0 backup]")
        _save_and_upload_gcp(model, config, save_path, 0)

    for step in range(1, num_steps + 1):
        # Sync torch gen model periodically
        if use_fast_gen and (step == 1 or step % sync_every == 0):
            sync_keras_to_torch(model, torch_gen, config)
            if step > 1:
                print(f"  [sync weights → torch gen model at step {step}]")

        # Randomly unfreeze train_layers blocks (+ embedding)
        for p in model.parameters():
            p.requires_grad = False
        for idx in random.sample(range(n_blocks), train_layers):
            for p in model.blocks_list[idx].parameters():
                p.requires_grad = True
        for p in model.token_embedding.parameters():
            p.requires_grad = True

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
            llm_api_key,
            return_components=True,
        )
        rewards = score_info["rewards"]
        binary = score_info["binary"]
        llm_bonus = score_info["llm_bonus"]
        if llm_api_key:
            print(
                f"  reward={[f'{r:.2f}' for r in rewards]}  "
                f"llm_bonus={[f'{b:.2f}' for b in llm_bonus]}  "
                f"binary={[int(b) for b in binary]}"
            )

        advantages = _compute_advantages(rewards)
        run_reward += sum(rewards)
        run_count += len(rewards)
        run_correct += int(sum(binary))
        run_adv_abs += float(np.mean(np.abs(advantages)))

        if np.all(advantages == 0):
            run_skipped += 1
        else:
            # LR warmup: linear ramp for first warmup_steps
            if warmup_steps > 0 and step <= warmup_steps:
                scale = step / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * scale
            elif warmup_steps > 0 and step == warmup_steps + 1:
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            # Policy update (gradients accumulate across steps)
            if use_gmpo:
                loss = gmpo_step(
                    model,
                    prompt_tokens,
                    ids_f,
                    advantages,
                    grad_scale=1.0 / grad_accum_steps,
                )
            else:
                loss = grpo_step(
                    model,
                    prompt_tokens,
                    ids_f,
                    advantages,
                    beta=beta,
                    ref_model=ref_model,
                    grad_scale=1.0 / grad_accum_steps,
                )
            pending_grad_steps += 1
            run_loss += loss

            if pending_grad_steps >= grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
                pending_grad_steps = 0

        # Log
        if step % log_every == 0:
            avg_r = run_reward / max(run_count, 1.0)
            avg_adv = run_adv_abs / log_every
            elapsed = time.time() - t0
            eta = elapsed / step * (num_steps - step)
            print(
                f"  Step {step:4d}/{num_steps} | loss={run_loss:.4f} | "
                f"reward_avg={avg_r:.3f} | adv_abs={avg_adv:.3f} | "
                f"correct={int(run_correct)}/{int(run_count)} | skip={int(run_skipped)} | "
                f"ETA {int(eta//60):02d}:{int(eta%60):02d}"
            )
            run_reward = run_loss = run_correct = run_skipped = 0.0
            run_count = run_adv_abs = 0.0

        # Flush pending gradients before eval
        if eval_every > 0 and step % eval_every == 0 and pending_grad_steps > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            pending_grad_steps = 0

        # Periodic eval
        if eval_every > 0 and step % eval_every == 0:
            if use_fast_gen:
                sync_keras_to_torch(model, torch_gen, config)
            evaluate(
                torch_gen if use_fast_gen else model,
                examples,
                max_examples=max_eval,
                num_completions=eval_num_completions,
                max_new_tokens=max_new_tokens,
                temperature=eval_temperature,
                rep_penalty=rep_penalty,
                gen_batch_size=gen_batch_size,
            )

        # Flush pending gradients before backup
        if (
            save_gcp_every > 0
            and step % save_gcp_every == 0
            and save_path
            and pending_grad_steps > 0
        ):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            pending_grad_steps = 0

        # Periodic GCP backup
        if save_gcp_every > 0 and step % save_gcp_every == 0 and save_path:
            _save_and_upload_gcp(model, config, save_path, step)

    # Flush any remaining accumulated gradients
    if pending_grad_steps > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    print(f"\n── {objective_name} complete ({time.time()-t0:.0f}s) ──\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="GRPO fine-tuning for SeqCond")
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
    p.add_argument("--rep_penalty", type=float, default=1.1)
    p.add_argument("--gen_batch_size", type=int, default=4)

    # Training
    p.add_argument(
        "--sync_every",
        type=int,
        default=20,
        help="Sync torch gen model weights every N steps (default: 20)",
    )
    p.add_argument("--num_steps", type=int, default=250)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--train_layers", type=int, default=3)
    p.add_argument("--optimizer", default="adamw", choices=["sgd", "adamw"])
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--max_eval", type=int, default=100)
    p.add_argument("--eval_num_completions", type=int, default=1)
    p.add_argument("--eval_temperature", type=float, default=0.0)
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

    # Reward
    p.add_argument(
        "--openai_api_key",
        default=None,
        help="OpenAI key for LLM reward (falls back to OPENAI_API_KEY env var)",
    )

    # GMPO
    p.add_argument(
        "--use-gmpo",
        action="store_true",
        help="Use GMPO objective instead of the default GRPO objective",
    )

    args = p.parse_args()

    model, config, torch_gen = load_model(args.checkpoint)
    examples = load_gsm8k(split="train", seed=42, max_examples=args.max_examples)
    print(f"Dataset: {len(examples)} GSM8K training examples")

    if not args.skip_baseline:
        print("\n── Baseline eval ──")
        evaluate(
            torch_gen,
            examples,
            max_examples=args.max_eval,
            num_completions=args.eval_num_completions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.eval_temperature,
            rep_penalty=args.rep_penalty,
            gen_batch_size=args.gen_batch_size,
        )

    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY") or None
    print(f"LLM reward: {'enabled (gpt-4.1-mini)' if api_key else 'disabled'}")

    save_path = args.save or os.path.join(
        "checkpoints",
        os.path.splitext(os.path.basename(args.checkpoint))[0] + "_grpo.pt",
    )

    train_grpo(
        model,
        config,
        examples,
        torch_gen=torch_gen,
        sync_every=args.sync_every,
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
        llm_api_key=api_key,
        num_steps=args.num_steps,
        eval_every=args.eval_every,
        max_eval=args.max_eval,
        eval_num_completions=args.eval_num_completions,
        eval_temperature=args.eval_temperature,
        save_gcp_every=args.save_gcp_every,
        save_path=save_path,
        grad_accum_steps=args.grad_accum_steps,
    )

    # Save: .pkl intermediary → .pt with correct PyTorch key mapping
    try:
        pkl_path = save_path[:-3] + ".pkl" if save_path.endswith(".pt") else save_path
        save_keras_checkpoint(model, config, pkl_path)
        if save_path.endswith(".pt"):
            keras_pkl_to_torch_pt(pkl_path, save_path)
        print(f"Saved: {save_path}")
    except OSError as e:
        print(f"WARNING: could not save ({e})")


if __name__ == "__main__":
    main()

"""
GRPO (Group Relative Policy Optimization) training for SeqCond via Keras 3.

Usage:
    KERAS_BACKEND=jax python train_grpo.py --checkpoint checkpoints/seqcond_torch_762k.pt
    KERAS_BACKEND=jax python train_grpo.py --checkpoint checkpoints/seqcond_lin5.pt --summary_only
"""

import argparse
import os
import re
import random
from typing import List, Dict, Optional

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import ops

from convert_torch_to_keras import (
    load_torch_checkpoint,
    build_keras_model,
    convert_weights,
    save_keras_checkpoint,
    keras_pkl_to_torch_pt,
)
from seqcond.dataset import Tokenizer


# ═════════════════════════════════════════════════════════════════════════
# Backend-agnostic helpers
# ═════════════════════════════════════════════════════════════════════════


def to_numpy(x):
    """Convert Keras tensor to numpy array (works with JAX and Torch backends)."""
    # For torch backend, tensors may be on GPU and require grad
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.array(x)


from contextlib import contextmanager


@contextmanager
def inference_mode():
    """Disable gradient tracking for generation (torch) or act as no-op (jax)."""
    backend = keras.backend.backend()
    if backend == "torch":
        import torch

        with torch.no_grad():
            yield
    else:
        yield


# ═════════════════════════════════════════════════════════════════════════
# GSM8K Dataset
# ═════════════════════════════════════════════════════════════════════════


def extract_answer(answer_text: str) -> str:
    """Extract the final numerical answer from GSM8K answer text.

    GSM8K answers end with '#### <number>'. We extract and normalize
    the number (strip commas, whitespace).
    """
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return ""


def load_gsm8k(
    split: str = "train",
    seed: int = 42,
    max_examples: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Load GSM8K dataset and format for GRPO.

    Returns a list of dicts with keys:
        - question: the math problem
        - answer: the full step-by-step solution
        - ground_truth: the final numerical answer (string)
        - prompt: formatted prompt ready for the model
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)

    examples = []
    for ex in ds:
        gt = extract_answer(ex["answer"])
        if not gt:
            continue
        examples.append(
            {
                "question": ex["question"],
                "answer": ex["answer"],
                "ground_truth": gt,
                "prompt": (
                    "<|im_start|>user\n"
                    + ex["question"]
                    + "\n<|im_end|><|im_start|>assistant\n<|think_start|>"
                ),
            }
        )

    rng = random.Random(seed)
    rng.shuffle(examples)

    if max_examples is not None:
        examples = examples[:max_examples]

    return examples


def check_answer(generated_text: str, ground_truth: str) -> bool:
    """Check if the generated text contains the correct numerical answer.

    Looks for the number after the last '####' or after the think_end token,
    or just anywhere in the final answer portion.
    """
    # Try to find #### pattern first (model might learn to use it)
    match = re.search(r"####\s*(.+)", generated_text)
    if match:
        candidate = match.group(1).strip().replace(",", "")
        return candidate == ground_truth

    # Otherwise look for the answer after <|think_end|>
    parts = generated_text.split("<|think_end|>")
    if len(parts) > 1:
        after_think = parts[-1].strip()
    else:
        after_think = generated_text.strip()

    # Extract all numbers from the answer portion
    numbers = re.findall(r"-?[\d,]+\.?\d*", after_think)
    # Normalize and check
    for num in numbers:
        normalized = num.replace(",", "").rstrip(".")
        if normalized == ground_truth:
            return True

    return False


# ═════════════════════════════════════════════════════════════════════════
# LLM reasoning reward  (GPT-4.1-mini, async parallel)
# ═════════════════════════════════════════════════════════════════════════

_LLM_SYSTEM = """\
You are evaluating the reasoning quality of a math student's solution.
Score the QUALITY OF REASONING 0-100, independent of whether the final answer is correct.
90-100: sound approach, clear logic. 60-89: mostly correct with minor errors.
30-59: partial understanding, key mistakes. 1-29: mostly wrong. 0: gibberish/off-topic.
Reply with ONLY a single integer 0-100.
Give extra points to a chain of thought that actually corrects itself.
"""

_LLM_USER = "Problem: {question}\n\nStudent response:\n{response}\n\nScore (0-100):"


async def _score_one(client, question: str, response: str, semaphore) -> float:
    async with semaphore:
        try:
            msg = await client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM},
                    {
                        "role": "user",
                        "content": _LLM_USER.format(
                            question=question, response=response
                        ),
                    },
                ],
                max_tokens=8,
                temperature=0.0,
            )
            raw = msg.choices[0].message.content.strip()
            score = float(re.search(r"\d+", raw).group())
            return min(max(score, 0.0), 100.0)
        except Exception:
            return 0.0


def llm_reasoning_scores(
    question: str, responses: list, api_key: str, max_concurrent: int = 8
) -> list:
    """Return list of reasoning bonuses in [0, 0.5] for each response."""
    import asyncio
    from openai import AsyncOpenAI

    async def _run():
        client = AsyncOpenAI(api_key=api_key)
        sem = asyncio.Semaphore(max_concurrent)
        scores = await asyncio.gather(
            *[_score_one(client, question, r, sem) for r in responses]
        )
        await client.close()
        return scores

    raw = asyncio.run(_run())
    return [s / 100.0 * 0.5 for s in raw]


def compute_rewards(
    question: str, completions: list, ground_truth: str, api_key: str = None
) -> list:
    """Binary reward + optional LLM reasoning bonus.

    Returns rewards in [0, 1.5]:
      - correct + perfect reasoning = 1.5
      - correct only               = 1.0  (no API key)
      - wrong + perfect reasoning  = 0.5  (always < any correct)
      - wrong + no reasoning       = 0.0
    """
    binary = [1.0 if check_answer(c, ground_truth) else 0.0 for c in completions]
    if not api_key:
        return binary
    bonuses = llm_reasoning_scores(question, completions, api_key)
    return [b + bon for b, bon in zip(binary, bonuses)]


def print_dataset_summary(examples: List[Dict[str, str]]):
    """Print a summary of the loaded dataset."""
    print(f"\n{'═' * 70}")
    print(f"  GSM8K Dataset")
    print(f"{'═' * 70}")
    print(f"  Examples loaded  : {len(examples)}")

    # Answer distribution stats
    answers = [ex["ground_truth"] for ex in examples]
    numeric = []
    for a in answers:
        try:
            numeric.append(float(a))
        except ValueError:
            pass
    if numeric:
        print(f"  Answer range     : [{min(numeric):.0f}, {max(numeric):.0f}]")
        print(f"  Median answer    : {sorted(numeric)[len(numeric)//2]:.0f}")

    # Prompt length stats
    tokenizer = Tokenizer()
    prompt_tokens = [len(tokenizer([ex["prompt"]])[0]) for ex in examples[:200]]
    print(
        f"  Prompt tokens    : {min(prompt_tokens)}-{max(prompt_tokens)} "
        f"(avg {sum(prompt_tokens)/len(prompt_tokens):.0f})"
    )

    # Show a few examples
    print(f"\n  Sample problems:")
    print(f"  {'─' * 60}")
    for i, ex in enumerate(examples[:3]):
        q = ex["question"][:80] + ("..." if len(ex["question"]) > 80 else "")
        print(f"  [{i}] {q}")
        print(f"      Answer: {ex['ground_truth']}")

    print(f"\n{'═' * 70}\n")


# ═════════════════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════════════════


def load_model(torch_checkpoint_path: str):
    """Load Keras SeqCond model from a PyTorch checkpoint."""
    print(f"{'=' * 70}")
    print(f"  Loading model from: {torch_checkpoint_path}")
    print(f"  Keras backend: {keras.backend.backend()}")
    print(f"{'=' * 70}")

    config, state_dict = load_torch_checkpoint(torch_checkpoint_path)
    model = build_keras_model(config)
    n_assigned = convert_weights(config, state_dict, model)

    return model, config


def print_model_summary(model, config):
    """Print a detailed, readable model summary."""
    print(f"\n{'═' * 70}")
    print(f"  SeqCond Model Summary")
    print(f"{'═' * 70}")

    # Architecture info
    n_seqcond = sum(1 for bt in model.block_types if bt == "seqcond")
    n_transformer = sum(1 for bt in model.block_types if bt == "transformer")

    print(f"\n  Architecture")
    print(f"  {'─' * 40}")
    print(f"  d_model          : {model.d_model}")
    print(f"  d_ff             : {model.d_ff}")
    print(f"  num_layers       : {model.num_layers_total}")
    print(f"  vocab_size       : {model.vocab_size}")
    print(f"  maxlen           : {model.maxlen}")
    print(f"  tie_weights      : {model.tie_weights}")

    print(f"\n  Block composition")
    print(f"  {'─' * 40}")
    print(f"  SeqCond blocks   : {n_seqcond}")
    print(f"  Transformer blocks: {n_transformer}")
    print(f"  Ratio            : {model.seqcond_ratio} seqcond per transformer")

    # Block pattern (compact)
    pattern = ""
    for bt in model.block_types:
        pattern += "S" if bt == "seqcond" else "T"
    # Wrap at 40 chars
    print(f"  Pattern          : ", end="")
    for i in range(0, len(pattern), 40):
        if i > 0:
            print(f"  {'':19s}", end="")
        print(pattern[i : i + 40])

    # Parameter count by category
    print(f"\n  Parameters")
    print(f"  {'─' * 40}")

    emb_params = 0
    seqcond_params = 0
    transformer_params = 0
    other_params = 0

    for w in model.weights:
        n = int(np.prod(w.shape))
        path = w.path.lower()
        if "embedding" in path:
            emb_params += n
        elif "seqcond" in path:
            seqcond_params += n
        elif "transformer" in path:
            transformer_params += n
        else:
            other_params += n

    total = model.count_params()
    print(f"  Embedding        : {emb_params:>12,}  ({100*emb_params/total:5.1f}%)")
    print(
        f"  SeqCond layers   : {seqcond_params:>12,}  ({100*seqcond_params/total:5.1f}%)"
    )
    print(
        f"  Transformer layers: {transformer_params:>12,}  ({100*transformer_params/total:5.1f}%)"
    )
    if other_params > 0:
        print(
            f"  Other            : {other_params:>12,}  ({100*other_params/total:5.1f}%)"
        )
    print(f"  {'─' * 40}")
    print(f"  Total            : {total:>12,}")

    # Trainable vs non-trainable
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    non_trainable = total - trainable
    print(f"  Trainable        : {trainable:>12,}")
    print(f"  Non-trainable    : {non_trainable:>12,}")

    # Memory estimate (float32)
    mem_f32 = total * 4 / (1024**3)
    mem_bf16 = total * 2 / (1024**3)
    print(f"\n  Memory estimate")
    print(f"  {'─' * 40}")
    print(f"  float32          : {mem_f32:.2f} GB")
    print(f"  bfloat16         : {mem_bf16:.2f} GB")
    print(f"  GRPO (2× model)  : {2*mem_bf16:.2f} GB  (policy + ref, bf16)")

    print(f"\n{'═' * 70}\n")


# ═════════════════════════════════════════════════════════════════════════
# Generation
# ═════════════════════════════════════════════════════════════════════════


def _tile_states(states, n):
    """Replicate states from batch_size=1 to batch_size=n.

    Uses ops.tile to keep tensors in the correct backend format.
    """
    tiled = []
    for state in states:
        tiled_state = tuple(ops.tile(s, (n,) + (1,) * (s.ndim - 1)) for s in state)
        tiled.append(tiled_state)
    return tiled


def sample_tokens_batch(logits_np, temperature=0.7, top_k=50, top_p=0.95):
    """Sample one token per row from (B, vocab) logits. Returns (B,) int array."""
    B = logits_np.shape[0]
    tokens = np.empty(B, dtype=np.int64)

    for i in range(B):
        row = logits_np[i].astype(np.float64)

        if temperature <= 0.0:
            tokens[i] = np.argmax(row)
            continue

        row = row / temperature

        # Top-k
        if top_k > 0 and top_k < len(row):
            topk_idx = np.argpartition(row, -top_k)[-top_k:]
            topk_vals = row[topk_idx]
        else:
            topk_idx = np.arange(len(row))
            topk_vals = row

        # Stable softmax
        topk_vals = topk_vals - np.max(topk_vals)
        probs = np.exp(topk_vals)
        probs = probs / np.sum(probs)

        # Top-p
        if top_p < 1.0:
            sorted_order = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_order])
            cutoff = np.searchsorted(cumsum, top_p) + 1
            nucleus = sorted_order[:cutoff]
            p = probs[nucleus]
            p = p / p.sum()
            chosen = np.random.choice(nucleus, p=p)
        else:
            chosen = np.random.choice(len(probs), p=probs)

        tokens[i] = topk_idx[chosen]

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
    """Generate multiple completions for a single prompt.

    Prefills once (batch_size=1), then tiles states and decodes
    gen_batch_size completions in parallel.

    If return_tokens=True, returns (texts, token_id_lists).
    """
    eos_id = tokenizer.encode("<|im_end|>")[0]
    tokens = tokenizer([prompt])[0]

    all_completions = []
    all_token_ids = []

    with inference_mode():
        for batch_start in range(0, num_completions, gen_batch_size):
            B = min(gen_batch_size, num_completions - batch_start)

            # Prefill: step through prompt (batch_size=1)
            states = model.init_state(batch_size=1)
            for t in range(len(tokens)):
                tok_arr = np.array([[tokens[t]]], dtype=np.int32)
                logits, states = model.step(tok_arr, states)

            # Tile to B completions
            if B > 1:
                states = _tile_states(states, B)
            logits_np = np.tile(to_numpy(logits), (B, 1))  # (B, vocab)

            # Decode
            generated = [[] for _ in range(B)]
            finished = [False] * B
            token_buf = np.zeros((B, 1), dtype=np.int32)

            for _ in range(max_new_tokens):
                if rep_penalty != 1.0:
                    for i in range(B):
                        for tok_id in set(generated[i]):
                            if logits_np[i, tok_id] > 0:
                                logits_np[i, tok_id] /= rep_penalty
                            else:
                                logits_np[i, tok_id] *= rep_penalty
                toks = sample_tokens_batch(logits_np, temperature, top_k, top_p)

                for i in range(B):
                    if not finished[i]:
                        generated[i].append(int(toks[i]))
                        if toks[i] == eos_id:
                            finished[i] = True
                        token_buf[i, 0] = toks[i]
                    else:
                        token_buf[i, 0] = eos_id

                if all(finished):
                    break

                logits, states = model.step(token_buf, states)
                logits_np = to_numpy(logits)

            for i in range(B):
                toks = generated[i]
                if toks and toks[-1] == eos_id:
                    toks = toks[:-1]
                all_token_ids.append(list(toks))
                try:
                    all_completions.append(tokenizer.decode(toks))
                except Exception:
                    all_completions.append("")

    if return_tokens:
        return all_completions, all_token_ids
    return all_completions


def evaluate_gsm8k(
    model,
    examples: List[Dict[str, str]],
    num_completions: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    rep_penalty: float = 1.0,
    gen_batch_size: int = 4,
    max_eval: int = 50,
    seed: int = 42,
):
    """Evaluate baseline accuracy on GSM8K.

    For each example, generate num_completions answers and check correctness.
    Reports per-example and overall accuracy.
    """
    tokenizer = Tokenizer()
    np.random.seed(seed)

    eval_examples = examples[:max_eval]
    total_correct = 0
    total_generated = 0
    per_example_acc = []

    print(f"\n{'═' * 70}")
    print(f"  GSM8K Baseline Evaluation")
    print(f"  {len(eval_examples)} problems × {num_completions} completions")
    print(f"  temperature={temperature}  top_k={top_k}  top_p={top_p}")
    print(f"  max_new_tokens={max_new_tokens}  gen_batch_size={gen_batch_size}")
    print(f"{'═' * 70}")

    import time

    t0 = time.time()

    for idx, ex in enumerate(eval_examples):
        completions = generate_completions(
            model,
            tokenizer,
            ex["prompt"],
            num_completions=num_completions,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rep_penalty=rep_penalty,
            gen_batch_size=gen_batch_size,
        )

        correct = sum(1 for c in completions if check_answer(c, ex["ground_truth"]))
        total_correct += correct
        total_generated += len(completions)
        acc = correct / len(completions)
        per_example_acc.append(acc)

        elapsed = time.time() - t0
        avg_time = elapsed / (idx + 1)
        remaining = avg_time * (len(eval_examples) - idx - 1)

        # Show progress
        status = "✓" if correct > 0 else "✗"
        print(
            f"  [{idx+1:3d}/{len(eval_examples)}] {status} "
            f"{correct}/{len(completions)} correct  "
            f"(gt={ex['ground_truth']:>8s})  "
            f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s left]"
        )

        # Show first example's completions in detail
        if idx < 2:
            q = ex["question"][:60] + ("..." if len(ex["question"]) > 60 else "")
            print(f"         Q: {q}")
            for ci, c in enumerate(completions[:2]):
                # Show last 80 chars (the answer part)
                tail = c[-80:] if len(c) > 80 else c
                ok = "✓" if check_answer(c, ex["ground_truth"]) else "✗"
                print(f"         {ok} [{ci}] ...{tail}")

    elapsed = time.time() - t0
    overall_acc = total_correct / max(total_generated, 1)
    any_correct = sum(1 for a in per_example_acc if a > 0) / max(
        len(per_example_acc), 1
    )

    print(f"\n{'─' * 70}")
    print(f"  Results ({elapsed:.1f}s total):")
    print(
        f"  Overall accuracy     : {total_correct}/{total_generated} = {100*overall_acc:.1f}%"
    )
    print(f"  Pass@{num_completions} (any correct): {100*any_correct:.1f}%")
    print(f"  Mean per-example acc : {100*np.mean(per_example_acc):.1f}%")
    print(f"{'═' * 70}\n")

    return {
        "overall_accuracy": overall_acc,
        "pass_at_k": any_correct,
        "per_example": per_example_acc,
    }


# ═════════════════════════════════════════════════════════════════════════
# GRPO Training
# ═════════════════════════════════════════════════════════════════════════


def _freeze_all(model):
    """Freeze all model parameters."""
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_blocks(model, block_indices):
    """Unfreeze specific blocks by index + always unfreeze embedding & output."""
    # Unfreeze selected blocks
    for idx in block_indices:
        for p in model.blocks_list[idx].parameters():
            p.requires_grad = True
    # Always unfreeze token embedding (needed for tied output)
    for p in model.token_embedding.parameters():
        p.requires_grad = True
    # Unfreeze output projection if not tied
    if not model.tie_weights and hasattr(model, "output_dense"):
        for p in model.output_dense.parameters():
            p.requires_grad = True


def compute_sequence_log_probs(model, input_ids, prompt_len):
    """Compute per-token log probs for completion tokens.

    Args:
        model: SeqCond model
        input_ids: (1, T) int32 array — prompt + completion tokens
        prompt_len: where the completion starts

    Returns:
        per_token_lps: (completion_len,) tensor
        total_lp: scalar — sum of per-token log probs
    """
    logits = model(input_ids)  # (1, T, V)
    log_probs = ops.log_softmax(logits, axis=-1)  # (1, T, V)

    # Completion tokens are at input_ids[0, prompt_len:]
    # Their log probs come from logits at positions prompt_len-1 to T-2
    shift_lps = log_probs[0, prompt_len - 1 : -1, :]  # (comp_len, V)
    targets = input_ids[0, prompt_len:]  # (comp_len,)

    # Gather per-token log probs
    targets_2d = ops.expand_dims(ops.cast(targets, "int32"), -1)  # (comp_len, 1)
    per_token_lps = ops.squeeze(
        ops.take_along_axis(shift_lps, targets_2d, axis=-1), axis=-1
    )  # (comp_len,)

    return per_token_lps, ops.sum(per_token_lps)


def compute_advantages(rewards):
    """Group-relative advantages: (r - mean) / (std + eps)."""
    r = np.array(rewards, dtype=np.float32)
    std = r.std()
    if std < 1e-8:
        return np.zeros_like(r)
    return (r - r.mean()) / std


def train_grpo(
    model,
    examples,
    *,
    num_completions=4,
    max_new_tokens=512,
    temperature=0.7,
    gen_batch_size=4,
    beta=0.1,
    lr=1e-4,
    optimizer_name="sgd",
    weight_decay=0.01,
    max_grad_norm=1.0,
    train_layers=2,
    rep_penalty=1.0,
    llm_api_key=None,
    num_steps=100,
    log_every=1,
    eval_every=50,
    max_eval=20,
    seed=42,
):
    """GRPO training loop with random layer unfreezing.

    For each step:
      0. Freeze all, randomly unfreeze `train_layers` blocks (+ embedding)
      1. Sample a prompt, generate G completions (no grad)
      2. Score completions → binary rewards → group-relative advantages
      3. Compute reference log probs (no grad, current model snapshot)
      4. For each completion with nonzero advantage:
         - Forward pass with grad → policy log probs
         - Loss = -advantage * avg_log_prob + β * KL
         - Backward (grad accumulation over group)
      5. Clip gradients, optimizer step
    """
    import torch

    tokenizer = Tokenizer()
    np.random.seed(seed)
    random.seed(seed)

    n_blocks = len(model.blocks_list)
    train_layers = min(train_layers, n_blocks)

    # SGD without momentum = zero optimizer state overhead
    # We pass ALL params; only unfrozen ones get gradients
    all_params = list(model.parameters())
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(
            all_params, lr=lr, momentum=0.0, weight_decay=weight_decay
        )

    print(f"\n{'═' * 70}")
    print(f"  GRPO Training")
    print(f"  {num_steps} steps, G={num_completions} completions/prompt")
    print(f"  optimizer={optimizer_name}, lr={lr}, β={beta}, temp={temperature}")
    print(f"  train_layers={train_layers}/{n_blocks} random blocks per step")
    print(f"  max_new_tokens={max_new_tokens}, gen_batch_size={gen_batch_size}")
    print(f"{'═' * 70}\n")

    import time

    t0 = time.time()
    running_reward = 0.0
    running_loss = 0.0
    running_count = 0
    running_correct = 0
    running_skipped = 0

    for step in range(1, num_steps + 1):
        # ── 0. Random layer selection ──────────────────────────────
        _freeze_all(model)
        chosen = random.sample(range(n_blocks), train_layers)
        _unfreeze_blocks(model, chosen)

        ex = random.choice(examples)
        prompt = ex["prompt"]
        gt = ex["ground_truth"]
        prompt_tokens = tokenizer([prompt])[0]
        prompt_len = len(prompt_tokens)

        # ── 1. Generate completions (no grad) ────────────────────────
        completions_text, completions_tokens = generate_completions(
            model,
            tokenizer,
            prompt,
            num_completions=num_completions,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            rep_penalty=rep_penalty,
            gen_batch_size=gen_batch_size,
            return_tokens=True,
        )

        # ── 2. Score & compute advantages ─────────────────────────────
        rewards = compute_rewards(ex["question"], completions_text, gt, llm_api_key)
        binary = [1.0 if check_answer(c, gt) else 0.0 for c in completions_text]
        if llm_api_key:
            print(
                f"  llm rewards={[f'{r:.2f}' for r in rewards]}  "
                f"binary={[int(b) for b in binary]}"
            )
        advantages = compute_advantages(rewards)

        running_reward += sum(rewards)
        running_count += len(rewards)
        running_correct += int(sum(binary))

        # Skip if no variance in rewards (all correct or all wrong)
        if np.all(advantages == 0):
            running_skipped += 1
            if step % log_every == 0:
                _log_grpo(
                    step,
                    num_steps,
                    running_reward,
                    running_count,
                    running_correct,
                    running_loss,
                    running_skipped,
                    t0,
                    llm_enabled=bool(llm_api_key),
                )
                running_reward = running_loss = 0.0
                running_count = running_correct = running_skipped = 0
            continue

        # ── 3. Compute ref log probs (no grad, current model) ─────────
        ref_avg_lps = []
        with inference_mode():
            for comp_toks in completions_tokens:
                if len(comp_toks) == 0:
                    ref_avg_lps.append(0.0)
                    continue
                full = np.array([prompt_tokens + comp_toks], dtype=np.int32)
                _, ref_lp = compute_sequence_log_probs(model, full, prompt_len)
                ref_avg_lps.append(float(ref_lp) / len(comp_toks))

        # ── 4. Policy gradient (grad accumulation over group) ─────────
        optimizer.zero_grad()
        step_loss = 0.0

        for i, comp_toks in enumerate(completions_tokens):
            if len(comp_toks) == 0 or advantages[i] == 0:
                continue

            full = np.array([prompt_tokens + comp_toks], dtype=np.int32)
            _, policy_lp = compute_sequence_log_probs(model, full, prompt_len)

            # Normalize by completion length
            avg_policy_lp = policy_lp / len(comp_toks)

            # GRPO loss: -advantage * log_prob + β * KL(π || π_ref)
            kl = avg_policy_lp - ref_avg_lps[i]
            loss_i = (-advantages[i] * avg_policy_lp + beta * kl) / num_completions

            loss_i.backward()
            step_loss += float(loss_i)

        # ── 5. Clip gradients & step ──────────────────────────────────
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        running_loss += step_loss

        # ── 6. Logging ────────────────────────────────────────────────
        if step % log_every == 0:
            _log_grpo(
                step,
                num_steps,
                running_reward,
                running_count,
                running_correct,
                running_loss,
                running_skipped,
                t0,
                llm_enabled=bool(llm_api_key),
            )
            running_reward = running_loss = 0.0
            running_count = running_correct = running_skipped = 0

        # ── 7. Periodic eval ──────────────────────────────────────────
        if eval_every > 0 and step % eval_every == 0:
            evaluate_gsm8k(
                model,
                examples,
                num_completions=num_completions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                rep_penalty=rep_penalty,
                gen_batch_size=gen_batch_size,
                max_eval=max_eval,
            )

    print(f"\n{'═' * 70}")
    print(f"  GRPO Training complete! ({time.time() - t0:.0f}s)")
    print(f"{'═' * 70}\n")


def _log_grpo(
    step, total, reward, count, correct, loss, skipped, t0, llm_enabled=False
):
    """Print one GRPO log line."""
    import time

    avg_r = reward / max(count, 1)
    elapsed = time.time() - t0
    eta_s = elapsed / step * (total - step) if step > 0 else 0
    eta_m = int(eta_s // 60)
    eta_sec = int(eta_s % 60)

    llm_str = f" llm_avg={avg_r:.3f} |" if llm_enabled else ""

    print(
        f"  Step {step:4d}/{total} | "
        f"loss={loss:.4f} |{llm_str} "
        f"correct={correct}/{count} | "
        f"skip={skipped} | "
        f"ETA {eta_m:02d}:{eta_sec:02d}"
    )


# ═════════════════════════════════════════════════════════════════════════
# EGGROLL — Zeroth-order fine-tuning (arXiv:2511.16652)
# ═════════════════════════════════════════════════════════════════════════


def _get_weight_matrices(model):
    """Return list of (name, param) for all 2-D weight tensors."""
    return [(n, p) for n, p in model.named_parameters() if p.ndim == 2]


def train_eggroll(
    model,
    examples,
    *,
    n_workers=6,
    sigma=0.01,
    alpha=1e-3,
    completions_per_worker=2,
    max_new_tokens=512,
    temperature=0.7,
    n_target_matrices=None,
    antithetic=True,
    rep_penalty=1.0,
    llm_api_key=None,
    num_steps=100,
    log_every=1,
    eval_every=50,
    max_eval=20,
    seed=42,
):
    """EGGROLL zeroth-order fine-tuning (arXiv:2511.16652, eq. 6).

    For each step:
      1. For each worker i = 1..N:
           a. Sample rank-1 perturbation per weight matrix:
                E_i = outer(A_i, B_i),  A ~ N(0,1/d_out), B ~ N(0,1/d_in)
           b. Apply W += σ * E_i to all weight matrices
           c. Generate completion(s), compute reward f_i
           d. Revert W -= σ * E_i
      2. Group-relative advantages: adv_i = (f_i - mean) / std
      3. EGGROLL update (no optimizer, no backprop):
              W += (α / N) * Σ_i  adv_i * E_i

    Memory: only 2 small vectors (A, B) stored per worker × matrix.
    Peak extra GPU mem ≈ one transient outer product (< 20 MB).

    antithetic=True (default): for each direction E_i, evaluate both W+σE_i
    and W-σE_i. Update weight = r(W+σE_i) - r(W-σE_i), a central difference
    estimator with much lower variance. Cost: 2× evals per direction.
    """
    import torch

    tokenizer = Tokenizer()
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    all_weight_matrices = _get_weight_matrices(model)
    n_matrices_total = len(all_weight_matrices)
    if n_target_matrices is None:
        n_target_matrices = n_matrices_total
    n_target_matrices = min(n_target_matrices, n_matrices_total)

    import time

    t0 = time.time()
    running_reward = 0.0
    running_count = 0
    running_skipped = 0

    # Pre-compute per-matrix normalization constants once
    norm = {
        n: 1.0 / (p.shape[0] ** 0.5 * p.shape[1] ** 0.5) for n, p in all_weight_matrices
    }

    print(f"\n{'═' * 70}")
    print(f"  EGGROLL Zeroth-Order Training")
    print(f"  {num_steps} steps, {n_workers} workers/step")
    print(f"  σ={sigma}, α={alpha}, temp={temperature}")
    print(
        f"  {n_target_matrices}/{n_matrices_total} weight matrices perturbed per worker"
    )
    print(
        f"  completions_per_worker={completions_per_worker}, "
        f"max_new_tokens={max_new_tokens}"
    )
    print(f"  antithetic={'yes (W+σE & W-σE per direction)' if antithetic else 'no'}")
    print(f"{'═' * 70}\n")

    for step in range(1, num_steps + 1):
        t_step = time.time()
        ex = random.choice(examples)
        prompt = ex["prompt"]
        gt = ex["ground_truth"]
        q_short = ex["question"][:60] + ("..." if len(ex["question"]) > 60 else "")
        print(f"  Step {step}/{num_steps}  Q: {q_short}")
        print(f"  gt={gt}")

        # Select target matrices for this step (random subset)
        weight_matrices = random.sample(all_weight_matrices, n_target_matrices)

        def _eval_worker(sign):
            """Apply ±σE, generate, revert. Returns reward."""
            for name, p in weight_matrices:
                A, B = eps[name]
                p.data.add_(sign * sigma * torch.outer(A, B))
            comps = generate_completions(
                model,
                tokenizer,
                prompt,
                num_completions=completions_per_worker,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                rep_penalty=rep_penalty,
                gen_batch_size=completions_per_worker,
            )
            rws = compute_rewards(ex["question"], comps, gt, llm_api_key)
            r = sum(rws) / len(rws)
            for name, p in weight_matrices:
                A, B = eps[name]
                p.data.sub_(sign * sigma * torch.outer(A, B))
            return r

        # ── 1. Evaluate N directions (no grad) ───────────────────────
        # all_eps[i] = (A,B) dict for direction i
        # deltas[i]  = r+ - r- (antithetic) or r (standard)
        all_eps = []
        deltas = []
        worker_rewards = []

        with inference_mode():
            for w_idx in range(n_workers):
                t_w = time.time()
                eps = {}
                for name, p in weight_matrices:
                    A = torch.randn(p.shape[0], device=p.device, dtype=p.dtype)
                    B = torch.randn(p.shape[1], device=p.device, dtype=p.dtype)
                    A *= norm[name] ** 0.5
                    B *= norm[name] ** 0.5
                    eps[name] = (A, B)

                if antithetic:
                    print(f"    dir {w_idx+1}/{n_workers}: +σE...", end=" ", flush=True)
                    r_pos = _eval_worker(+1.0)
                    print(
                        f"{'✓' if r_pos > 0 else '✗'}{r_pos:.2f}  -σE...",
                        end=" ",
                        flush=True,
                    )
                    r_neg = _eval_worker(-1.0)
                    print(
                        f"{'✓' if r_neg > 0 else '✗'}{r_neg:.2f}  "
                        f"Δ={r_pos-r_neg:+.2f}  ({time.time()-t_w:.1f}s)"
                    )
                    deltas.append(r_pos - r_neg)
                    worker_rewards.extend([r_pos, r_neg])
                else:
                    print(
                        f"    worker {w_idx+1}/{n_workers}: generating...",
                        end=" ",
                        flush=True,
                    )
                    r = _eval_worker(+1.0)
                    print(
                        f"{'✓' if r > 0 else '✗'} r={r:.2f}  ({time.time()-t_w:.1f}s)"
                    )
                    deltas.append(r)
                    worker_rewards.append(r)

                all_eps.append(eps)

        # ── 2. Advantages / signal ────────────────────────────────────
        if antithetic:
            # deltas[i] = r+ - r-: already a centered estimate, no normalization
            # Normalize by std for stability
            d = np.array(deltas, dtype=np.float32)
            std = d.std() + 1e-8
            advantages = d / std
            print(
                f"  Δs={[f'{x:+.2f}' for x in deltas]}  "
                f"advantages={[f'{a:+.2f}' for a in advantages]}"
            )
        else:
            advantages = compute_advantages(worker_rewards)
            print(
                f"  rewards={[f'{r:.2f}' for r in worker_rewards]}  "
                f"advantages={[f'{a:+.2f}' for a in advantages]}"
            )

        avg_rew = sum(worker_rewards) / len(worker_rewards)
        running_reward += avg_rew
        running_count += 1

        # ── 3. EGGROLL weight update (no backprop) ────────────────────
        if np.any(np.array(advantages) != 0):
            print(f"  updating weights...", end=" ", flush=True)
            scale = alpha / n_workers
            for i in range(n_workers):
                if advantages[i] == 0.0:
                    continue
                w = scale * float(advantages[i])
                for name, p in weight_matrices:
                    A, B = all_eps[i][name]
                    p.data.add_(w * torch.outer(A, B))
            print("done")
        else:
            print("  skipped (no Δ signal)")
            running_skipped += 1

        print(f"  step time: {time.time()-t_step:.1f}s\n")

        # ── 4. Logging ────────────────────────────────────────────────
        if step % log_every == 0:
            avg_r = running_reward / max(running_count, 1)
            elapsed = time.time() - t0
            eta_s = elapsed / step * (num_steps - step) if step > 0 else 0
            eta_m, eta_sec = int(eta_s // 60), int(eta_s % 60)
            print(
                f"  Step {step:4d}/{num_steps} | "
                f"reward={avg_r:.3f} ({int(running_reward)}/{running_count}) | "
                f"skip={running_skipped} | "
                f"ETA {eta_m:02d}:{eta_sec:02d}"
            )
            running_reward = running_count = 0
            running_skipped = 0

        # ── 5. Periodic eval ──────────────────────────────────────────
        if eval_every > 0 and step % eval_every == 0:
            evaluate_gsm8k(
                model,
                examples,
                num_completions=4,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                rep_penalty=rep_penalty,
                gen_batch_size=4,
                max_eval=max_eval,
            )

    print(f"\n{'═' * 70}")
    print(f"  EGGROLL complete! ({time.time() - t0:.0f}s)")
    print(f"{'═' * 70}\n")


# ═════════════════════════════════════════════════════════════════════════
# Quick forward pass sanity check
# ═════════════════════════════════════════════════════════════════════════


def sanity_check(model):
    """Run a quick forward pass to verify the model works."""
    tokenizer = Tokenizer()
    prompt = "<|im_start|>user\nWhat is 2+2?\n<|im_end|><|im_start|>assistant\n<|think_start|>"
    tokens = tokenizer([prompt])[0]
    input_ids = np.array([tokens], dtype=np.int32)

    print("Sanity check: forward pass...")
    with inference_mode():
        logits = model(input_ids, training=False)
    logits_np = to_numpy(logits)

    # Show top-5 predictions for last token
    last_logits = logits_np[0, -1]
    top5_idx = np.argsort(last_logits)[-5:][::-1]
    top5_vals = last_logits[top5_idx]

    print(f"  Input: {len(tokens)} tokens")
    print(f"  Output logits shape: {logits_np.shape}")
    print(f"  Top-5 next tokens:")
    for idx, val in zip(top5_idx, top5_vals):
        tok_str = tokenizer.decode([int(idx)])
        print(f"    {idx:6d} ({val:8.3f}) = '{tok_str}'")

    # Quick greedy decode of 10 tokens
    print(f"\n  Greedy decode (10 tokens):", end=" ")
    generated = list(tokens)
    current = input_ids
    with inference_mode():
        for _ in range(10):
            logits = model(current, training=False)
            next_id = int(np.argmax(to_numpy(logits)[0, -1]))
            generated.append(next_id)
            current = np.array([generated], dtype=np.int32)
    decoded = tokenizer.decode(generated[len(tokens) :])
    print(decoded)
    print()


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="GRPO training for SeqCond (Keras 3)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Only print model summary, don't train",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Limit number of GSM8K examples (default: all)",
    )
    parser.add_argument(
        "--num_completions",
        type=int,
        default=4,
        help="Number of completions per prompt (GRPO group size)",
    )
    parser.add_argument(
        "--max_eval",
        type=int,
        default=50,
        help="Number of examples to evaluate",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max tokens per completion",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--rep_penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (1.0=off, 1.1 recommended)",
    )
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        default=4,
        help="Parallel completions per generation call (fits in VRAM)",
    )

    # Common training args
    grp = parser.add_argument_group("Training")
    grp.add_argument(
        "--algo",
        type=str,
        default="grpo",
        choices=["grpo", "eggroll"],
        help="Training algorithm",
    )
    grp.add_argument("--num_steps", type=int, default=100, help="Training steps")
    grp.add_argument("--log_every", type=int, default=1)
    grp.add_argument(
        "--eval_every", type=int, default=50, help="Eval during training (0=off)"
    )
    grp.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip baseline eval before training",
    )
    grp.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key for LLM reasoning reward (gpt-4.1-mini). "
        "Falls back to OPENAI_API_KEY env var. Disabled if not set.",
    )
    grp.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save fine-tuned weights (.pkl). "
        "Default: checkpoints/<base>_<algo>.pkl",
    )

    # GRPO-specific args
    grpo_grp = parser.add_argument_group("GRPO")
    grpo_grp.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    grpo_grp.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adamw"],
        help="Optimizer (sgd recommended for 8GB GPU)",
    )
    grpo_grp.add_argument("--weight_decay", type=float, default=0.01)
    grpo_grp.add_argument(
        "--beta", type=float, default=0.1, help="KL penalty coefficient"
    )
    grpo_grp.add_argument("--max_grad_norm", type=float, default=1.0)
    grpo_grp.add_argument(
        "--train_layers",
        type=int,
        default=2,
        help="Random blocks to unfreeze per step (GRPO)",
    )

    # EGGROLL-specific args
    egg_grp = parser.add_argument_group("EGGROLL")
    egg_grp.add_argument(
        "--n_workers", type=int, default=10, help="Workers (perturbations) per step"
    )
    egg_grp.add_argument(
        "--sigma", type=float, default=0.01, help="Perturbation magnitude"
    )
    egg_grp.add_argument("--alpha", type=float, default=1e-3, help="Update step size")
    egg_grp.add_argument(
        "--completions_per_worker",
        type=int,
        default=2,
        help="Completions evaluated per worker (batched within each worker)",
    )
    egg_grp.add_argument(
        "--n_target_matrices",
        type=int,
        default=None,
        help="Matrices to perturb per worker (default: all). Smaller = faster, e.g. 20-50",
    )
    egg_grp.add_argument(
        "--no_antithetic",
        action="store_true",
        help="Disable antithetic sampling (default: on)",
    )

    args = parser.parse_args()

    # Load model
    model, config = load_model(args.checkpoint)
    print_model_summary(model, config)

    # Sanity check
    sanity_check(model)

    # Load GSM8K
    examples = load_gsm8k(split="train", seed=42, max_examples=args.max_examples)
    print_dataset_summary(examples)

    if args.summary_only:
        print("Done (summary only).")
        return

    # Baseline evaluation
    if not args.skip_baseline:
        evaluate_gsm8k(
            model,
            examples,
            num_completions=args.num_completions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            rep_penalty=args.rep_penalty,
            gen_batch_size=args.gen_batch_size,
            max_eval=args.max_eval,
        )

    # Resolve LLM API key (arg > env var)
    llm_api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY") or None
    if llm_api_key:
        print(f"  LLM reward: enabled (gpt-4.1-mini)")
    else:
        print(f"  LLM reward: disabled (pass --openai_api_key or set OPENAI_API_KEY)")

    # Determine save path
    if args.save is None:
        base = os.path.splitext(os.path.basename(args.checkpoint))[0]
        save_path = os.path.join("checkpoints", f"{base}_{args.algo}.pt")
    else:
        save_path = args.save

    if args.algo == "eggroll":
        train_eggroll(
            model,
            examples,
            n_workers=args.n_workers,
            sigma=args.sigma,
            alpha=args.alpha,
            completions_per_worker=args.completions_per_worker,
            n_target_matrices=args.n_target_matrices,
            antithetic=not args.no_antithetic,
            rep_penalty=args.rep_penalty,
            llm_api_key=llm_api_key,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_steps=args.num_steps,
            log_every=args.log_every,
            eval_every=args.eval_every,
            max_eval=args.max_eval,
        )
    else:
        train_grpo(
            model,
            examples,
            num_completions=args.num_completions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            gen_batch_size=args.gen_batch_size,
            beta=args.beta,
            lr=args.lr,
            optimizer_name=args.optimizer,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            train_layers=args.train_layers,
            rep_penalty=args.rep_penalty,
            llm_api_key=llm_api_key,
            num_steps=args.num_steps,
            log_every=args.log_every,
            eval_every=args.eval_every,
            max_eval=args.max_eval,
        )

    # Save fine-tuned weights
    try:
        if save_path.endswith(".pt"):
            # Save .pkl intermediary, then convert to .pt with correct key mapping
            pkl_path = save_path[:-3] + ".pkl"
            save_keras_checkpoint(model, config, pkl_path)
            keras_pkl_to_torch_pt(pkl_path, save_path)
            print(f"  Weights saved to: {save_path}  (intermediary: {pkl_path})")
        else:
            save_keras_checkpoint(model, config, save_path)
            print(f"  Weights saved to: {save_path}")
    except OSError as e:
        print(f"  WARNING: could not save ({e})")


if __name__ == "__main__":
    main()

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
    keras_pkl_to_torch_pt,
    load_torch_checkpoint,
    save_keras_checkpoint,
)
from seqcond.dataset import Tokenizer


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
) -> List[float]:
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

    rewards = []
    for c in completions:
        correct = 1.0 if check_answer(c, ground_truth) else 0.0
        fmt = FORMAT_BONUS if "<|think_end|>" in c else 0.0
        rewards.append(correct + fmt)

    if not api_key:
        return rewards
    bonuses = _llm_bonuses(question, completions, api_key)
    return [r + bon for r, bon in zip(rewards, bonuses)]


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
    logits = model(input_ids)
    log_probs = ops.log_softmax(logits, axis=-1)
    shift = log_probs[0, prompt_len - 1 : -1, :]
    targets = ops.expand_dims(ops.cast(input_ids[0, prompt_len:], "int32"), -1)
    per_tok = ops.squeeze(ops.take_along_axis(shift, targets, axis=-1), axis=-1)
    return ops.sum(per_tok)


def _compute_advantages(rewards):
    """Group-relative advantages: (r - mean) / (std + eps)."""
    r = np.array(rewards, dtype=np.float32)
    std = r.std()
    return np.zeros_like(r) if std < 1e-8 else (r - r.mean()) / std


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

    # Reference log probs (no grad — current model snapshot before this step)
    ref_lps = []
    with inference_mode():
        for comp in completions_tokens:
            if len(comp) == 0:
                ref_lps.append(0.0)
                continue
            ids = np.array([prompt_tokens + comp], dtype=np.int32)
            ref_lps.append(float(_seq_log_prob(model, ids, prompt_len)) / len(comp))

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
        loss_i.backward()
        total_loss += float(loss_i)

    return total_loss


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────


def load_model(checkpoint_path: str):
    """Load Keras SeqCond model from a PyTorch .pt checkpoint."""
    config, state_dict = load_torch_checkpoint(checkpoint_path)
    model = build_keras_model(config)
    convert_weights(config, state_dict, model)
    n_params = model.count_params()
    print(
        f"Loaded {checkpoint_path}  ({n_params:,} params, backend={keras.backend.backend()})"
    )
    return model, config


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(
    model,
    examples,
    max_examples=50,
    num_completions=4,
    max_new_tokens=512,
    temperature=0.5,
    rep_penalty=1.1,
    gen_batch_size=4,
):
    """Evaluate pass@k accuracy on the first max_examples problems."""
    tokenizer = Tokenizer()
    examples = examples[:max_examples]
    correct = 0
    t0 = time.time()
    for i, ex in enumerate(examples):
        comps = generate_completions(
            model,
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
    """Save checkpoint locally and upload to GCS."""
    import subprocess

    GCS_BUCKET = "gs://telekinesis-43/checkpoints"

    # Save locally
    pkl_path = save_path[:-3] + ".pkl" if save_path.endswith(".pt") else save_path
    save_keras_checkpoint(model, config, pkl_path)
    if save_path.endswith(".pt"):
        keras_pkl_to_torch_pt(pkl_path, save_path)

    # Upload to GCS
    try:
        filename = os.path.basename(save_path)
        base, ext = os.path.splitext(filename)
        gcs_filename = f"{base}_step{step}{ext}"
        gcs_path = f"{GCS_BUCKET}/{gcs_filename}"

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
            print(f"✗ ({result.stderr.strip()[:50]})")
    except Exception as e:
        print(f"✗ ({str(e)[:50]})")


def train_grpo(
    model,
    config,
    examples,
    *,
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
    max_eval: int = 50,
    save_gcp_every: int = 0,
    save_path: str = None,
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

    print(
        f"\n── GRPO  {num_steps} steps  G={num_completions}  "
        f"lr={lr}  β={beta}  train_layers={train_layers}/{n_blocks}  "
        f"warmup={warmup_steps} ──\n"
    )

    t0 = time.time()
    run_reward = run_loss = run_correct = run_skipped = 0.0

    for step in range(1, num_steps + 1):
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

        # Generate
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

        # Score
        rewards = score_output(ex["question"], texts_f, ex["ground_truth"], llm_api_key)
        binary = [1.0 if check_answer(t, ex["ground_truth"]) else 0.0 for t in texts_f]
        if llm_api_key:
            print(
                f"  llm={[f'{r:.2f}' for r in rewards]}  binary={[int(b) for b in binary]}"
            )

        advantages = _compute_advantages(rewards)
        run_reward += sum(rewards)
        run_correct += int(sum(binary))

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

            # GRPO update
            optimizer.zero_grad()
            loss = grpo_step(model, prompt_tokens, ids_f, advantages, beta=beta)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            run_loss += loss

        # Log
        if step % log_every == 0:
            n = num_completions * log_every
            avg_r = run_reward / n
            elapsed = time.time() - t0
            eta = elapsed / step * (num_steps - step)
            llm_str = f" llm_avg={avg_r:.3f} |" if llm_api_key else ""
            print(
                f"  Step {step:4d}/{num_steps} | loss={run_loss:.4f} |{llm_str} "
                f"correct={int(run_correct)}/{n} | skip={int(run_skipped)} | "
                f"ETA {int(eta//60):02d}:{int(eta%60):02d}"
            )
            run_reward = run_loss = run_correct = run_skipped = 0.0

        # Periodic eval
        if eval_every > 0 and step % eval_every == 0:
            evaluate(
                model,
                examples,
                max_examples=max_eval,
                num_completions=4,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                rep_penalty=rep_penalty,
                gen_batch_size=gen_batch_size,
            )

        # Periodic GCP backup
        if save_gcp_every > 0 and step % save_gcp_every == 0 and save_path:
            _save_and_upload_gcp(model, config, save_path, step)

    print(f"\n── GRPO complete ({time.time()-t0:.0f}s) ──\n")


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
    p.add_argument("--num_steps", type=int, default=250)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--train_layers", type=int, default=3)
    p.add_argument("--optimizer", default="adamw", choices=["sgd", "adamw"])
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--max_eval", type=int, default=50)
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

    args = p.parse_args()

    model, config = load_model(args.checkpoint)
    examples = load_gsm8k(split="train", seed=42, max_examples=args.max_examples)
    print(f"Dataset: {len(examples)} GSM8K training examples")

    if not args.skip_baseline:
        print("\n── Baseline eval ──")
        evaluate(
            model,
            examples,
            max_examples=args.max_eval,
            num_completions=4,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
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
        save_gcp_every=args.save_gcp_every,
        save_path=save_path,
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

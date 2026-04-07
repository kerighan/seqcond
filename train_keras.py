"""
Single-GPU Keras 3 (torch backend) training script for SeqCond models.

Simplified from train_jax.py — no multi-host, no FSDP, no pmap.
Runs on a single GPU with unified memory (e.g. Apple M-series 128 GB).

Usage:
    KERAS_BACKEND=torch python train_keras.py --load-checkpoint checkpoints/seqcond_lin5.pt --size large --maxlen 4096 --batch-size 2 --lr 3e-4
    KERAS_BACKEND=torch python train_keras.py --size small --total-steps 100000 --batch-size 4
"""

import os
import sys
import time
import math
import argparse
import itertools

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import keras
from keras import ops

from seqcond.config import Config, ModelConfig, TrainingConfig
from seqcond.dataset import tokenizer, data_generator
from seqcond.keras3.train import (
    MaskedSparseCategoricalCrossentropy,
    MaskedAccuracy,
    Perplexity,
    WarmupCosineDecay,
    save_checkpoint,
    load_checkpoint,
    create_model_from_config,
)
from convert_torch_to_keras import (
    load_torch_checkpoint,
    build_keras_model,
    convert_weights,
    save_keras_checkpoint,
    keras_pkl_to_torch_pt,
)


# ═══════════════════════════════════════════════════════════════════════
# Metrics accumulator (simple, no JAX)
# ═══════════════════════════════════════════════════════════════════════

class MetricsAccumulator:
    """Accumulate loss, accuracy, perplexity over multiple steps."""

    def __init__(self):
        self.total_loss = 0.0
        self.total_correct = 0
        self.total_tokens = 0
        self.num_steps = 0

    def update(self, loss_val, logits, y_true, ignore_class=0):
        """Update with one batch's results."""
        y_np = np.array(y_true).reshape(-1).astype(np.int32)
        logits_np = np.array(logits).reshape(-1, logits.shape[-1])
        mask = y_np != ignore_class

        self.total_loss += float(loss_val) * int(mask.sum())
        preds = logits_np.argmax(axis=-1)
        self.total_correct += int(((preds == y_np) & mask).sum())
        self.total_tokens += int(mask.sum())
        self.num_steps += 1

    def result(self):
        if self.total_tokens == 0:
            return {"loss": 0.0, "acc": 0.0, "ppx": 1.0}
        avg_loss = self.total_loss / self.total_tokens
        acc = self.total_correct / self.total_tokens
        ppx = min(math.exp(avg_loss), 10000.0)
        return {"loss": avg_loss, "acc": acc, "ppx": ppx}

    def reset(self):
        self.total_loss = 0.0
        self.total_correct = 0
        self.total_tokens = 0
        self.num_steps = 0


# ═══════════════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_sample(model, tokenizer_obj, maxlen, max_new_tokens=128, temperature=0.8):
    """Generate a short text sample for monitoring training progress."""
    import torch

    prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<|think_start|>"
    token_ids = tokenizer_obj.encode(prompt)

    model_training = model.training if hasattr(model, 'training') else False

    generated = list(token_ids)
    for _ in range(max_new_tokens):
        # Prepare input: take last maxlen tokens
        inp = generated[-maxlen:]
        inp_np = np.array([inp], dtype=np.int32)
        with torch.no_grad():
            logits = model(inp_np, training=False)
        logits_np = np.array(logits[0, -1, :]).astype(np.float64)

        # Temperature sampling
        if temperature > 0:
            logits_np /= temperature
            logits_np -= logits_np.max()
            probs = np.exp(logits_np)
            probs /= probs.sum()
            next_token = int(np.random.choice(len(probs), p=probs))
        else:
            next_token = int(np.argmax(logits_np))

        generated.append(next_token)

        # Stop on EOS
        eos_id = tokenizer_obj.encode("<|im_end|>")[0]
        if next_token == eos_id:
            break

    text = tokenizer_obj.decode(generated)
    return text


# ═══════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════

def train_loop(
    model,
    config,
    *,
    total_steps=100000,
    batch_size=1,
    maxlen=768,
    base_lr=3e-4,
    alpha=1e-5,
    warmup_steps=2000,
    weight_decay=1e-2,
    clipnorm=1.0,
    beta_1=0.9,
    beta_2=0.99,
    grad_accum_steps=1,
    log_every=100,
    save_every=10000,
    generate_every=5000,
    checkpoint_dir="checkpoints",
    model_name="seqcond",
    wandb_project=None,
    extra_data=None,
    start_step=0,
):
    """Manual training loop with gradient accumulation, logging, checkpointing."""
    import torch

    # ── Mixed precision ──────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Grad accumulation: {grad_accum_steps}")
    print(f"Effective batch size: {batch_size * grad_accum_steps}")

    # ── Loss function ────────────────────────────────────────────────
    loss_fn = MaskedSparseCategoricalCrossentropy(ignore_class=0)

    # ── LR schedule ──────────────────────────────────────────────────
    lr_schedule = WarmupCosineDecay(
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        alpha=alpha,
    )

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=1e-7,
        weight_decay=weight_decay,
        clipnorm=clipnorm,
    )

    # Build optimizer state
    optimizer.build(model.trainable_variables)

    # ── WandB ────────────────────────────────────────────────────────
    _wandb = None
    if wandb_project:
        try:
            import wandb
            wandb.init(project=wandb_project, name=model_name, config=config.to_dict())
            _wandb = wandb
            print(f"WandB: logging to project '{wandb_project}'")
        except ImportError:
            print("Warning: wandb not installed, skipping.")

    # ── Data ─────────────────────────────────────────────────────────
    micro_steps = total_steps * grad_accum_steps
    data_iter = iter(data_generator(
        batch_size=batch_size,
        max_steps=micro_steps,
        maxlen=maxlen,
        log_every_n_steps=10000,
        extra_data=extra_data,
    ))

    # ── Training state ───────────────────────────────────────────────
    metrics = MetricsAccumulator()
    macro_step = start_step
    accum_count = 0
    start_time = time.time()
    last_log_time = start_time
    tokens_seen = 0
    last_tokens_seen = 0

    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"Training: {model_name} | {total_steps} steps | bs={batch_size} | maxlen={maxlen} | lr={base_lr}")
    print(f"{'='*70}\n")

    for micro_step in itertools.count(start=start_step * grad_accum_steps + 1):
        if micro_step > micro_steps:
            break

        # ── Get batch ────────────────────────────────────────────────
        try:
            x_batch, y_batch, real_tokens = next(data_iter)
        except StopIteration:
            print("Data exhausted.")
            break

        tokens_seen += int(real_tokens)

        # ── Forward + backward ───────────────────────────────────────
        x_t = torch.tensor(x_batch, dtype=torch.int32, device=device)
        y_t = torch.tensor(y_batch, dtype=torch.int32, device=device)

        # Forward pass
        logits = model(x_t, training=True)
        loss = loss_fn(y_t, logits)
        scaled_loss = loss / grad_accum_steps

        # Backward
        scaled_loss.backward()

        accum_count += 1

        if accum_count >= grad_accum_steps:
            # Compute grad norm before clipping (for logging)
            grad_norm = 0.0
            for v in model.trainable_variables:
                if hasattr(v, 'value') and hasattr(v.value, 'grad') and v.value.grad is not None:
                    grad_norm += float(v.value.grad.detach().norm().item() ** 2)
                elif hasattr(v, 'grad') and v.grad is not None:
                    grad_norm += float(v.grad.detach().norm().item() ** 2)
            grad_norm = grad_norm ** 0.5

            # Optimizer step
            optimizer.apply(
                [v.value.grad if hasattr(v, 'value') and hasattr(v.value, 'grad') and v.value.grad is not None
                 else (v.grad if hasattr(v, 'grad') and v.grad is not None
                       else torch.zeros_like(v.value if hasattr(v, 'value') else v))
                 for v in model.trainable_variables],
                model.trainable_variables,
            )

            # Zero gradients
            for v in model.trainable_variables:
                if hasattr(v, 'value') and hasattr(v.value, 'grad') and v.value.grad is not None:
                    v.value.grad = None
                elif hasattr(v, 'grad') and v.grad is not None:
                    v.grad = None

            accum_count = 0
            macro_step += 1

            # ── Update metrics ───────────────────────────────────────
            with torch.no_grad():
                metrics.update(
                    float(loss.detach()),
                    logits.detach().cpu().numpy(),
                    y_batch,
                )

            # ── Logging ──────────────────────────────────────────────
            if macro_step > 0 and macro_step % log_every == 0:
                current_time = time.time()
                elapsed = current_time - last_log_time
                tokens_delta = tokens_seen - last_tokens_seen
                last_tokens_seen = tokens_seen
                tokens_per_sec = tokens_delta / elapsed if elapsed > 0 else 0

                steps_remaining = total_steps - macro_step
                total_elapsed = current_time - start_time
                avg_per_step = total_elapsed / max(macro_step - start_step, 1)
                eta_s = steps_remaining * avg_per_step
                eta_h = int(eta_s // 3600)
                eta_m = int((eta_s % 3600) // 60)

                results = metrics.result()
                current_lr = float(lr_schedule(macro_step))

                print(
                    f"Step {macro_step:6d}/{total_steps} | "
                    f"loss: {results['loss']:.4f} | "
                    f"acc: {results['acc']:.4f} | "
                    f"ppx: {results['ppx']:.2f} | "
                    f"grad_norm: {grad_norm:.2f} | "
                    f"lr: {current_lr:.2e} | "
                    f"{tokens_per_sec:,.0f} tok/s | "
                    f"{tokens_seen:,} tokens | "
                    f"ETA: {eta_h:02d}:{eta_m:02d}"
                )

                if _wandb is not None:
                    _wandb.log({
                        **{k: float(v) for k, v in results.items()},
                        "tokens_per_sec": tokens_per_sec,
                        "tokens_seen": tokens_seen,
                        "lr": current_lr,
                        "grad_norm": grad_norm,
                    }, step=macro_step)

                metrics.reset()
                last_log_time = current_time

            # ── Generation sample ────────────────────────────────────
            if generate_every > 0 and macro_step > 0 and macro_step % generate_every == 0:
                print(f"\n--- Generation at step {macro_step} ---")
                try:
                    txt = generate_sample(model, tokenizer, maxlen)
                    # Print first 500 chars
                    print(txt[:500])
                    if len(txt) > 500:
                        print(f"... ({len(txt)} chars total)")
                except Exception as e:
                    print(f"Generation failed: {e}")
                print("---\n")

            # ── Checkpoint ───────────────────────────────────────────
            if save_every > 0 and macro_step > 0 and macro_step % save_every == 0:
                _save(model, config, checkpoint_dir, model_name, macro_step)

    # ── Final checkpoint ─────────────────────────────────────────────
    print("\nTraining complete!")
    _save(model, config, checkpoint_dir, model_name, macro_step, final=True)

    if _wandb is not None:
        _wandb.finish()


def _save(model, config, checkpoint_dir, model_name, step, final=False):
    """Save checkpoint in both .pkl and .pt formats."""
    if final:
        pkl_path = f"{checkpoint_dir}/{model_name}.pkl"
        pt_path = f"{checkpoint_dir}/{model_name}.pt"
    else:
        pkl_path = f"{checkpoint_dir}/{model_name}_step{step}.pkl"
        pt_path = f"{checkpoint_dir}/{model_name}_step{step}.pt"

    config_dict = config.to_dict() if hasattr(config, "to_dict") else config
    save_keras_checkpoint(model, config_dict, pkl_path)
    try:
        keras_pkl_to_torch_pt(pkl_path, pt_path)
    except Exception as e:
        print(f"  Warning: .pt conversion failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Train SeqCond model (single-GPU, Keras 3 / torch backend)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--size", choices=["small", "medium", "large", "xlarge"], default="small")
    p.add_argument("--load-checkpoint", type=str, default=None, help="Load weights from .pt or .pkl checkpoint")
    p.add_argument("--resume-step", type=int, default=None, help="Step to resume from (affects LR schedule and data)")

    # Model overrides
    p.add_argument("--maxlen", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--d-ff", type=int, default=None)
    p.add_argument("--num-thetas", type=int, default=None)
    p.add_argument("--ratio", type=int, default=None, dest="seqcond_ratio")
    p.add_argument("--expand", type=float, default=None, dest="expand_factor")
    p.add_argument("--gate-expand", type=int, default=None, dest="out_expand_factor")
    p.add_argument("--chunk", type=int, default=0, dest="chunk_size")
    p.add_argument("--seqcond-heads", type=int, default=None)
    p.add_argument("--seqcond-query-heads", type=int, default=None, dest="num_query_heads")
    p.add_argument("--anchor", type=int, default=None, dest="num_anchor_heads")

    # Training
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--total-steps", type=int, default=100000)
    p.add_argument("--lr", type=float, default=3e-4, dest="base_lr")
    p.add_argument("--alpha", type=float, default=1e-5, help="Min LR at end of cosine decay")
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--clipnorm", type=float, default=1.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)

    # Logging / saving
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=10000)
    p.add_argument("--generate-every", type=int, default=5000)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--wandb-project", type=str, default=None)

    # Data
    p.add_argument("--extra-data", type=str, default=None, help="Path to JSONL to interleave with synth data")
    p.add_argument("--seed", type=int, default=42)

    # Precision
    p.add_argument("--mixed-precision", choices=["bfloat16", "float16", "float32"], default="bfloat16")

    return p.parse_args()


def main():
    args = parse_args()

    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Mixed precision ──────────────────────────────────────────────
    if args.mixed_precision == "bfloat16":
        keras.mixed_precision.set_global_policy("mixed_bfloat16")
        print("Mixed precision: bfloat16")
    elif args.mixed_precision == "float16":
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: float16")
    else:
        print("Precision: float32")

    # ── Build config ─────────────────────────────────────────────────
    model_factory = getattr(ModelConfig, args.size)
    model_overrides = {}
    for field in ["num_layers", "d_model", "d_ff", "num_thetas", "seqcond_heads",
                  "num_query_heads", "seqcond_ratio", "expand_factor", "out_expand_factor",
                  "chunk_size", "num_anchor_heads"]:
        val = getattr(args, field, None)
        if val is not None:
            model_overrides[field] = val
    model_overrides["maxlen"] = args.maxlen

    # ── Load checkpoint or create from scratch ───────────────────────
    start_step = args.resume_step or 0

    if args.load_checkpoint and args.load_checkpoint.endswith(".pt"):
        print(f"Loading PyTorch checkpoint: {args.load_checkpoint}")
        ckpt_config, state_dict = load_torch_checkpoint(args.load_checkpoint)

        # Use checkpoint config but allow CLI overrides for maxlen
        for key in model_overrides:
            if key in ckpt_config:
                ckpt_config[key] = model_overrides[key]
            elif isinstance(ckpt_config, dict):
                ckpt_config[key] = model_overrides[key]

        model = build_keras_model(ckpt_config)
        n_assigned = convert_weights(ckpt_config, state_dict, model)
        print(f"  Loaded {n_assigned}/{len(model.weights)} weights")

        # Build Config object from checkpoint config
        if isinstance(ckpt_config, dict):
            model_config = ModelConfig(**{
                k: v for k, v in ckpt_config.items()
                if k in {f.name for f in __import__('dataclasses').fields(ModelConfig)}
            })
        else:
            model_config = ckpt_config
        config = Config(model=model_config, training=TrainingConfig())

    elif args.load_checkpoint and args.load_checkpoint.endswith(".pkl"):
        print(f"Loading Keras checkpoint: {args.load_checkpoint}")
        weights, ckpt_config_dict, ckpt_step = load_checkpoint(args.load_checkpoint)

        model_config = model_factory(**model_overrides)
        config = Config(model=model_config, training=TrainingConfig())
        model = create_model_from_config(model_config)

        # Build model
        dummy = np.ones((1, model_config.maxlen), dtype=np.int32)
        _ = model(dummy, training=False)

        # Load weights
        loaded = 0
        for w in model.weights:
            if w.path in weights:
                w.assign(weights[w.path])
                loaded += 1
        print(f"  Loaded {loaded}/{len(model.weights)} weights (step {ckpt_step})")
        if ckpt_step and args.resume_step is None:
            start_step = ckpt_step

    else:
        print(f"Creating model from scratch: size={args.size}")
        model_config = model_factory(**model_overrides)
        config = Config(model=model_config, training=TrainingConfig())
        model = create_model_from_config(model_config)

        # Build model
        dummy = np.ones((1, model_config.maxlen), dtype=np.int32)
        _ = model(dummy, training=False)

    # ── Print model info ─────────────────────────────────────────────
    num_params = model.count_params()
    model_name = config.name
    print(f"\nModel: {model_name}")
    print(f"Parameters: {num_params:,}")
    print(f"Maxlen: {config.model.maxlen}")
    print(f"Backend: {keras.backend.backend()}")

    # ── Train ────────────────────────────────────────────────────────
    train_loop(
        model,
        config,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        maxlen=config.model.maxlen,
        base_lr=args.base_lr,
        alpha=args.alpha,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        clipnorm=args.clipnorm,
        beta_1=args.beta1,
        beta_2=args.beta2,
        grad_accum_steps=args.grad_accum_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        generate_every=args.generate_every,
        checkpoint_dir=args.checkpoint_dir,
        model_name=model_name,
        wandb_project=args.wandb_project,
        extra_data=args.extra_data,
        start_step=start_step,
    )


if __name__ == "__main__":
    main()

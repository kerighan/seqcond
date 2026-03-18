"""
Train the Depth-SeqCond layer on top of a frozen base model.

Usage:
    KERAS_BACKEND=jax python train_depth.py --checkpoint checkpoints/seqcond_lin5.pt
    KERAS_BACKEND=jax python train_depth.py --checkpoint checkpoints/seqcond_lin5.pt --use-outputs
"""

import os
import sys
import time
import argparse
import numpy as np

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
from keras import ops
from contextlib import contextmanager


@contextmanager
def inference_mode():
    """Disable gradient tracking for inference (torch) or no-op (jax)."""
    if keras.backend.backend() == "torch":
        import torch

        with torch.no_grad():
            yield
    else:
        yield


from convert_torch_to_keras import (
    load_torch_checkpoint,
    build_keras_model,
    convert_weights,
)
from seqcond.keras.depth_model import DepthSeqCondModel
from seqcond.dataset import Tokenizer, DataLoader, iterate_synth


# ═══════════════════════════════════════════════════════════════════
# PPL callback
# ═══════════════════════════════════════════════════════════════════


class PPLLogger(keras.callbacks.Callback):
    """Print loss / PPL at end of each Keras 'epoch' (= log_every steps)."""

    def __init__(self, log_every, total_steps):
        super().__init__()
        self.log_every = log_every
        self.total_steps = total_steps
        self.start_time = None
        self.epoch_start = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        step = (epoch + 1) * self.log_every
        loss = logs.get("loss", 0.0)
        ppl = float(np.exp(min(loss, 20.0)))  # clip to avoid overflow

        elapsed = time.time() - self.start_time
        epoch_time = time.time() - self.epoch_start
        steps_per_sec = self.log_every / epoch_time if epoch_time > 0 else 0

        remaining = self.total_steps - step
        eta_sec = remaining / steps_per_sec if steps_per_sec > 0 else 0
        eta_h, eta_m = int(eta_sec // 3600), int((eta_sec % 3600) // 60)

        gate_val = None
        for w in self.model.weights:
            if "depth_gate" in w.path:
                gate_val = float(1.0 / (1.0 + np.exp(-np.array(w))))
                break

        gate_str = f"  gate={gate_val:.4f}" if gate_val is not None else ""
        print(
            f"Step {step:6d}/{self.total_steps} | "
            f"loss={loss:.4f} | ppl={ppl:.2f}{gate_str} | "
            f"{steps_per_sec:.1f} steps/s | "
            f"ETA {eta_h:02d}:{eta_m:02d}"
        )


# ═══════════════════════════════════════════════════════════════════
# Data generator
# ═══════════════════════════════════════════════════════════════════


def _iterate_synth_wrapper(maxlen=1024, **kwargs):
    """Thin wrapper: iterate_synth doesn't accept maxlen, DataLoader injects it."""
    kwargs.pop("maxlen", None)
    return iterate_synth(**kwargs)


def make_data_generator(batch_size, maxlen, total_steps):
    """Wrap DataLoader into a generator that yields (x, y) forever."""
    tok = Tokenizer()
    loader = DataLoader(
        batch_size=batch_size,
        max_steps=total_steps,
        maxlen=maxlen,
        tok=tok,
        iterator_fn=_iterate_synth_wrapper,
    )
    for X, y in loader:
        yield np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Train Depth-SeqCond layer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Base model checkpoint")
    parser.add_argument(
        "--use-outputs",
        action="store_true",
        help="Use raw layer outputs instead of deltas for depth sequence",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--maxlen", type=int, default=512)
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip baseline PPL eval"
    )
    parser.add_argument(
        "--baseline-maxlen",
        type=int,
        default=512,
        help="Maxlen for baseline eval (shorter to avoid OOM)",
    )

    # Depth block hyperparameters
    parser.add_argument("--depth-heads", type=int, default=16)
    parser.add_argument("--depth-query-heads", type=int, default=16)
    parser.add_argument("--depth-thetas", type=int, default=2)
    parser.add_argument("--depth-expand", type=float, default=1.0)
    parser.add_argument("--depth-gate-expand", type=int, default=2)
    parser.add_argument("--depth-conv-kernel", type=int, default=4)
    parser.add_argument("--depth-dropout", type=float, default=0.0)
    parser.add_argument("--depth-gate-init", type=float, default=0.0)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Temporal chunk size for depth block (avoid OOM)",
    )

    args = parser.parse_args()
    use_deltas = not args.use_outputs

    # ── 1. Load base model ────────────────────────────────────────
    print(f"{'=' * 70}")
    print(f"  Depth-SeqCond Training")
    print(f"  Backend: {keras.backend.backend()}")
    print(f"  Mode: {'deltas (o[i+1] - o[i])' if use_deltas else 'raw outputs (o[i])'}")
    print(f"{'=' * 70}")

    print(f"\nLoading base model from {args.checkpoint}...")
    config, state_dict = load_torch_checkpoint(args.checkpoint)
    base_model = build_keras_model(config)
    convert_weights(config, state_dict, base_model)

    # ── 2. Create depth model ─────────────────────────────────────
    print("Creating DepthSeqCondModel...")
    model = DepthSeqCondModel(
        base_model,
        use_deltas=use_deltas,
        depth_num_heads=args.depth_heads,
        depth_num_query_heads=args.depth_query_heads,
        depth_num_thetas=args.depth_thetas,
        depth_expand_factor=args.depth_expand,
        depth_out_expand_factor=args.depth_gate_expand,
        depth_conv_kernel_size=args.depth_conv_kernel,
        depth_dropout=args.depth_dropout,
        depth_scale_init=args.depth_gate_init,
        chunk_size=args.chunk_size,
    )

    # Trigger build with dummy forward pass
    dummy = np.array([[1, 2, 3]], dtype=np.int32)
    _ = model(dummy, training=False)

    # Print parameter summary
    total = model.count_params()
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    frozen = total - trainable
    print(f"\n  Total params:      {total:>12,}")
    print(f"  Trainable (depth): {trainable:>12,}")
    print(f"  Frozen (base):     {frozen:>12,}")
    print(f"  Overhead:          {100 * trainable / frozen:.2f}%")

    # ── 3. Compile ────────────────────────────────────────────────
    optimizer = keras.optimizers.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    model.compile(optimizer=optimizer)

    # ── 4. Baseline PPL (before any training) ─────────────────────
    if not args.skip_baseline:
        print(
            f"\nBaseline PPL (untrained depth layer, maxlen={args.baseline_maxlen})..."
        )
        tok = Tokenizer()
        eval_loader = DataLoader(
            batch_size=1,
            max_steps=20,
            maxlen=args.baseline_maxlen,
            tok=tok,
            iterator_fn=_iterate_synth_wrapper,
        )
        total_loss = 0.0
        total_count = 0
        with inference_mode():
            for X, y in eval_loader:
                X = np.array(X, dtype=np.int32)
                y_np = np.array(y, dtype=np.int32)
                logits = model(X, training=False)
                mask = ops.cast(y_np != 0, "float32")
                per_token = keras.losses.sparse_categorical_crossentropy(
                    y_np, logits, from_logits=True
                )
                total_loss += float(ops.sum(per_token * mask))
                total_count += float(ops.sum(mask))
        baseline_loss = total_loss / max(total_count, 1)
        baseline_ppl = float(np.exp(min(baseline_loss, 20.0)))
        print(f"  Baseline: loss={baseline_loss:.4f}  ppl={baseline_ppl:.2f}")
    else:
        print("\nSkipping baseline eval (--skip-baseline)")

    # ── 5. Train ──────────────────────────────────────────────────
    print(
        f"\nStarting training ({args.total_steps} steps, bs={args.batch_size}, "
        f"maxlen={args.maxlen}, lr={args.lr})...\n"
    )

    num_epochs = args.total_steps // args.log_every
    data_gen = make_data_generator(args.batch_size, args.maxlen, args.total_steps)

    callbacks = [PPLLogger(args.log_every, args.total_steps)]

    # Periodic checkpoint saving (only depth block weights)
    if args.save_every > 0:
        save_epochs = args.save_every // args.log_every
        if save_epochs > 0:
            mode_str = "deltas" if use_deltas else "outputs"
            ckpt_path = os.path.join(
                args.save_dir,
                f"depth_seqcond_{mode_str}_step{{epoch:05d}}.weights.h5",
            )
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    filepath=ckpt_path,
                    save_weights_only=True,
                    save_freq=save_epochs,
                )
            )

    model.fit(
        data_gen,
        steps_per_epoch=args.log_every,
        epochs=num_epochs,
        verbose=0,
        callbacks=callbacks,
    )

    # ── 6. Save final (only depth block weights — much smaller) ──
    mode_str = "deltas" if use_deltas else "outputs"
    final_path = os.path.join(
        args.save_dir, f"depth_seqcond_{mode_str}_final.weights.h5"
    )
    try:
        model.save_weights(final_path)
        print(f"\nSaved final weights to {final_path}")
    except OSError as e:
        print(f"\nWarning: could not save weights: {e}")
    print("Done!")


if __name__ == "__main__":
    main()

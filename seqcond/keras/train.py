"""Training utilities for Keras 3 SeqCond models.

Provides:
- Masked sparse cross-entropy loss (ignoring padding token 0)
- Warmup + cosine-decay LR schedule
- Optimizer factory (AdamW with gradient clipping)
- Checkpoint save / load (compatible with JAX .pkl format)
- High-level Trainer class that wraps model.fit()
"""

import os
import time
import math
import pickle
from typing import Any, Optional

import numpy as np
import keras
from keras import ops

from ..config import ModelConfig, TrainingConfig, Config
from .model import create_seqcond_model


# ═══════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════

class MaskedSparseCategoricalCrossentropy(keras.losses.Loss):
    """Sparse categorical cross-entropy that ignores a padding class."""

    def __init__(self, ignore_class=0, **kwargs):
        super().__init__(**kwargs)
        self.ignore_class = ignore_class

    def call(self, y_true, y_pred):
        y_true = ops.cast(ops.squeeze(y_true, axis=-1) if y_true.ndim > 2 else y_true, "int32")
        mask = ops.cast(y_true != self.ignore_class, "float32")

        y_true_flat = ops.reshape(y_true, (-1,))
        y_pred_flat = ops.reshape(y_pred, (-1, ops.shape(y_pred)[-1]))
        mask_flat = ops.reshape(mask, (-1,))

        log_probs = ops.log_softmax(ops.cast(y_pred_flat, "float32"), axis=-1)
        indices = ops.arange(ops.shape(y_true_flat)[0])
        selected = ops.take_along_axis(
            log_probs, ops.expand_dims(y_true_flat, -1), axis=-1
        )
        selected = ops.squeeze(selected, axis=-1)

        masked = selected * mask_flat
        total = ops.sum(masked)
        count = ops.sum(mask_flat)
        return ops.where(count > 0, -total / count, 0.0)


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

class MaskedAccuracy(keras.metrics.Metric):
    """Accuracy metric that ignores a specific class (e.g. padding=0)."""

    def __init__(self, ignore_class=0, name="accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.ignore_class = ignore_class
        self.correct = self.add_variable(
            name="correct", shape=(), initializer="zeros"
        )
        self.total = self.add_variable(
            name="total", shape=(), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(y_true, "int32")
        if y_true.ndim > 2:
            y_true = ops.squeeze(y_true, axis=-1)
        mask = ops.cast(y_true != self.ignore_class, "float32")
        preds = ops.argmax(y_pred, axis=-1)
        correct = ops.cast(y_true == preds, "float32") * mask
        self.correct.assign(self.correct + ops.sum(correct))
        self.total.assign(self.total + ops.sum(mask))

    def result(self):
        return ops.where(self.total > 0, self.correct / self.total, 0.0)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


class Perplexity(keras.metrics.Metric):
    """Perplexity metric (from masked loss)."""

    def __init__(self, ignore_class=0, name="perplexity", **kwargs):
        super().__init__(name=name, **kwargs)
        self.ignore_class = ignore_class
        self.total_loss = self.add_variable(
            name="total_loss", shape=(), initializer="zeros"
        )
        self.total_count = self.add_variable(
            name="total_count", shape=(), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(y_true, "int32")
        if y_true.ndim > 2:
            y_true = ops.squeeze(y_true, axis=-1)
        mask = ops.cast(y_true != self.ignore_class, "float32")

        y_true_flat = ops.reshape(y_true, (-1,))
        y_pred_flat = ops.reshape(y_pred, (-1, ops.shape(y_pred)[-1]))
        mask_flat = ops.reshape(mask, (-1,))

        log_probs = ops.log_softmax(ops.cast(y_pred_flat, "float32"), axis=-1)
        selected = ops.take_along_axis(
            log_probs, ops.expand_dims(y_true_flat, -1), axis=-1
        )
        selected = ops.squeeze(selected, axis=-1)

        count = ops.sum(mask_flat)
        loss_sum = -ops.sum(selected * mask_flat)
        self.total_loss.assign(self.total_loss + loss_sum)
        self.total_count.assign(self.total_count + count)

    def result(self):
        mean_loss = ops.where(
            self.total_count > 0,
            self.total_loss / self.total_count,
            0.0,
        )
        return ops.clip(ops.exp(mean_loss), 0.0, 10000.0)

    def reset_state(self):
        self.total_loss.assign(0.0)
        self.total_count.assign(0.0)


# ═══════════════════════════════════════════════════════════════════════
# LR Schedule
# ═══════════════════════════════════════════════════════════════════════

class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Warmup then cosine decay to alpha."""

    def __init__(self, base_lr, warmup_steps, total_steps, alpha=1e-5):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.decay_steps = max(total_steps - warmup_steps, 1)
        self.alpha = float(alpha)

    def __call__(self, step):
        step = ops.cast(step, "float32")
        # Warmup phase
        warmup_lr = self.base_lr * (step / max(self.warmup_steps, 1))
        # Cosine decay phase
        progress = ops.clip(
            (step - self.warmup_steps) / self.decay_steps, 0.0, 1.0
        )
        cosine_lr = self.alpha + (self.base_lr - self.alpha) * 0.5 * (
            1.0 + ops.cos(ops.cast(math.pi, "float32") * progress)
        )
        return ops.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "alpha": self.alpha,
        }


# ═══════════════════════════════════════════════════════════════════════
# Optimizer
# ═══════════════════════════════════════════════════════════════════════

def create_optimizer(
    base_lr=1e-3,
    warmup_steps=1000,
    total_steps=1_000_000,
    weight_decay=1e-2,
    clipnorm=1.0,
    beta_1=0.9,
    beta_2=0.999,
):
    """Create AdamW optimizer with warmup-cosine LR schedule."""
    lr_schedule = WarmupCosineDecay(
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    return keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=1e-7,
        weight_decay=weight_decay,
        clipnorm=clipnorm,
    )


# ═══════════════════════════════════════════════════════════════════════
# Checkpointing (compatible with JAX .pkl format)
# ═══════════════════════════════════════════════════════════════════════

def save_checkpoint(model, config, path, step=None):
    """Save model weights + config in the shared .pkl format."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Extract weights as numpy arrays keyed by layer name
    params = {}
    for w in model.weights:
        params[w.path] = np.array(w.numpy())

    data = {
        "params": params,
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
        "step": step,
        "backend": "keras3",
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path):
    """Load checkpoint. Returns (weight_dict, config_dict, step)."""
    with open(path, "rb") as f:
        try:
            data = pickle.load(f)
        except (UnicodeDecodeError, ValueError):
            f.seek(0)
            data = pickle.load(f, encoding="latin1")
    return data["params"], data.get("config"), data.get("step")


# ═══════════════════════════════════════════════════════════════════════
# Model factory from config
# ═══════════════════════════════════════════════════════════════════════

def create_model_from_config(config: ModelConfig):
    """Create a SeqCondModel from a ModelConfig."""
    if config.model_type != "seqcond":
        raise ValueError(
            f"Keras backend only supports 'seqcond' model_type, got '{config.model_type}'"
        )
    return create_seqcond_model(
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
        maxlen=config.maxlen,
        seqcond_ratio=config.seqcond_ratio,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        seqcond_heads=config.seqcond_heads,
        num_query_heads=config.num_query_heads,
        num_anchor_heads=config.num_anchor_heads,
        num_thetas=config.num_thetas,
        dropout=config.dropout,
        tie_weights=config.tie_weights,
        qk_norm=config.qk_norm,
        qk_norm_eps=config.qk_norm_eps,
        conv_kernel_size=config.conv_kernel_size,
        expand_factor=config.expand_factor,
        out_expand_factor=config.out_expand_factor,
        use_square_matrix=config.use_square_matrix,
    )


# ═══════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════

class Trainer:
    """High-level trainer for Keras 3 SeqCond models.

    Uses model.compile() + model.fit() with custom loss and metrics.
    Compatible with JAX, TensorFlow, and PyTorch backends.
    """

    def __init__(
        self,
        config: Config,
        tokenizer: Any = None,
        model_name: Optional[str] = None,
        load_checkpoint_path: Optional[str] = None,
    ):
        self.config = config
        self.model_config = config.model
        self.train_config = config.training
        self.tokenizer = tokenizer
        self.model_name = model_name or config.name
        self.load_checkpoint_path = load_checkpoint_path
        self.model = None

    def setup(self):
        """Create model, optimizer, compile."""
        tc = self.train_config
        mc = self.model_config
        print("=" * 60)
        print("Keras 3 Trainer Setup")
        print(f"Backend: {keras.backend.backend()}")
        print("=" * 60)

        # Mixed precision
        if tc.mixed_precision == "bfloat16":
            keras.mixed_precision.set_global_policy("mixed_bfloat16")
            print("Mixed precision: bfloat16")
        elif tc.mixed_precision == "float16":
            keras.mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision: float16")
        else:
            print("Precision: float32")

        # Model
        print("Creating model...")
        self.model = create_model_from_config(mc)

        # Build by running a dummy forward pass
        dummy = np.ones((1, mc.maxlen), dtype=np.int32)
        _ = self.model(dummy, training=False)

        num_params = self.model.count_params()
        print(f"Model type: {mc.model_type}")
        print(f"Parameters: {num_params:,}")

        # Load weights if requested
        if self.load_checkpoint_path:
            print(f"Loading weights from {self.load_checkpoint_path}...")
            weights, _, ckpt_step = load_checkpoint(self.load_checkpoint_path)
            # Try to match by name
            loaded = 0
            for w in self.model.weights:
                if w.path in weights:
                    w.assign(weights[w.path])
                    loaded += 1
            print(f"Loaded {loaded}/{len(self.model.weights)} weights (step {ckpt_step})")

        # Optimizer
        optimizer = create_optimizer(
            base_lr=tc.base_lr,
            warmup_steps=tc.warmup_steps,
            total_steps=tc.total_steps,
            weight_decay=tc.weight_decay,
            clipnorm=tc.clipnorm,
            beta_1=tc.beta_1,
            beta_2=tc.beta_2,
        )

        # Compile
        self.model.compile(
            optimizer=optimizer,
            loss=MaskedSparseCategoricalCrossentropy(ignore_class=0),
            metrics=[
                MaskedAccuracy(ignore_class=0),
                Perplexity(ignore_class=0),
            ],
        )
        print("Model compiled.\n")
        return self

    def train(self, dataset, validation_data=None):
        """Run training with model.fit().

        Args:
            dataset: A tf.data.Dataset or generator yielding (x, y) batches.
                     x: (batch, seq_len) int32 input token IDs
                     y: (batch, seq_len) int32 target token IDs
            validation_data: Optional validation dataset.

        Returns:
            Training history.
        """
        tc = self.train_config

        callbacks = [
            keras.callbacks.TerminateOnNaN(),
        ]

        # Checkpointing
        if tc.save_every_n_steps > 0:
            ckpt_cb = CheckpointCallback(
                model_name=self.model_name,
                config=self.config,
                checkpoint_dir=tc.checkpoint_dir,
                save_every=tc.save_every_n_steps,
            )
            callbacks.append(ckpt_cb)

        # Wandb
        if tc.use_wandb:
            try:
                import wandb

                wandb.init(
                    project=tc.wandb_project,
                    name=self.model_name,
                    config=self.config.to_dict(),
                )
                callbacks.append(
                    keras.callbacks.LambdaCallback(
                        on_batch_end=lambda batch, logs: wandb.log(logs, step=batch)
                    )
                )
            except ImportError:
                print("Warning: wandb not installed, skipping.")

        history = self.model.fit(
            dataset,
            epochs=1,  # single epoch, total steps controlled by dataset size
            steps_per_epoch=tc.total_steps,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
        )
        return history


class CheckpointCallback(keras.callbacks.Callback):
    """Save checkpoints every N steps in .pkl format."""

    def __init__(self, model_name, config, checkpoint_dir, save_every):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every

    def on_batch_end(self, batch, logs=None):
        step = batch + 1
        if step % self.save_every == 0:
            path = f"{self.checkpoint_dir}/{self.model_name}_step{step}.pkl"
            save_checkpoint(self.model, self.config, path, step=step)

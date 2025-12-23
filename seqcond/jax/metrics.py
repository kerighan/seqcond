from typing import Optional

import jax
import jax.numpy as jnp


def sparse_categorical_accuracy_ignore_zero(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    ignore_class: int = 0,
) -> jnp.ndarray:
    """Compute sparse categorical accuracy ignoring a specific class."""
    mask = y_true != ignore_class
    y_pred_classes = jnp.argmax(y_pred, axis=-1)
    correct = (y_true == y_pred_classes).astype(jnp.float32)
    masked_correct = jnp.where(mask, correct, 0.0)
    total = jnp.sum(mask.astype(jnp.float32))
    count = jnp.sum(masked_correct)
    return jnp.where(total > 0, count / total, 0.0)


def sparse_top_k_categorical_accuracy_ignore_zero(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    k: int = 5,
    ignore_class: int = 0,
) -> jnp.ndarray:
    """Compute sparse top-k categorical accuracy ignoring a specific class."""
    mask = y_true != ignore_class
    _, top_k_indices = jax.lax.top_k(y_pred, k)
    y_true_expanded = jnp.expand_dims(y_true, -1)
    correct = jnp.any(y_true_expanded == top_k_indices, axis=-1).astype(jnp.float32)
    masked_correct = jnp.where(mask, correct, 0.0)
    total = jnp.sum(mask.astype(jnp.float32))
    count = jnp.sum(masked_correct)
    return jnp.where(total > 0, count / total, 0.0)


def sparse_weighted_categorical_accuracy_ignore_zero(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    ignore_class: int = 0,
    weighting_method: str = "linear",
) -> jnp.ndarray:
    """Compute sparse weighted categorical accuracy ignoring a specific class."""
    mask = y_true != ignore_class
    y_pred_classes = jnp.argmax(y_pred, axis=-1)
    correct = (y_true == y_pred_classes).astype(jnp.float32)
    masked_correct = correct * mask.astype(jnp.float32)

    sequence_length = y_true.shape[1]
    positions = jnp.arange(1, sequence_length + 1, dtype=jnp.float32)
    sequence_length_float = jnp.float32(sequence_length)

    if weighting_method == "linear":
        weights = positions / sequence_length_float
    elif weighting_method == "quadratic":
        weights = jnp.square(positions / sequence_length_float)
    elif weighting_method == "exponential":
        weights = jnp.exp(positions / sequence_length_float) - 1.0
        weights = weights / jnp.max(weights)
    else:
        weights = positions / sequence_length_float

    batch_size = y_true.shape[0]
    position_weights = jnp.tile(weights[None, :], (batch_size, 1))

    weighted_correct = masked_correct * position_weights
    weighted_mask = mask.astype(jnp.float32) * position_weights

    weighted_total = jnp.sum(weighted_mask)
    weighted_count = jnp.sum(weighted_correct)
    return jnp.where(weighted_total > 0, weighted_count / weighted_total, 0.0)


def perplexity(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    ignore_class: int = 0,
) -> jnp.ndarray:
    """Compute perplexity ignoring a specific class."""
    y_true = y_true.astype(jnp.int32)
    mask = y_true != ignore_class

    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    mask_flat = mask.reshape(-1)

    log_probs = jax.nn.log_softmax(y_pred_flat, axis=-1)
    indices = jnp.arange(y_true_flat.shape[0])
    selected_log_probs = log_probs[indices, y_true_flat]

    masked_log_probs = jnp.where(mask_flat, selected_log_probs, 0.0)
    total_log_prob = jnp.sum(masked_log_probs)
    count = jnp.sum(mask_flat.astype(jnp.float32))

    mean_loss = jnp.where(count > 0, -total_log_prob / count, 0.0)
    ppx = jnp.exp(mean_loss)
    ppx = jnp.clip(ppx, 0.0, 10000.0)
    return jnp.round(ppx * 100.0) / 100.0


def sparse_categorical_crossentropy(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    ignore_class: int = 0,
) -> jnp.ndarray:
    """Compute sparse categorical cross-entropy loss ignoring a specific class."""
    y_true = y_true.astype(jnp.int32)
    mask = y_true != ignore_class

    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    mask_flat = mask.reshape(-1)

    log_probs = jax.nn.log_softmax(y_pred_flat, axis=-1)
    indices = jnp.arange(y_true_flat.shape[0])
    selected_log_probs = log_probs[indices, y_true_flat]

    masked_log_probs = jnp.where(mask_flat, selected_log_probs, 0.0)
    total_log_prob = jnp.sum(masked_log_probs)
    count = jnp.sum(mask_flat.astype(jnp.float32))

    return jnp.where(count > 0, -total_log_prob / count, 0.0)


class MetricsAccumulator:
    """Accumulator for computing metrics over multiple batches."""

    def __init__(self, ignore_class: int = 0):
        self.ignore_class = ignore_class
        self.reset()

    def reset(self):
        # Initialize as JAX arrays to avoid implicit sync
        self.total_loss = jnp.array(0.0, dtype=jnp.float32)
        self.total_correct = jnp.array(0.0, dtype=jnp.float32)
        self.total_top_k_correct = jnp.array(0.0, dtype=jnp.float32)
        self.total_count = jnp.array(0.0, dtype=jnp.float32)

    def update(
        self,
        y_true: jnp.ndarray,
        y_pred: jnp.ndarray,
        loss: Optional[jnp.ndarray] = None,
        k: int = 5,
    ):
        mask = y_true != self.ignore_class
        count = jnp.sum(mask.astype(jnp.float32))

        y_pred_classes = jnp.argmax(y_pred, axis=-1)
        correct = (y_true == y_pred_classes).astype(jnp.float32)
        masked_correct = jnp.where(mask, correct, 0.0)

        _, top_k_indices = jax.lax.top_k(y_pred, k)
        y_true_expanded = jnp.expand_dims(y_true, -1)
        top_k_correct = jnp.any(y_true_expanded == top_k_indices, axis=-1).astype(
            jnp.float32
        )
        masked_top_k_correct = jnp.where(mask, top_k_correct, 0.0)

        # Accumulate as JAX arrays (non-blocking)
        self.total_count += count
        self.total_correct += jnp.sum(masked_correct)
        self.total_top_k_correct += jnp.sum(masked_top_k_correct)

        if loss is not None:
            self.total_loss += loss * count

    def result(self):
        # Returns JAX arrays. Call jax.device_get() on the result dict.
        # Avoid division by zero
        safe_count = jnp.maximum(self.total_count, 1.0)
        
        acc = self.total_correct / safe_count
        topk = self.total_top_k_correct / safe_count
        mean_loss = self.total_loss / safe_count
        
        ppx = jnp.exp(mean_loss)
        # Clip and round could be done here or in logging, but keep consistent
        ppx = jnp.clip(ppx, 0.0, 10000.0)
        
        # Return scalar JAX arrays
        return {
            "acc": acc,
            "topk": topk,
            "ppx": ppx,
            "loss": mean_loss,
        }

"""Normalization layers for Keras 3."""

import keras
from keras import ops, layers


class RMSNorm(layers.Layer):
    """Root Mean Square Layer Normalization."""

    supports_masking = True

    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(input_shape[-1],),
            initializer="ones",
        )
        super().build(input_shape)

    def call(self, x):
        x_f32 = ops.cast(x, "float32")
        mean_sq = ops.mean(ops.square(x_f32), axis=-1, keepdims=True)
        y = x_f32 * ops.rsqrt(mean_sq + self.epsilon)
        y = y * ops.cast(self.scale, y.dtype)
        return ops.cast(y, x.dtype)


def gated_rmsnorm(x, residual, weight, epsilon=1e-6):
    """Functional GatedRMSNorm: x = rmsnorm(x * silu(residual)).

    Args:
        x: Input tensor.
        residual: Gating tensor (same shape as x).
        weight: Scale weight (last dim of x).
        epsilon: Numerical stability constant.

    Returns:
        Normalized and gated tensor (float32).
    """
    x_f32 = ops.cast(x, "float32")
    res_f32 = ops.cast(residual, "float32")
    x_f32 = x_f32 * ops.silu(res_f32)
    variance = ops.mean(ops.square(x_f32), axis=-1, keepdims=True)
    x_f32 = x_f32 * ops.rsqrt(variance + epsilon)
    return x_f32 * weight

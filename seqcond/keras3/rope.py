"""Rotary Position Embedding and Transformer blocks for Keras 3."""

from typing import Optional, Tuple

import numpy as np
import keras
from keras import ops, layers

from .norm import RMSNorm


def precompute_freqs(maxlen: int, head_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute cos and sin for rotary embeddings."""
    if head_dim % 2:
        raise ValueError("head_dim must be even")
    half_d = head_dim // 2
    pos = np.arange(maxlen)[:, None]
    dim = np.arange(half_d)[None, :]
    inv = 1.0 / (10000 ** (dim / half_d))
    angles = pos * inv
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    return cos, sin


def get_rope_embeddings(
    seq_len: int,
    cos_emb: np.ndarray,
    sin_emb: np.ndarray,
    batch_size: int,
    n_heads: Optional[int] = None,
):
    """Get rotary embeddings for a given sequence length."""
    cos, sin = cos_emb[:seq_len], sin_emb[:seq_len]
    if n_heads is None:
        cos = ops.broadcast_to(ops.expand_dims(cos, 0), (batch_size, *cos.shape))
        sin = ops.broadcast_to(ops.expand_dims(sin, 0), (batch_size, *sin.shape))
    else:
        cos = ops.broadcast_to(
            ops.reshape(cos, (1, seq_len, 1, cos.shape[-1])),
            (batch_size, seq_len, n_heads, cos.shape[-1]),
        )
        sin = ops.broadcast_to(
            ops.reshape(sin, (1, seq_len, 1, sin.shape[-1])),
            (batch_size, seq_len, n_heads, sin.shape[-1]),
        )
    return cos, sin


def apply_rope(tensor, cos, sin):
    """Apply rotary positional embedding to tensor."""
    dim = tensor.shape[-1] // 2
    cos = cos[..., :dim]
    sin = sin[..., :dim]
    x1, x2 = tensor[..., :dim], tensor[..., dim:]
    rot = ops.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return rot


class RotarySelfAttention(layers.Layer):
    """Multi-head self-attention with Rotary Position Embeddings and GQA support."""

    supports_masking = True

    def __init__(
        self,
        d_model,
        num_heads,
        num_kv_heads=None,
        dropout=0.0,
        qk_norm=False,
        qk_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self._num_kv_heads = num_kv_heads or num_heads
        self.dropout_rate = dropout
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps
        self.num_groups = num_heads // self._num_kv_heads
        self.head_dim = d_model // num_heads

        self.q_proj = layers.Dense(d_model, use_bias=False, name="q_proj")
        self.k_proj = layers.Dense(
            self._num_kv_heads * self.head_dim, use_bias=False, name="k_proj"
        )
        self.v_proj = layers.Dense(
            self._num_kv_heads * self.head_dim, use_bias=False, name="v_proj"
        )
        self.out_proj = layers.Dense(d_model, use_bias=False, name="out_proj")
        self.attn_dropout = layers.Dropout(dropout)

    def _repeat_kv(self, x):
        """Repeat KV heads to match Q heads for GQA."""
        if self.num_groups == 1:
            return x
        b = ops.shape(x)[0]
        l = ops.shape(x)[1]
        x = ops.reshape(x, (b, l, self._num_kv_heads, 1, self.head_dim))
        x = ops.broadcast_to(
            x, (b, l, self._num_kv_heads, self.num_groups, self.head_dim)
        )
        return ops.reshape(x, (b, l, self.num_heads, self.head_dim))

    def call(self, x, cos, sin, mask=None, training=False):
        b = ops.shape(x)[0]
        l = ops.shape(x)[1]

        q = ops.reshape(self.q_proj(x), (b, l, self.num_heads, self.head_dim))
        k = ops.reshape(self.k_proj(x), (b, l, self._num_kv_heads, self.head_dim))
        v = ops.reshape(self.v_proj(x), (b, l, self._num_kv_heads, self.head_dim))

        q = apply_rope(q, cos, sin)
        if self._num_kv_heads < self.num_heads:
            cos_kv = cos[:, :, : self._num_kv_heads, :]
            sin_kv = sin[:, :, : self._num_kv_heads, :]
            k = apply_rope(k, cos_kv, sin_kv)
        else:
            k = apply_rope(k, cos, sin)

        if self.qk_norm:
            q_f32 = ops.cast(q, "float32")
            k_f32 = ops.cast(k, "float32")
            q_ms = ops.mean(ops.square(q_f32), axis=-1, keepdims=True)
            k_ms = ops.mean(ops.square(k_f32), axis=-1, keepdims=True)
            q = ops.cast(q_f32 * ops.rsqrt(q_ms + self.qk_norm_eps), q.dtype)
            k = ops.cast(k_f32 * ops.rsqrt(k_ms + self.qk_norm_eps), k.dtype)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        scale = ops.rsqrt(ops.cast(self.head_dim, "float32"))
        scores = ops.einsum("blhd,bmhd->bhlm", q, k) * scale

        # Causal mask + padding mask
        causal_mask = ops.tril(ops.ones((l, l)))
        large_neg = ops.cast(-1e4, scores.dtype)

        if mask is not None:
            key_mask = ops.cast(mask, scores.dtype)
            key_mask = ops.reshape(key_mask, (b, 1, 1, l))
            scores = scores + (1.0 - key_mask) * large_neg

        causal_mask = ops.reshape(causal_mask, (1, 1, l, l))
        scores = scores + (1.0 - causal_mask) * large_neg

        attn = ops.softmax(ops.cast(scores, "float32"), axis=-1)
        attn = ops.cast(attn, v.dtype)
        attn = self.attn_dropout(attn, training=training)

        out = ops.einsum("bhql,blhd->bqhd", attn, v)
        out = ops.reshape(out, (b, l, self.d_model))
        return self.out_proj(out)

    def step(self, x_t, kv_cache, pos, cos_t, sin_t):
        """Single autoregressive step with KV cache.

        Args:
            x_t: (B, d_model) current token embedding
            kv_cache: (k_cache, v_cache) tuple
                - k_cache: (B, maxlen, num_kv_heads, head_dim)
                - v_cache: (B, maxlen, num_kv_heads, head_dim)
            pos: (B,) current position indices
            cos_t: (B, 1, num_heads, head_dim) RoPE cos for current position
            sin_t: (B, 1, num_heads, head_dim) RoPE sin for current position

        Returns:
            out: (B, d_model) output
            new_kv_cache: updated (k_cache, v_cache)
        """
        k_cache, v_cache = kv_cache
        B = ops.shape(x_t)[0]

        # Project query for current token
        q = ops.reshape(self.q_proj(x_t), (B, 1, self.num_heads, self.head_dim))
        k_new = ops.reshape(self.k_proj(x_t), (B, 1, self._num_kv_heads, self.head_dim))
        v_new = ops.reshape(self.v_proj(x_t), (B, 1, self._num_kv_heads, self.head_dim))

        # Apply RoPE
        q = apply_rope(q, cos_t, sin_t)
        if self._num_kv_heads < self.num_heads:
            cos_kv = cos_t[:, :, : self._num_kv_heads, :]
            sin_kv = sin_t[:, :, : self._num_kv_heads, :]
            k_new = apply_rope(k_new, cos_kv, sin_kv)
        else:
            k_new = apply_rope(k_new, cos_t, sin_t)

        if self.qk_norm:
            q_f32 = ops.cast(q, "float32")
            k_f32 = ops.cast(k_new, "float32")
            q_ms = ops.mean(ops.square(q_f32), axis=-1, keepdims=True)
            k_ms = ops.mean(ops.square(k_f32), axis=-1, keepdims=True)
            q = ops.cast(q_f32 * ops.rsqrt(q_ms + self.qk_norm_eps), q.dtype)
            k_new = ops.cast(k_f32 * ops.rsqrt(k_ms + self.qk_norm_eps), k_new.dtype)

        # Update KV cache at position pos
        pos_int = ops.cast(pos, "int32")

        import keras

        if keras.backend.backend() == "torch":
            # Efficient in-place scatter update
            import torch

            pos_idx = (
                pos_int.long()
                .view(B, 1, 1, 1)
                .expand(-1, 1, k_new.size(2), k_new.size(3))
            )
            k_cache.scatter_(1, pos_idx, k_new.to(k_cache.dtype))
            v_cache.scatter_(1, pos_idx, v_new.to(v_cache.dtype))
        else:
            # Fallback: one-hot mask blend
            maxlen_full = ops.shape(k_cache)[1]
            positions = ops.reshape(
                ops.arange(maxlen_full, dtype="int32"), (1, maxlen_full)
            )
            pos_mask = ops.cast(
                ops.equal(positions, ops.reshape(pos_int, (B, 1))), k_cache.dtype
            )
            pos_mask = ops.reshape(pos_mask, (B, maxlen_full, 1, 1))
            k_new_broadcast = ops.broadcast_to(
                k_new, (B, maxlen_full, self._num_kv_heads, self.head_dim)
            )
            v_new_broadcast = ops.broadcast_to(
                v_new, (B, maxlen_full, self._num_kv_heads, self.head_dim)
            )
            k_cache = k_cache * (1.0 - pos_mask) + k_new_broadcast * pos_mask
            v_cache = v_cache * (1.0 - pos_mask) + v_new_broadcast * pos_mask

        # Slice KV cache to max_pos+1 to avoid operating on unused positions
        max_pos = int(ops.max(pos_int)) + 1
        k_slice = k_cache[:, :max_pos, :, :]
        v_slice = v_cache[:, :max_pos, :, :]

        # Repeat KV for GQA (only over the active slice)
        k_repeated = self._repeat_kv(k_slice)  # (B, max_pos, num_heads, head_dim)
        v_repeated = self._repeat_kv(v_slice)

        # Create causal mask over the slice only
        all_positions = ops.reshape(
            ops.arange(max_pos, dtype="int32"), (1, 1, 1, max_pos)
        )
        pos_for_mask = ops.reshape(pos_int, (B, 1, 1, 1))
        causal_mask = ops.cast(all_positions <= pos_for_mask, k_slice.dtype)

        # Attention scores
        scale = ops.rsqrt(ops.cast(self.head_dim, "float32"))
        scores = (
            ops.einsum("bqhd,bkhd->bhqk", q, k_repeated) * scale
        )  # (B, num_heads, 1, max_pos)

        # Apply causal mask
        large_neg = ops.cast(-1e9, scores.dtype)
        scores = scores + (1.0 - causal_mask) * large_neg

        attn = ops.softmax(ops.cast(scores, "float32"), axis=-1)
        attn = ops.cast(attn, v_repeated.dtype)

        # Output
        out = ops.einsum(
            "bhqk,bkhd->bqhd", attn, v_repeated
        )  # (B, 1, num_heads, head_dim)
        out = ops.reshape(out, (B, self.d_model))

        return self.out_proj(out), (k_cache, v_cache)


class TransformerDecoderBlock(layers.Layer):
    """Pre-norm Transformer decoder block with SwiGLU FFN."""

    supports_masking = True

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        num_kv_heads=None,
        dropout=0.0,
        norm_eps=1e-6,
        qk_norm=False,
        qk_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm1 = RMSNorm(epsilon=norm_eps, name="norm1")
        self.attn = RotarySelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            name="attn",
        )
        self.drop1 = layers.Dropout(dropout)

        self.norm2 = RMSNorm(epsilon=norm_eps, name="norm2")
        self.ff_in = layers.Dense(2 * d_ff, name="ff_in")
        self.ff_out = layers.Dense(d_model, name="ff_out")
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, cos, sin, mask=None, training=False):
        y = self.norm1(x)
        y = self.attn(y, cos=cos, sin=sin, mask=mask, training=training)
        x = x + self.drop1(y, training=training)

        y = self.norm2(x)
        u, v = ops.split(self.ff_in(y), 2, axis=-1)
        y = ops.silu(v) * u
        y = self.ff_out(y)
        return x + self.drop2(y, training=training)

    def step(self, x_t, kv_cache, pos, cos_t, sin_t):
        """Single autoregressive step.

        Args:
            x_t: (B, d_model) current token embedding
            kv_cache: (k_cache, v_cache) tuple
                - k_cache: (B, maxlen, num_kv_heads, head_dim)
                - v_cache: (B, maxlen, num_kv_heads, head_dim)
            pos: (B,) current position indices
            cos_t: (B, 1, num_heads, head_dim) RoPE cos for current position
            sin_t: (B, 1, num_heads, head_dim) RoPE sin for current position

        Returns:
            out: (B, d_model) output
            new_kv_cache: updated (k_cache, v_cache)
        """
        k_cache, v_cache = kv_cache
        B = ops.shape(x_t)[0]

        # Attention
        y = self.norm1(x_t)
        y, new_kv_cache = self.attn.step(y, kv_cache, pos, cos_t, sin_t)
        x_t = x_t + y

        # FFN
        y = self.norm2(x_t)
        u, v = ops.split(self.ff_in(y), 2, axis=-1)
        y = ops.silu(v) * u
        y = self.ff_out(y)

        return x_t + y, new_kv_cache

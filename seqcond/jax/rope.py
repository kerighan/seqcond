import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

from .norm import RMSNorm


def precompute_freqs(maxlen: int, head_dim: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    return jnp.array(cos), jnp.array(sin)


def get_rope_embeddings(
    seq_len: int,
    cos_emb: jnp.ndarray,
    sin_emb: jnp.ndarray,
    batch_size: int,
    n_heads: Optional[int] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get rotary embeddings for a given sequence length."""
    cos, sin = cos_emb[:seq_len], sin_emb[:seq_len]
    if n_heads is None:
        cos = jnp.broadcast_to(cos[None, ...], (batch_size, *cos.shape))
        sin = jnp.broadcast_to(sin[None, ...], (batch_size, *sin.shape))
    else:
        cos = jnp.broadcast_to(
            cos[None, :, None, :], (batch_size, seq_len, n_heads, cos.shape[-1])
        )
        sin = jnp.broadcast_to(
            sin[None, :, None, :], (batch_size, seq_len, n_heads, sin.shape[-1])
        )
    return cos, sin


def apply_rope(tensor: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """Apply rotary positional embedding to tensor.

    Automatically crops cos/sin to the size required by `tensor` to support
    caller subspace projections (e.g. bivector attention halves)."""
    if tensor.shape[-1] % 2 != 0:
        raise ValueError("RoPE tensor last dimension must be even.")

    dim = tensor.shape[-1] // 2
    if cos.shape[-1] < dim or sin.shape[-1] < dim:
        raise ValueError(
            f"RoPE embeddings too small for tensor. "
            f"Need {dim}, got cos={cos.shape[-1]}, sin={sin.shape[-1]}"
        )

    cos = cos[..., :dim]
    sin = sin[..., :dim]

    x1, x2 = tensor[..., :dim], tensor[..., dim:]
    rot = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return rot.reshape(tensor.shape)


def _make_causal_mask(length: int) -> jnp.ndarray:
    """Create causal mask. JAX will cache/optimize this during compilation."""
    return jnp.tril(jnp.ones((length, length), dtype=jnp.bool_))


class RotarySelfAttention(nn.Module):
    d_model: int
    num_heads: int
    num_kv_heads: Optional[int] = None
    dropout: float = 0.0
    qk_norm: bool = False
    qk_norm_eps: float = 1e-6

    def setup(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self._num_kv_heads = (
            self.num_kv_heads if self.num_kv_heads is not None else self.num_heads
        )
        if self.num_heads % self._num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_groups = self.num_heads // self._num_kv_heads
        self.head_dim = self.d_model // self.num_heads

        self.q_proj = nn.Dense(self.d_model, use_bias=False)
        self.k_proj = nn.Dense(self._num_kv_heads * self.head_dim, use_bias=False)
        self.v_proj = nn.Dense(self._num_kv_heads * self.head_dim, use_bias=False)
        self.out_proj = nn.Dense(self.d_model, use_bias=False)
        self.attn_dropout = nn.Dropout(self.dropout)

    def _repeat_kv(self, x: jnp.ndarray) -> jnp.ndarray:
        """Repeat KV heads to match Q heads for GQA."""
        if self.num_groups == 1:
            return x
        b, l = x.shape[:2]
        extra_shape = x.shape[2:]
        x = x.reshape(b, l, self._num_kv_heads, 1, *extra_shape[1:])
        x = jnp.broadcast_to(
            x, (b, l, self._num_kv_heads, self.num_groups, *extra_shape[1:])
        )
        return x.reshape(b, l, self.num_heads, *extra_shape[1:])

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        cos: jnp.ndarray,
        sin: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        b, l = x.shape[0], x.shape[1]

        q = self.q_proj(x).reshape(b, l, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, l, self._num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, l, self._num_kv_heads, self.head_dim)

        q = apply_rope(q, cos, sin)
        if self._num_kv_heads < self.num_heads:
            cos_kv = cos[:, :, : self._num_kv_heads, :]
            sin_kv = sin[:, :, : self._num_kv_heads, :]
            k = apply_rope(k, cos_kv, sin_kv)
        else:
            k = apply_rope(k, cos, sin)

        if self.qk_norm:
            q_f32 = q.astype(jnp.float32)
            k_f32 = k.astype(jnp.float32)
            q_ms = jnp.mean(jnp.square(q_f32), axis=-1, keepdims=True)
            k_ms = jnp.mean(jnp.square(k_f32), axis=-1, keepdims=True)
            q = (q_f32 * jax.lax.rsqrt(q_ms + self.qk_norm_eps)).astype(q.dtype)
            k = (k_f32 * jax.lax.rsqrt(k_ms + self.qk_norm_eps)).astype(k.dtype)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        scale = jax.lax.rsqrt(jnp.float32(self.head_dim))
        scores = jnp.einsum("blhd,bmhd->bhlm", q, k) * scale

        causal_mask = _make_causal_mask(l)[None, None, :, :]

        large_neg = jnp.float32(-1e4)  # -1e9 overflows float16/bfloat16
        if mask is not None:
            key_mask = mask.astype(scores.dtype)[:, None, None, :]
            scores = scores + (1.0 - key_mask) * large_neg

        scores = jnp.where(causal_mask, scores, jnp.full_like(scores, large_neg))
        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
        attn = attn.astype(v.dtype)
        attn = self.attn_dropout(attn, deterministic=deterministic)

        out = jnp.einsum("bhql,blhd->bqhd", attn, v)
        out = out.reshape(b, l, self.d_model)
        return self.out_proj(out)

    @nn.compact
    def step(self, x_t, kv_cache, pos, cos_t, sin_t, deterministic=True):
        """
        O(L) autoregressive decoding step with KV cache.

        Args:
            x_t: Input token embedding (B, D)
            kv_cache: Tuple of (k_cache, v_cache) where each is (B, L_cache, num_kv_heads, head_dim)
            pos: Current position (scalar or (B,))
            cos_t: RoPE cos for current position (B, 1, num_heads, head_dim//2)
            sin_t: RoPE sin for current position (B, 1, num_heads, head_dim//2)
            deterministic: Whether to use dropout

        Returns:
            out: (B, D) - output for this step
            new_kv_cache: Updated KV cache tuple
        """
        b = x_t.shape[0]

        # Project query for current token
        q = self.q_proj(x_t).reshape(b, 1, self.num_heads, self.head_dim)
        k_new = self.k_proj(x_t).reshape(b, 1, self._num_kv_heads, self.head_dim)
        v_new = self.v_proj(x_t).reshape(b, 1, self._num_kv_heads, self.head_dim)

        # Apply RoPE to query and new key
        q = apply_rope(q, cos_t, sin_t)
        if self._num_kv_heads < self.num_heads:
            cos_kv = cos_t[:, :, : self._num_kv_heads, :]
            sin_kv = sin_t[:, :, : self._num_kv_heads, :]
            k_new = apply_rope(k_new, cos_kv, sin_kv)
        else:
            k_new = apply_rope(k_new, cos_t, sin_t)

        # Update KV cache
        k_cache, v_cache = kv_cache
        k_cache = jnp.concatenate(
            [k_cache, k_new], axis=1
        )  # (B, L+1, num_kv_heads, head_dim)
        v_cache = jnp.concatenate([v_cache, v_new], axis=1)

        # QK normalization
        if self.qk_norm:
            q_f32 = q.astype(jnp.float32)
            k_f32 = k_cache.astype(jnp.float32)
            q_ms = jnp.mean(jnp.square(q_f32), axis=-1, keepdims=True)
            k_ms = jnp.mean(jnp.square(k_f32), axis=-1, keepdims=True)
            q = (q_f32 * jax.lax.rsqrt(q_ms + self.qk_norm_eps)).astype(q.dtype)
            k_cache_norm = (k_f32 * jax.lax.rsqrt(k_ms + self.qk_norm_eps)).astype(
                k_cache.dtype
            )
        else:
            k_cache_norm = k_cache

        # Repeat KV for GQA
        k_repeated = self._repeat_kv(k_cache_norm)
        v_repeated = self._repeat_kv(v_cache)

        # Compute attention (only for current query position)
        scale = jax.lax.rsqrt(jnp.float32(self.head_dim))
        scores = (
            jnp.einsum("bqhd,bkhd->bhqk", q, k_repeated) * scale
        )  # (B, num_heads, 1, L+1)

        # Causal mask: current position can attend to all previous positions
        # No masking needed since we're only computing for the last position

        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
        attn = attn.astype(v_repeated.dtype)
        attn = self.attn_dropout(attn, deterministic=deterministic)

        # Compute output
        out = jnp.einsum(
            "bhqk,bkhd->bqhd", attn, v_repeated
        )  # (B, 1, num_heads, head_dim)
        out = out.reshape(b, self.d_model)

        return self.out_proj(out), (k_cache, v_cache)


class TransformerDecoderBlock(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    num_kv_heads: Optional[int] = None
    dropout: float = 0.0
    norm_eps: float = 1e-6
    qk_norm: bool = False
    qk_norm_eps: float = 1e-6

    def setup(self):
        self.norm1 = RMSNorm(epsilon=self.norm_eps)
        self.attn = RotarySelfAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            dropout=self.dropout,
            qk_norm=self.qk_norm,
            qk_norm_eps=self.qk_norm_eps,
        )
        self.drop1 = nn.Dropout(self.dropout)

        self.norm2 = RMSNorm(epsilon=self.norm_eps)
        self.ff_in = nn.Dense(2 * self.d_ff)
        self.ff_out = nn.Dense(self.d_model)
        self.drop2 = nn.Dropout(self.dropout)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        cos: jnp.ndarray,
        sin: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        y = self.norm1(x)
        y = self.attn(y, cos=cos, sin=sin, mask=mask, deterministic=deterministic)
        x = x + self.drop1(y, deterministic=deterministic)

        y = self.norm2(x)
        u, v = jnp.split(self.ff_in(y), 2, axis=-1)
        y = jax.nn.swish(v) * u
        y = self.ff_out(y)
        return x + self.drop2(y, deterministic=deterministic)

    def step(self, x_t, kv_cache, pos, cos_t, sin_t, deterministic=True):
        """
        O(L) autoregressive decoding step for transformer block.

        Args:
            x_t: Input token embedding (B, D)
            kv_cache: KV cache from attention layer
            pos: Current position
            cos_t: RoPE cos for current position
            sin_t: RoPE sin for current position
            deterministic: Whether to use dropout

        Returns:
            out: (B, D) - output for this step
            new_kv_cache: Updated KV cache
        """
        # Attention with KV cache
        y = self.norm1(x_t)
        y, new_kv_cache = self.attn.step(
            y, kv_cache, pos, cos_t, sin_t, deterministic=deterministic
        )
        x_t = x_t + self.drop1(y, deterministic=deterministic)

        # FFN
        y = self.norm2(x_t)
        u, v = jnp.split(self.ff_in(y), 2, axis=-1)
        y = jax.nn.swish(v) * u
        y = self.ff_out(y)

        return x_t + self.drop2(y, deterministic=deterministic), new_kv_cache

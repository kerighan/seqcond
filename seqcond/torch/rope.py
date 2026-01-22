"""
Rotary Position Embedding and Transformer blocks - mirrors JAX rope.py exactly.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .norm import RMSNorm


def precompute_freqs(maxlen: int, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos and sin for rotary embeddings - matches JAX."""
    if head_dim % 2:
        raise ValueError("head_dim must be even")
    half_d = head_dim // 2
    pos = np.arange(maxlen)[:, None]
    dim = np.arange(half_d)[None, :]
    inv = 1.0 / (10000 ** (dim / half_d))
    angles = pos * inv
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    return torch.from_numpy(cos), torch.from_numpy(sin)


def apply_rope(
    tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary positional embedding to tensor - matches JAX apply_rope."""
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
    rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return rot.view(tensor.shape)


class RotarySelfAttention(nn.Module):
    """Self-attention with rotary position embeddings - matches JAX RotarySelfAttention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self._num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        if num_heads % self._num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.num_groups = num_heads // self._num_kv_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self._num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self._num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = dropout
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match Q heads for GQA - matches JAX."""
        if self.num_groups == 1:
            return x
        b, l = x.shape[:2]
        extra_shape = x.shape[2:]
        x = x.view(b, l, self._num_kv_heads, 1, *extra_shape[1:])
        x = x.expand(b, l, self._num_kv_heads, self.num_groups, *extra_shape[1:])
        return x.reshape(b, l, self.num_heads, *extra_shape[1:])

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        """Forward pass - matches JAX __call__."""
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
            q_f32 = q.float()
            k_f32 = k.float()
            q_ms = q_f32.pow(2).mean(dim=-1, keepdim=True)
            k_ms = k_f32.pow(2).mean(dim=-1, keepdim=True)
            q = (q_f32 * torch.rsqrt(q_ms + self.qk_norm_eps)).to(q.dtype)
            k = (k_f32 * torch.rsqrt(k_ms + self.qk_norm_eps)).to(k.dtype)

        # Save K, V before repeat for KV cache (non-repeated)
        k_for_cache = k
        v_for_cache = v

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum("blhd,bmhd->bhlm", q, k) * scale

        causal_mask = torch.tril(torch.ones(l, l, dtype=torch.bool, device=x.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        large_neg = -1e4
        if mask is not None:
            key_mask = mask.to(scores.dtype).unsqueeze(1).unsqueeze(2)
            scores = scores + (1.0 - key_mask) * large_neg

        scores = torch.where(causal_mask, scores, torch.full_like(scores, large_neg))
        attn = F.softmax(scores.float(), dim=-1).to(v.dtype)

        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        out = torch.einsum("bhql,blhd->bqhd", attn, v)
        out = out.reshape(b, l, self.d_model)
        output = self.out_proj(out)

        if return_state:
            return output, (k_for_cache, v_for_cache)
        return output

    def step(
        self,
        x_t: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        pos: torch.Tensor,
        cos_t: torch.Tensor,
        sin_t: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """O(L) autoregressive decoding step - matches JAX step.

        Args:
            seq_len: If provided, only attend to first seq_len positions of KV cache.
                     Used for power-of-2 CUDA graph optimization.
        """
        b = x_t.shape[0]

        # Project query for current token
        q = self.q_proj(x_t).reshape(b, 1, self.num_heads, self.head_dim)
        k_new = self.k_proj(x_t).reshape(b, 1, self._num_kv_heads, self.head_dim)
        v_new = self.v_proj(x_t).reshape(b, 1, self._num_kv_heads, self.head_dim)

        # Apply RoPE
        q = apply_rope(q, cos_t, sin_t)
        if self._num_kv_heads < self.num_heads:
            cos_kv = cos_t[:, :, : self._num_kv_heads, :]
            sin_kv = sin_t[:, :, : self._num_kv_heads, :]
            k_new = apply_rope(k_new, cos_kv, sin_kv)
        else:
            k_new = apply_rope(k_new, cos_t, sin_t)

        if self.qk_norm:
            q_f32 = q.float()
            k_f32 = k_new.float()
            q_ms = q_f32.pow(2).mean(dim=-1, keepdim=True)
            k_ms = k_f32.pow(2).mean(dim=-1, keepdim=True)
            q = (q_f32 * torch.rsqrt(q_ms + self.qk_norm_eps)).to(q.dtype)
            k_new = (k_f32 * torch.rsqrt(k_ms + self.qk_norm_eps)).to(k_new.dtype)

        # Update KV cache
        k_cache, v_cache = kv_cache

        # Use index_copy_ for CUDA graph compatibility (no dynamic slicing)
        pos_idx = pos[0:1].long()
        k_cache.index_copy_(1, pos_idx, k_new.to(k_cache.dtype))
        v_cache.index_copy_(1, pos_idx, v_new.to(v_cache.dtype))

        # Use seq_len slice if provided (power-of-2 optimization), else full cache
        if seq_len is not None:
            k_slice = k_cache[:, :seq_len, :, :]
            v_slice = v_cache[:, :seq_len, :, :]
            L = seq_len
        else:
            k_slice = k_cache
            v_slice = v_cache
            L = k_cache.shape[1]

        k_repeated = self._repeat_kv(k_slice)
        v_repeated = self._repeat_kv(v_slice)

        # Create mask for positions > current pos within the slice
        all_pos = torch.arange(L, device=k_cache.device)
        mask = all_pos.view(1, 1, 1, L) > pos[0:1].view(b, 1, 1, 1)

        # Attention with masking
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k_repeated) * scale
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", attn, v_repeated)
        out = out.reshape(b, self.d_model)

        return self.out_proj(out), (k_cache, v_cache)


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block - matches JAX TransformerDecoderBlock exactly."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model, epsilon=norm_eps)
        self.attn = RotarySelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
        )

        self.norm2 = RMSNorm(d_model, epsilon=norm_eps)
        # Match JAX: single ff_in that gets split
        self.ff_in = nn.Linear(d_model, 2 * d_ff, bias=True)
        self.ff_out = nn.Linear(d_ff, d_model, bias=True)

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        """Forward pass - matches JAX __call__ exactly."""
        y = self.norm1(x)
        if return_state:
            y, kv_state = self.attn(y, cos=cos, sin=sin, mask=mask, return_state=True)
        else:
            y = self.attn(y, cos=cos, sin=sin, mask=mask)
        if self.dropout > 0 and self.training:
            y = F.dropout(y, p=self.dropout)
        x = x + y

        y = self.norm2(x)
        u, v = self.ff_in(y).chunk(2, dim=-1)
        y = F.silu(v) * u
        y = self.ff_out(y)
        if self.dropout > 0 and self.training:
            y = F.dropout(y, p=self.dropout)
        output = x + y

        if return_state:
            return output, kv_state
        return output

    def step(
        self,
        x_t: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        pos: torch.Tensor,
        cos_t: torch.Tensor,
        sin_t: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """O(L) autoregressive decoding step - matches JAX step exactly."""
        # Attention with KV cache
        y = self.norm1(x_t)
        y, new_kv_cache = self.attn.step(
            y, kv_cache, pos, cos_t, sin_t, seq_len=seq_len
        )
        x_t = x_t + y

        # FFN - match JAX exactly
        y = self.norm2(x_t)
        u, v = self.ff_in(y).chunk(2, dim=-1)
        y = F.silu(v) * u  # Match JAX: swish(v) * u
        y = self.ff_out(y)

        return x_t + y, new_kv_cache

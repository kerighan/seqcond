# mamba2_jax/ssd.py

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from einops import rearrange, repeat


def _pad_seq_dim(x: jnp.ndarray, pad_size: int) -> jnp.ndarray:
    """Pad zeros at the end of the sequence dimension (axis=1)."""
    if pad_size == 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[1] = (0, pad_size)
    return jnp.pad(x, pad_width, mode="constant", constant_values=0.0)


def segsum(x: jnp.ndarray) -> jnp.ndarray:
    """
    Stable segment sum calculation using broadcasting.
    Computes sum_{k=j+1}^i x_k.
    """
    T = x.shape[-1]

    # Cumulative sum
    cs = jnp.cumsum(x, axis=-1)

    # Difference: cs[i] - cs[j] = sum_{k=j+1}^i x_k
    # Shape (..., T, 1) - (..., 1, T) -> (..., T, T)
    # Using float32 for stability
    cs = cs.astype(jnp.float32)

    diff = cs[..., :, None] - cs[..., None, :]

    # Mask: keep lower triangle (including diagonal)
    # For upper triangle, we want the result to be -inf so exp() gives 0
    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)

    # Use a large negative value that won't cause issues
    # When we take exp() of this, we want it to be effectively 0
    # -100 gives exp(-100) ≈ 3.7e-44, which is safe
    neg_large = jnp.array(-100.0, dtype=diff.dtype)

    return jnp.where(mask, diff, neg_large)


def ssd_naive(
    x: jnp.ndarray,  # (B, L, H, P)
    dt: jnp.ndarray,  # (B, L, H)
    A: jnp.ndarray,  # (H,)
    B_mat: jnp.ndarray,  # (B, L, H, N)
    C_mat: jnp.ndarray,  # (B, L, H, N)
    chunk_size: int,
    D: jnp.ndarray,  # (H,)
    dt_bias: jnp.ndarray,  # (H,)
    dt_min: float,
    dt_max: float,
    initial_states: Optional[jnp.ndarray] = None,  # (B, 1, H, P, N)
    return_final_states: bool = False,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Args:
        x:   (batch_size, seq_len, num_heads, head_dim)
        dt:  (batch_size, seq_len, num_heads)
        A:   (num_heads)
        B_mat, C_mat: (batch_size, seq_len, num_heads, ssm_state_size)

    Returns:
        y:          (batch_size, seq_len, num_heads, head_dim)
        final_state (optional): (batch_size, num_heads, head_dim, ssm_state_size)
    """
    # Force float32 for SSM stability
    dtype_in = x.dtype
    x = x.astype(jnp.float32)
    dt = dt.astype(jnp.float32)
    A = A.astype(jnp.float32)
    B_mat = B_mat.astype(jnp.float32)
    C_mat = C_mat.astype(jnp.float32)
    if initial_states is not None:
        initial_states = initial_states.astype(jnp.float32)

    B_size, seq_len, num_heads, head_dim = x.shape

    # Padding size for equal-sized chunks
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    # dt softplus and clamping (with bias)
    # Add safety: clip input to softplus to avoid overflow
    dt_input = jnp.clip(dt + dt_bias, -20.0, 20.0)
    dt = jax.nn.softplus(dt_input)  # broadcast over batch/seq
    dt = jnp.clip(dt, dt_min, dt_max)

    # Additional safety: ensure dt is finite and positive
    dt = jnp.where(jnp.isfinite(dt), dt, dt_min)

    # Pad tensors along sequence dimension
    x_padded = _pad_seq_dim(x, pad_size)  # (B, L_pad, H, P)
    dt_padded = _pad_seq_dim(dt, pad_size)  # (B, L_pad, H)
    B_padded = _pad_seq_dim(B_mat, pad_size)  # (B, L_pad, H, N)
    C_padded = _pad_seq_dim(C_mat, pad_size)  # (B, L_pad, H, N)

    L_pad = x_padded.shape[1]

    # D residual
    D_residual = (
        D.reshape(1, 1, num_heads, 1).astype(jnp.float32) * x_padded
    )  # (B, L_pad, H, P)

    # Discretize x and A
    x_disc = x_padded * dt_padded[..., None]  # (B, L_pad, H, P)
    A_disc = A.astype(x_disc.dtype) * dt_padded  # (B, L_pad, H)

    # Safety: clip A_disc to prevent extreme exponentials
    # A is negative, so A_disc is negative. Clip to prevent underflow in exp
    A_disc = jnp.clip(A_disc, -20.0, 0.0)

    # Chunk everything
    def pad_and_chunk(t):
        return rearrange(t, "b (c l) ... -> b c l ...", l=chunk_size)

    x_blk = pad_and_chunk(x_disc)  # (B, Ck, Lc, H, P)
    A_blk = pad_and_chunk(A_disc)  # (B, Ck, Lc, H)
    B_blk = pad_and_chunk(B_padded)  # (B, Ck, Lc, H, N)
    C_blk = pad_and_chunk(C_padded)  # (B, Ck, Lc, H, N)

    # A_cumsum over intra-chunk time dimension
    A_blk2 = rearrange(A_blk, "b c l h -> b h c l")  # (B, H, Ck, Lc)
    A_cumsum = jnp.cumsum(A_blk2, axis=-1)  # (B, H, Ck, Lc)

    # 1. Intra-chunk (diagonal blocks)
    # Compute segsum and clip before exp to prevent overflow/underflow
    segsum_result = segsum(A_blk2)  # (B, H, Ck, Lc, Lc)
    segsum_clipped = jnp.clip(segsum_result, -50.0, 10.0)
    L_mat = jnp.exp(segsum_clipped)  # (B, H, Ck, Lc, Lc)

    # C_blk: (B, Ck, Lc, H, N) -> b c l h n
    # B_blk: (B, Ck, Lc, H, N) -> b c s h n
    # L_mat: (B, H, Ck, Lc, Lc)-> b h c l s
    # x_blk: (B, Ck, Lc, H, P) -> b c s h p
    Y_diag = jnp.einsum(
        "bclhn,bcshn,bhcls,bcshp->bclhp",
        C_blk,
        B_blk,
        L_mat,
        x_blk,
    )  # (B, Ck, Lc, H, P)

    # 2. States within each chunk
    decay_diff = A_cumsum[..., :, -1:] - A_cumsum
    decay_diff = jnp.clip(decay_diff, -50.0, 10.0)  # Prevent extreme values
    decay_states = jnp.exp(decay_diff)  # (B, H, Ck, Lc)

    states = jnp.einsum(
        "bclhn,bhcl,bclhp->bchpn",
        B_blk,
        decay_states,
        x_blk,
    )  # (B, Ck, H, P, N)

    # 3. Inter-chunk recurrence
    if initial_states is None:
        initial_states = jnp.zeros_like(states[:, :1, ...])  # (B, 1, H, P, N)

    states = jnp.concatenate([initial_states, states], axis=1)  # (B, Ck+1, H, P, N)

    # A at chunk boundaries
    A_end = A_cumsum[..., -1]  # (B, H, Ck)
    A_end_padded = jnp.pad(A_end, ((0, 0), (0, 0), (1, 0)))  # (B, H, Ck+1)

    segsum_chunk = segsum(A_end_padded)
    segsum_chunk = jnp.clip(segsum_chunk, -50.0, 10.0)  # Prevent extreme values
    decay_chunk = jnp.exp(segsum_chunk)  # (B, H, Ck+1, Ck+1)

    new_states = jnp.einsum(
        "bhzc,bchpn->bzhpn",
        decay_chunk,
        states,
    )  # (B, Ck+1, H, P, N)

    states, final_state = (
        new_states[:, :-1, ...],
        new_states[:, -1, ...],
    )  # (B, Ck, H, P, N), (B, H, P, N)

    # 4. Convert states -> outputs
    A_cumsum_clipped = jnp.clip(A_cumsum, -50.0, 10.0)
    state_decay_out = jnp.exp(A_cumsum_clipped)  # (B, H, Ck, Lc)

    Y_off = jnp.einsum(
        "bclhn,bchpn,bhcl->bclhp",
        C_blk,
        states,
        state_decay_out,
    )  # (B, Ck, Lc, H, P)

    y = Y_diag + Y_off  # (B, Ck, Lc, H, P)
    y = rearrange(y, "b c l h p -> b (c l) h p")  # (B, L_pad, H, P)
    y = y + D_residual  # add residual

    # Remove padding
    if pad_size > 0:
        y = y[:, :seq_len, :, :]

    if return_final_states:
        return y.astype(dtype_in), final_state.astype(dtype_in)
    else:
        return y.astype(dtype_in), None

from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


class GatedRMSNorm(nn.Module):
    """RMSNorm with gating on residual (Mamba2 style).

    Applies: x = rmsnorm(x * sigmoid(residual))
    This allows the residual to gate what information passes through.
    """

    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, residual: jnp.ndarray) -> jnp.ndarray:
        hidden_size = x.shape[-1]
        weight = self.param(
            "weight",
            nn.initializers.ones,
            (hidden_size,),
        )

        x = x.astype(jnp.float32)
        res = residual.astype(jnp.float32)

        # Gate with silu
        x = x * jax.nn.silu(res)

        # RMSNorm
        variance = jnp.mean(x**2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.epsilon)
        x = x * weight

        return x.astype(self.dtype)


class SeqCondAttention(nn.Module):
    # Dimensions Architecture
    num_heads: int = 12  # K
    num_query_heads: int = 6  # K'
    num_anchor_heads: int = 0
    num_thetas: int = 1  # M

    # Paramètres Locaux
    conv_kernel_size: int = 4
    expand_factor: int = 1  # Input Slim (Scan Rapide)
    out_expand_factor: int = 3  # Output Fat (Cerveau SwiGLU) - Ajustable selon VRAM
    skip_low_rank: bool = (
        True  # If True, use D//4 latent skip; if False, use full dim_swiglu_total
    )

    dropout: float = 0.0
    maxlen: Optional[int] = None
    chunk_size: int = 0

    use_square_matrix: bool = False

    compute_dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert self.num_heads % self.num_query_heads == 0
        self.n_rep = self.num_heads // self.num_query_heads

        self.K = self.num_heads
        self.K_q = self.num_query_heads
        self.M = self.num_thetas
        self.num_decay_heads = self.K - self.num_anchor_heads

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        # ======================================================================
        # 0. SETUP DIMENSIONS
        # ======================================================================
        B, L, D = x.shape
        d_inner = int(D * self.expand_factor)

        # Iso-Param H scaling
        H = max(1, d_inner // (self.K * self.M))

        dim_memory = self.K * H

        dim_query_head = H * self.M * 2
        dim_query_total = self.K_q * dim_query_head

        dim_expand = H * self.out_expand_factor
        dim_swiglu_head = dim_expand * 2
        dim_swiglu_total = self.K * dim_swiglu_head

        # ======================================================================
        # 1. FUSED INPUT PROJECTION (Mamba-style single projection + conv)
        # ======================================================================
        # Only project conv_branch (mem + query), gate comes from x directly
        dim_mem_total = dim_memory + self.K  # k_val + s_raw
        dim_conv_total = dim_mem_total + dim_query_total

        z_conv = nn.Dense(dim_conv_total, use_bias=False, name="in_proj")(x)
        z_conv = z_conv.astype(self.compute_dtype)

        # Single fused conv on mem+query (Mamba style)
        z_conv = nn.Conv(
            features=z_conv.shape[-1],
            kernel_size=(self.conv_kernel_size,),
            padding=((self.conv_kernel_size - 1, 0),),
            feature_group_count=z_conv.shape[-1],
            use_bias=False,
            name="conv",
        )(z_conv)
        z_conv = jax.nn.silu(z_conv)

        # Split conv output into memory and query
        z_mem = z_conv[..., :dim_mem_total]
        q_raw = z_conv[..., dim_mem_total:]

        # Process memory branch
        k_val = z_mem[..., :dim_memory].reshape(B, L, self.K, H)
        # k_val = nn.RMSNorm(dtype=self.compute_dtype, name="k_norm")(k_val)
        s_raw = z_mem[..., dim_memory:]

        if mask is not None:
            m = mask.astype(x.dtype)[:, :, None, None]
            s_raw = s_raw * mask.astype(x.dtype)[..., None]
            k_val = k_val * m

        # Process query branch
        # q_raw = nn.RMSNorm(dtype=self.compute_dtype, name="q_norm")(q_raw)
        q_raw = q_raw.reshape(B, L, self.K_q, 1, H, self.M, 2)
        q_re, q_im = q_raw[..., 0], q_raw[..., 1]

        # ======================================================================
        # 2. GRILLE SPECTRALE (THETA & W_INT)
        # ======================================================================
        # theta_min=0.001, theta_max=3.0 (hardcoded to avoid closure capture)
        if self.M == 1:
            # Use lambda with shape to avoid closure capture
            theta_raw = self.param(
                "theta_raw",
                lambda key, shape: jnp.array(
                    np.log(
                        np.clip(
                            (
                                np.tile(
                                    np.geomspace(0.001, 3.0, shape[2]).reshape(
                                        1, 1, shape[2], 1, 1
                                    ),
                                    (1, 1, 1, shape[3], 1),
                                )
                                - 0.001
                            )
                            / 2.999,
                            1e-4,
                            1 - 1e-4,
                        )
                    )
                    - np.log(
                        1
                        - np.clip(
                            (
                                np.tile(
                                    np.geomspace(0.001, 3.0, shape[2]).reshape(
                                        1, 1, shape[2], 1, 1
                                    ),
                                    (1, 1, 1, shape[3], 1),
                                )
                                - 0.001
                            )
                            / 2.999,
                            1e-4,
                            1 - 1e-4,
                        )
                    ),
                    dtype=jnp.float32,
                ),
                (1, 1, self.K, H, 1),
            )
            theta = 0.001 + 2.999 * jax.nn.sigmoid(theta_raw).astype(jnp.float32)
        else:
            # Use lambda with shape to avoid closure capture
            theta_d_raw = self.param(
                "theta_d_raw",
                lambda key, shape: jnp.log(
                    jnp.exp(
                        jnp.tile(
                            jnp.array(np.geomspace(0.001, 3.0, shape[4])).reshape(
                                1, 1, 1, 1, shape[4]
                            ),
                            (1, 1, shape[2], shape[3], 1),
                        )
                    )
                    - 1.0
                    + 1e-4
                ).astype(jnp.float32),
                (1, 1, self.K, H, self.M),
            )

            theta_d = jax.nn.softplus(theta_d_raw).astype(jnp.float32) + 1e-4
            theta_accum = jnp.cumsum(theta_d, axis=-1)

            scale_range = 2.999  # theta_max - theta_min
            total_sum = theta_accum[..., -1:]
            theta = 0.001 + (theta_accum / total_sum) * scale_range

        # Learnable integration weights (independent of theta)
        w_int_raw = self.param(
            "w_int_raw", nn.initializers.ones, (1, 1, self.K_q, self.n_rep, H, self.M)
        )
        w_int = jnp.exp(w_int_raw).astype(jnp.float32)
        w_int /= jnp.sum(w_int, axis=-1, keepdims=True) + 1e-6

        # ======================================================================
        # 3. MODULATION & SCAN UNIFIÉ (MATRIX vs LINEAR)
        # ======================================================================
        # Decay (Log-Space)
        pos = jnp.arange(L, dtype=jnp.float32)
        log_w_list = []
        if self.num_decay_heads > 0:
            d_slopes = self.param(
                "decay_slopes",
                lambda r, s: jnp.log(jnp.exp(np.geomspace(0.001, 0.1, s[0])) - 1),
                (self.num_decay_heads,),
            )
            slopes = jax.nn.softplus(d_slopes).reshape(1, 1, -1)
            dist = jnp.maximum(jnp.float32((self.maxlen or L) - 1) - pos, 0.0)
            log_w_list.append(-slopes * dist[None, :, None])
        if self.num_anchor_heads > 0:
            a_slopes = self.param(
                "anchor_slopes",
                lambda r, s: jnp.log(jnp.exp(np.geomspace(0.01, 0.1, s[0])) - 1),
                (self.num_anchor_heads,),
            )
            slopes_a = jax.nn.softplus(a_slopes).reshape(1, 1, -1)
            log_w_list.append(-slopes_a * pos[None, :, None])

        log_time_weight = (
            jnp.concatenate(log_w_list, axis=2)
            if log_w_list
            else jnp.zeros((1, L, self.K), dtype=jnp.float32)
        )
        score_scale = self.param("score_scale", nn.initializers.ones, (self.K,))
        score_bias = self.param("score_bias", nn.initializers.zeros, (self.K,))
        score_raw = (
            score_scale[None, None, :] * s_raw.astype(jnp.float32)
            + score_bias[None, None, :]
        )

        # Softplus instead of exp for stability (Mamba-like)
        p_w_content = jax.nn.softplus(score_raw)

        # Temporal weight with exp
        temporal_weight = jnp.exp(log_time_weight)

        # Combine and clip for safety
        p_w = p_w_content * temporal_weight
        p_w = jnp.clip(p_w, 1e-6, 1000.0)  # (B, L, K)

        # Modulation (with softsign to bound phase - smoother than tanh)
        k_f32 = k_val.astype(jnp.float32)[..., None]
        p_w_b = p_w[..., None, None]

        phase_scale = self.param("phase_scale", nn.initializers.ones, (self.K,))
        phase_scale_b = phase_scale[None, None, :, None, None]
        k_scaled = k_f32 * phase_scale_b
        phi = (k_scaled / (1.0 + jnp.abs(k_scaled))) * theta  # softsign modulation
        kvw = k_f32 * p_w_b

        # re, im: (B, L, K, H, M)
        re = kvw * jnp.cos(phi)
        im = kvw * jnp.sin(phi)

        # --- DYNAMIC SWITCH ---
        # use_square_matrix = L <= self.matrix_threshold
        if self.use_square_matrix:
            # PATH 1: O(L^2) Matrix Multiply (Tensor Core Optimized)
            # Causal Mask (L, L) - use fp32 to avoid bf16 accumulation issues
            mask_idx = jnp.arange(L)[:, None] >= jnp.arange(L)[None, :]
            causal_mask = mask_idx.astype(jnp.float32)

            # Stack RE/IM for single einsum - accumulate in fp32
            # (B, L, K, H, M, 2)
            stack_ri = jnp.stack([re, im], axis=-1).astype(jnp.float32)
            # acc_ri: (B, L, K, H, M, 2)
            acc_ri = jnp.einsum("ts,bskhmc->btkhmc", causal_mask, stack_ri)

            re_acc = acc_ri[..., 0]
            im_acc = acc_ri[..., 1]

        else:
            # PATH 2: O(L) Linear Scan (Memory Optimized)
            # Use fp32 for accumulation to avoid precision issues
            flat_size = self.K * H * self.M
            re_flat = re.reshape(B, L, flat_size).astype(jnp.float32)
            im_flat = im.reshape(B, L, flat_size).astype(jnp.float32)

            # Stack & Scan in fp32
            stack = jnp.concatenate([re_flat, im_flat], axis=-1)
            cumsum = jnp.cumsum(stack, axis=1)

            # Unpack
            re_acc_flat, im_acc_flat = jnp.split(cumsum, [flat_size], axis=-1)
            re_acc = re_acc_flat.reshape(B, L, self.K, H, self.M)
            im_acc = im_acc_flat.reshape(B, L, self.K, H, self.M)

        # Normalisation removed to preserve energy
        # inv_den = 1.0 / jnp.maximum(den_acc, 1e-4)
        # inv_den = inv_den[..., None, None]  # (B, L, K, 1, 1)

        state_re = re_acc
        state_im = im_acc

        # ======================================================================
        # 4. READOUT & GQA & INTEGRATION
        # ======================================================================

        # Normalize states before readout to prevent explosion
        state_re_g = state_re.reshape(B, L, self.K_q, self.n_rep, H, self.M)
        state_im_g = state_im.reshape(B, L, self.K_q, self.n_rep, H, self.M)

        scale = jax.lax.rsqrt(jnp.array(H, dtype=jnp.float32))
        match_re = (state_re_g * q_re + state_im_g * q_im) * scale
        match_im = (state_im_g * q_re - state_re_g * q_im) * scale

        out_re_g = jnp.sum(match_re * w_int, axis=-1)
        out_im_g = jnp.sum(match_im * w_int, axis=-1)

        # Normalize before fusion
        out_re = out_re_g.reshape(B, L, self.K, H).astype(self.compute_dtype)
        out_im = out_im_g.reshape(B, L, self.K, H).astype(self.compute_dtype)

        out_complex = jnp.concatenate([out_re, out_im], axis=-1)

        # ======================================================================
        # 5. FUSION FINALE with GatedRMSNorm + SwiGLU (no skip)
        # ======================================================================
        out_complex_flat = out_complex.reshape(B, L, -1)

        # GatedRMSNorm with gate from x
        gate_for_norm = nn.Dense(
            out_complex_flat.shape[-1], use_bias=False, name="gate_proj"
        )(x)
        out_normed = GatedRMSNorm(dtype=self.compute_dtype, name="gated_norm")(
            out_complex_flat, gate_for_norm
        )
        out_complex = out_normed.reshape(B, L, self.K, 2 * H)

        # Readout to SwiGLU dimension
        W_readout = self.param(
            "W_readout",
            nn.initializers.glorot_uniform(),
            (self.K, 2 * H, dim_swiglu_head),
        )
        y_spec_raw = jnp.einsum("blkf,kfn->blkn", out_complex, W_readout)

        # SwiGLU activation (no skip/highway)
        y_val, y_gate = jnp.split(y_spec_raw, 2, axis=-1)
        y_act = y_val * jax.nn.sigmoid(y_gate)

        # Output projection
        y_flat = y_act.reshape(B, L, -1)
        out = nn.Dense(D, use_bias=False, name="out_proj")(y_flat)

        if self.dropout > 0:
            out = nn.Dropout(self.dropout)(out, deterministic=deterministic)

        return out

    @nn.compact
    def step(self, x_t, state, deterministic=True):
        """
        O(1) autoregressive decoding step for seqcond_fast.

        Args:
            x_t: Input token embedding (B, D)
            state: Tuple of (den_acc, re_acc, im_acc, pos, conv_buffer) where:
                - den_acc: (B, K) - accumulated denominator
                - re_acc: (B, K, H, M) - accumulated real part
                - im_acc: (B, K, H, M) - accumulated imaginary part
                - pos: (B,) - current position
                - conv_buffer: (B, conv_kernel_size-1, dim_conv_total) - fused conv history
            deterministic: Whether to use dropout

        Returns:
            out: (B, D) - output for this step
            new_state: Updated state tuple
        """
        B, D = x_t.shape
        d_inner = int(D * self.expand_factor)

        H = max(1, d_inner // (self.K * self.M))
        dim_memory = self.K * H

        dim_query_head = H * self.M * 2
        dim_query_total = self.K_q * dim_query_head

        dim_expand = H * self.out_expand_factor
        dim_swiglu_head = dim_expand * 2
        dim_swiglu_total = self.K * dim_swiglu_head

        # Unpack state (now with single fused conv buffer)
        den_acc, re_acc, im_acc, pos, conv_buffer = state

        # ======================================================================
        # FUSED INPUT PROJECTION + CONV (matches __call__)
        # ======================================================================
        dim_mem_total = dim_memory + self.K
        dim_conv_total = dim_mem_total + dim_query_total

        z_conv = nn.Dense(dim_conv_total, use_bias=False, name="in_proj")(x_t)
        z_conv = z_conv.astype(self.compute_dtype)

        # Single fused conv with buffer
        z_conv_expanded = z_conv[:, None, :]
        conv_input = jnp.concatenate([conv_buffer, z_conv_expanded], axis=1)

        z_conv_out = nn.Conv(
            features=conv_input.shape[-1],
            kernel_size=(self.conv_kernel_size,),
            padding="VALID",
            feature_group_count=conv_input.shape[-1],
            use_bias=False,
            name="conv",
        )(conv_input)

        z_conv = jax.nn.silu(z_conv_out[:, 0, :])
        conv_buffer_new = jnp.concatenate(
            [conv_buffer[:, 1:, :], z_conv_expanded], axis=1
        )

        # Split conv output into memory and query
        z_mem = z_conv[..., :dim_mem_total]
        q_raw = z_conv[..., dim_mem_total:]

        # Process memory branch
        k_val = z_mem[..., :dim_memory].reshape(B, self.K, H)
        # k_val = nn.RMSNorm(dtype=self.compute_dtype, name="k_norm")(k_val)
        s_raw = z_mem[..., dim_memory:]

        # Process query branch (matches __call__)
        # q_raw = nn.RMSNorm(dtype=self.compute_dtype, name="q_norm")(q_raw)
        q_raw = q_raw.reshape(B, self.K_q, 1, H, self.M, 2)
        q_re, q_im = q_raw[..., 0], q_raw[..., 1]  # (B, K_q, 1, H, M)

        # Theta grid (theta_min=0.001, theta_max=3.0 hardcoded to avoid closure capture)
        if self.M == 1:
            # Use lambda with shape to avoid closure capture
            theta_raw = self.param(
                "theta_raw",
                lambda key, shape: jnp.array(
                    np.log(
                        np.clip(
                            (
                                np.tile(
                                    np.geomspace(0.001, 3.0, shape[2]).reshape(
                                        1, 1, shape[2], 1, 1
                                    ),
                                    (1, 1, 1, shape[3], 1),
                                )
                                - 0.001
                            )
                            / 2.999,
                            1e-4,
                            1 - 1e-4,
                        )
                    )
                    - np.log(
                        1
                        - np.clip(
                            (
                                np.tile(
                                    np.geomspace(0.001, 3.0, shape[2]).reshape(
                                        1, 1, shape[2], 1, 1
                                    ),
                                    (1, 1, 1, shape[3], 1),
                                )
                                - 0.001
                            )
                            / 2.999,
                            1e-4,
                            1 - 1e-4,
                        )
                    ),
                    dtype=jnp.float32,
                ),
                (1, 1, self.K, H, 1),
            )
            theta = 0.001 + 2.999 * jax.nn.sigmoid(theta_raw).astype(jnp.float32)
        else:
            # Use lambda with shape to avoid closure capture
            theta_d_raw = self.param(
                "theta_d_raw",
                lambda key, shape: jnp.log(
                    jnp.exp(
                        jnp.tile(
                            jnp.array(np.geomspace(0.001, 3.0, shape[4])).reshape(
                                1, 1, 1, 1, shape[4]
                            ),
                            (1, 1, shape[2], shape[3], 1),
                        )
                    )
                    - 1.0
                    + 1e-4
                ).astype(jnp.float32),
                (1, 1, self.K, H, self.M),
            )
            theta_d = jax.nn.softplus(theta_d_raw).astype(jnp.float32) + 1e-4
            theta_accum = jnp.cumsum(theta_d, axis=-1)

            scale_range = 2.999  # theta_max - theta_min
            total_sum = theta_accum[..., -1:]
            theta = 0.001 + (theta_accum / total_sum) * scale_range

            # Compute w_int from theta_accum (reuse for later)
            dtheta_raw = theta_accum[..., 1:] - theta_accum[..., :-1]
            dtheta = dtheta_raw * (scale_range / total_sum)
            w0 = dtheta[..., :1] * 0.5
            w_mid = 0.5 * (dtheta[..., :-1] + dtheta[..., 1:])
            wL = dtheta[..., -1:] * 0.5
            w_int_m = jnp.concatenate([w0, w_mid, wL], axis=-1)
            w_int_m = w_int_m.reshape(1, 1, self.K_q, self.n_rep, H, self.M)

        theta = theta[0, 0]  # (K, H, M)

        # Decay weights
        log_w_list = []
        if self.num_decay_heads > 0:
            d_slopes = self.param(
                "decay_slopes",
                lambda r, s: jnp.log(jnp.exp(np.geomspace(0.001, 0.1, s[0])) - 1),
                (self.num_decay_heads,),
            )
            slopes = jax.nn.softplus(d_slopes).reshape(1, -1)
            dist = jnp.maximum(
                jnp.float32((self.maxlen or 2048) - 1) - pos[:, None], 0.0
            )
            log_w_list.append(-slopes * dist)
        if self.num_anchor_heads > 0:
            a_slopes = self.param(
                "anchor_slopes",
                lambda r, s: jnp.log(jnp.exp(np.geomspace(0.01, 0.1, s[0])) - 1),
                (self.num_anchor_heads,),
            )
            slopes_a = jax.nn.softplus(a_slopes).reshape(1, -1)
            log_w_list.append(-slopes_a * pos[:, None])

        log_time_weight = (
            jnp.concatenate(log_w_list, axis=1)
            if log_w_list
            else jnp.zeros((B, self.K), dtype=jnp.float32)
        )

        score_scale = self.param("score_scale", nn.initializers.ones, (self.K,))
        score_bias = self.param("score_bias", nn.initializers.zeros, (self.K,))
        score_raw = (
            score_scale[None, :] * s_raw.astype(jnp.float32) + score_bias[None, :]
        )
        # score_raw = jnp.clip(score_raw, -20.0, 20.0)

        # Softplus instead of exp for stability
        p_w_content = jax.nn.softplus(score_raw)
        temporal_weight = jnp.exp(log_time_weight)
        p_w = p_w_content * temporal_weight
        p_w = jnp.clip(p_w, 1e-6, 1000.0)

        # Modulation
        k_f32 = k_val[..., None].astype(jnp.float32)  # (B, K, H, 1)
        p_w_b = p_w[..., None, None]  # (B, K, 1, 1)

        phase_scale = self.param("phase_scale", nn.initializers.ones, (self.K,))
        phase_scale_b = phase_scale.reshape(1, self.K, 1, 1)
        k_scaled = k_f32 * phase_scale_b
        phi = (k_scaled / (1.0 + jnp.abs(k_scaled))) * theta  # softsign (B, K, H, M)
        kvw = k_f32 * p_w_b

        re = kvw * jnp.cos(phi)
        im = kvw * jnp.sin(phi)

        # Update accumulation
        den_acc_new = den_acc + p_w
        re_acc_new = re_acc + re
        im_acc_new = im_acc + im

        # Normalize by denominator
        # inv_den = 1.0 / jnp.maximum(den_acc_new, 1e-4)
        # inv_den = inv_den[..., None, None]  # (B, K, 1, 1)

        state_re = re_acc_new
        state_im = im_acc_new

        # Group state for GQA: (B, K, H, M) -> (B, K_q, n_rep, H, M)
        state_re_grouped = state_re.reshape(B, self.K_q, self.n_rep, H, self.M)
        state_im_grouped = state_im.reshape(B, self.K_q, self.n_rep, H, self.M)

        # w_int for integration (must match __call__)
        w_int_raw = self.param(
            "w_int_raw", nn.initializers.ones, (1, 1, self.K_q, self.n_rep, H, self.M)
        )
        w_int = jnp.exp(w_int_raw).astype(jnp.float32)
        w_int = w_int / (jnp.sum(w_int, axis=-1, keepdims=True) + 1e-6)

        # Remove batch/seq dims from w_int for step
        w_int_step = w_int[0, 0]  # (K_q, n_rep, H, M)

        # Compute complex multiplication (matching) with 1/sqrt(H) scale
        # q_re, q_im: (B, K_q, 1, H, M) from reshape at line 461
        # state_re_grouped, state_im_grouped: (B, K_q, n_rep, H, M)
        scale = jax.lax.rsqrt(jnp.array(H, dtype=jnp.float32))  # = 1/sqrt(H)
        match_re = (state_re_grouped * q_re + state_im_grouped * q_im) * scale
        match_im = (state_im_grouped * q_re - state_re_grouped * q_im) * scale

        # Integrate over M (frequencies) with w_int
        out_re_g = jnp.sum(match_re * w_int_step, axis=-1)  # (B, K_q, n_rep, H)
        out_im_g = jnp.sum(match_im * w_int_step, axis=-1)

        # Reshape to (B, K, H)
        out_re = out_re_g.reshape(B, self.K, H).astype(self.compute_dtype)
        out_im = out_im_g.reshape(B, self.K, H).astype(self.compute_dtype)

        # Concatenate (not stack!) to match __call__
        out_complex = jnp.concatenate([out_re, out_im], axis=-1)  # (B, K, 2*H)

        # GatedRMSNorm with gate from x_t (matches __call__)
        out_complex_flat = out_complex.reshape(B, -1)
        gate_for_norm = nn.Dense(
            out_complex_flat.shape[-1], use_bias=False, name="gate_proj"
        )(x_t)
        out_normed = GatedRMSNorm(dtype=self.compute_dtype, name="gated_norm")(
            out_complex_flat, gate_for_norm
        )
        out_complex = out_normed.reshape(B, self.K, 2 * H)

        # Readout to SwiGLU dimension (matches __call__)
        W_readout = self.param(
            "W_readout",
            nn.initializers.glorot_uniform(),
            (self.K, 2 * H, dim_swiglu_head),
        )
        y_spec_raw = jnp.einsum("bkf,kfn->bkn", out_complex, W_readout)

        # SwiGLU activation (no skip/highway)
        y_val, y_gate = jnp.split(y_spec_raw, 2, axis=-1)
        y_act = y_val * jax.nn.sigmoid(y_gate)

        # Output projection
        y_flat = y_act.reshape(B, -1)
        out = nn.Dense(D, use_bias=False, name="out_proj")(y_flat)

        if self.dropout > 0:
            out = nn.Dropout(self.dropout)(out, deterministic=deterministic)

        # Update position
        pos_new = pos + 1

        return out, (
            den_acc_new,
            re_acc_new,
            im_acc_new,
            pos_new,
            conv_buffer_new,
        )


class SeqCondBlock(nn.Module):
    num_heads: int = 32
    num_query_heads: int = 6
    expand_factor: float = 1.0
    out_expand_factor: int = 3  # SwiGLU expansion factor
    num_thetas: int = 1
    num_anchor_heads: int = 0
    conv_kernel_size: int = 4
    skip_low_rank: bool = False  # If True, use D//4 latent skip; if False, use full dim
    dropout: float = 0.0
    norm_eps: float = 1e-5
    maxlen: Optional[int] = None
    derivative_order: Optional[int] = 0
    chunk_size: int = 32
    use_square_matrix: bool = False

    compute_dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        h = nn.RMSNorm(
            epsilon=self.norm_eps,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )(x)
        h = SeqCondAttention(
            num_heads=self.num_heads,
            num_query_heads=self.num_query_heads,
            expand_factor=self.expand_factor,
            out_expand_factor=self.out_expand_factor,
            num_thetas=self.num_thetas,
            num_anchor_heads=self.num_anchor_heads,
            conv_kernel_size=self.conv_kernel_size,
            skip_low_rank=self.skip_low_rank,
            dropout=self.dropout,
            maxlen=self.maxlen,
            chunk_size=self.chunk_size,
            use_square_matrix=self.use_square_matrix,
            compute_dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )(h, mask=mask, deterministic=deterministic)
        return x + h

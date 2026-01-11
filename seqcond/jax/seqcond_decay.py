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

        # Gate with sigmoid (bounded, stable)
        x = x * jax.nn.sigmoid(res)

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

        # Skip dimension: low-rank (D//4) or full (dim_swiglu_total)
        dim_skip = D // 4 if self.skip_low_rank else dim_swiglu_total
        dim_gate = self.K

        # ======================================================================
        # 1. FUSED INPUT PROJECTION (Mamba-style single projection + conv)
        # ======================================================================
        # All inputs in one projection: conv_branch (mem + query) + skip + gate
        dim_mem_total = dim_memory + self.K  # k_val + s_raw
        dim_conv_total = (
            dim_mem_total + dim_query_total
        )  # Everything that goes through conv
        dim_total = dim_conv_total + dim_skip + dim_gate

        z_all = nn.Dense(dim_total, use_bias=False, name="in_proj")(x)
        z_all = z_all.astype(self.compute_dtype)

        # Split: conv_branch vs non-conv branches
        z_conv = z_all[..., :dim_conv_total]
        c_skip = z_all[..., dim_conv_total : dim_conv_total + dim_skip]
        gate_logits = z_all[..., dim_conv_total + dim_skip :]

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
        k_val = nn.RMSNorm(dtype=self.compute_dtype, name="k_norm")(k_val)
        s_raw = z_mem[..., dim_memory:]

        if mask is not None:
            m = mask.astype(x.dtype)[:, :, None, None]
            s_raw = s_raw * mask.astype(x.dtype)[..., None]
            k_val = k_val * m

        # Process query branch
        q_raw = nn.RMSNorm(dtype=self.compute_dtype, name="q_norm")(q_raw)
        q_raw = q_raw.reshape(B, L, self.K_q, 1, H, self.M, 2)
        q_re, q_im = q_raw[..., 0], q_raw[..., 1]

        # ======================================================================
        # 2. GRILLE SPECTRALE (THETA & W_INT)
        # ======================================================================
        theta_min, theta_max = 0.001, 3.0
        if self.M == 1:

            def init_theta_m1(key, shape):
                grid = np.geomspace(theta_min, theta_max, self.K).reshape(
                    1, 1, self.K, 1, 1
                )
                base = np.tile(grid, (1, 1, 1, H, 1))
                # Inverse Softplus (Mamba2 style robust)
                u = (base - theta_min) / max(theta_max - theta_min, 1e-6)
                u = np.clip(u, 1e-4, 1 - 1e-4)
                raw = np.log(u) - np.log(1 - u)
                return jnp.array(raw, dtype=jnp.float32)

            theta_raw = self.param("theta_raw", init_theta_m1, (1, 1, self.K, H, 1))
            theta = theta_min + (theta_max - theta_min) * jax.nn.sigmoid(
                theta_raw
            ).astype(jnp.float32)

            # Poids d'intégration appris (initialisés à 1)
            w_int_raw = self.param(
                "w_int_raw", nn.initializers.zeros, (1, 1, self.K_q, self.n_rep, H, 1)
            )
            w_int_raw_clipped = jnp.clip(w_int_raw, -5.0, 5.0)  # Prevent exp overflow
            w_int = jnp.exp(w_int_raw_clipped).astype(jnp.float32)  # Toujours positif
        else:

            def init_theta_deltas(key, shape):
                grid_m = np.geomspace(theta_min, theta_max, self.M)
                base = np.tile(grid_m.reshape(1, 1, 1, 1, self.M), (1, 1, self.K, H, 1))
                return jnp.log(jnp.exp(base) - 1.0 + 1e-4)

            theta_d_raw = self.param(
                "theta_d_raw", init_theta_deltas, (1, 1, self.K, H, self.M)
            )

            theta_d = jax.nn.softplus(theta_d_raw).astype(jnp.float32) + 1e-4
            theta_accum = jnp.cumsum(theta_d, axis=-1)

            scale_range = theta_max - theta_min
            total_sum = theta_accum[..., -1:]
            theta = theta_min + (theta_accum / total_sum) * scale_range

            dtheta_raw = theta_accum[..., 1:] - theta_accum[..., :-1]
            dtheta = dtheta_raw * (scale_range / total_sum)

            w0 = dtheta[..., :1] * 0.5
            w_mid = 0.5 * (dtheta[..., :-1] + dtheta[..., 1:])
            wL = dtheta[..., -1:] * 0.5

            w_int = jnp.concatenate([w0, w_mid, wL], axis=-1)
            w_int = w_int.reshape(1, 1, self.K_q, self.n_rep, H, self.M)

        # ======================================================================
        # 3. MODULATION & SCAN UNIFIÉ (MATRIX vs LINEAR)
        # ======================================================================
        # dt: content-based "delta time" (like Mamba)
        # Controls BOTH contribution weight AND decay rate
        #
        # Mamba initialization:
        # - dt_bias initialized so softplus(dt_bias) gives dt in [dt_min, dt_max]
        # - dt_min=0.001, dt_max=0.1 (from Mamba config)
        # - Use inverse softplus: dt_bias = log(exp(dt) - 1)
        dt_scale = self.param("dt_scale", nn.initializers.ones, (self.K,))
        dt_bias = self.param(
            "dt_bias",
            lambda k, s: jnp.log(
                jnp.expm1(jnp.array(np.geomspace(0.001, 0.1, s[0])))
            ).astype(jnp.float32),
            (self.K,),
        )
        dt_raw = (
            dt_scale[None, None, :] * s_raw.astype(jnp.float32) + dt_bias[None, None, :]
        )
        dt_raw = jnp.clip(dt_raw, -20.0, 20.0)

        # Softplus for positive dt (like Mamba)
        dt = jax.nn.softplus(dt_raw)
        dt = jnp.clip(dt, 0.001, 0.1)  # (B, L, K) - clamp like Mamba

        # A: base decay rate per head (like Mamba)
        # Mamba uses A in [1, 16], stored as log(A)
        # decay = exp(-A * dt), so:
        # - A=1, dt=0.001 -> decay=0.999 (very slow, ~1000 token memory)
        # - A=16, dt=0.1 -> decay=0.2 (fast, ~3 token memory)
        log_A = self.param(
            "log_A",
            lambda k, s: jnp.log(jnp.array(np.geomspace(1.0, 16.0, s[0]))).astype(
                jnp.float32
            ),
            (self.K,),
        )
        A = -jnp.exp(log_A)  # (K,) - negative for decay

        # A_disc = A * dt (discretized decay, like Mamba)
        A_disc = A[None, None, :] * dt  # (B, L, K)
        A_disc = jnp.clip(A_disc, -20.0, 0.0)  # Prevent extreme values

        # Modulation (with softsign to bound phase - smoother than tanh)
        k_f32 = k_val.astype(jnp.float32)[..., None]
        dt_b = dt[..., None, None]  # (B, L, K, 1, 1)

        phase_scale = self.param("phase_scale", nn.initializers.ones, (self.K,))
        phase_scale_b = phase_scale[None, None, :, None, None]
        k_scaled = k_f32 * phase_scale_b
        phi = (k_scaled / (1.0 + jnp.abs(k_scaled))) * theta  # softsign modulation

        # x_disc = x * dt (discretized input, like Mamba)
        kvw = k_f32 * dt_b

        # re, im: (B, L, K, H, M)
        re = kvw * jnp.cos(phi)
        im = kvw * jnp.sin(phi)

        # ======================================================================
        # 3.5 STATE ACCUMULATION with Mamba-style decay
        # ======================================================================
        # state_new = exp(A * dt) * state_old + x * dt
        # For matrix form: need cumulative product of exp(A_disc)

        # --- DYNAMIC SWITCH ---
        if self.use_square_matrix:
            # PATH 1: O(L^2) Matrix Multiply with content-dependent decay
            # Build decay matrix using cumulative sum of A_disc (like Mamba's segsum)

            # A_disc: (B, L, K) -> cumsum over L
            A_cumsum = jnp.cumsum(A_disc, axis=1)  # (B, L, K)

            # decay_mat[b, t, s, k] = exp(A_cumsum[b, t, k] - A_cumsum[b, s, k])
            # = exp(sum_{i=s+1}^{t} A_disc[b, i, k])
            # This is the decay from position s to position t

            # (B, L, K) -> (B, L, 1, K) - (B, 1, L, K) = (B, L, L, K)
            decay_diff = A_cumsum[:, :, None, :] - A_cumsum[:, None, :, :]

            # Causal mask: only t >= s (upper triangle is future)
            pos_t = jnp.arange(L)[:, None]
            pos_s = jnp.arange(L)[None, :]
            causal_mask = (pos_t >= pos_s).astype(jnp.float32)  # (L, L)

            # Apply causal mask before exp (set future to -inf)
            decay_diff = jnp.where(
                causal_mask[None, :, :, None],
                decay_diff,
                jnp.array(-100.0, dtype=decay_diff.dtype),
            )
            decay_diff = jnp.clip(decay_diff, -50.0, 10.0)

            # decay_mat: (B, L, L, K)
            decay_mat = jnp.exp(decay_diff).astype(self.compute_dtype)

            # Stack RE/IM for single einsum
            # (B, L, K, H, M, 2)
            stack_ri = jnp.stack([re, im], axis=-1).astype(self.compute_dtype)

            # einsum: decay_mat[b, t, s, k] * stack_ri[b, s, k, h, m, c] -> [b, t, k, h, m, c]
            acc_ri = jnp.einsum("btsk,bskhmc->btkhmc", decay_mat, stack_ri)

            re_acc = acc_ri[..., 0]
            im_acc = acc_ri[..., 1]

        else:
            # PATH 2: O(L) Linear Scan with Mamba-style decay
            # state_new = exp(A_disc) * state_old + x_disc
            flat_size = self.K * H * self.M
            re_flat = re.reshape(B, L, flat_size)
            im_flat = im.reshape(B, L, flat_size)

            # Stack for scan
            stack = jnp.concatenate([re_flat, im_flat], axis=-1)  # (B, L, 2*flat_size)

            # exp(A_disc) per position: (B, L, K) -> (B, L, K*H*M) by repeating
            decay_per_pos = jnp.exp(A_disc)  # (B, L, K)
            decay_per_pos = jnp.repeat(
                decay_per_pos, H * self.M, axis=-1
            )  # (B, L, K*H*M)
            decay_per_pos = jnp.concatenate(
                [decay_per_pos, decay_per_pos], axis=-1
            )  # (B, L, 2*K*H*M) for re+im

            def scan_fn(carry, inputs):
                # carry: (B, 2*flat_size) - accumulated state
                # inputs: (x, decay) where x: (B, 2*flat_size), decay: (B, 2*flat_size)
                x, decay = inputs
                new_state = decay * carry + x
                return new_state, new_state

            init_state = jnp.zeros((B, 2 * flat_size), dtype=stack.dtype)
            # Transpose to (L, B, ...) for scan
            stack_t = stack.transpose(1, 0, 2)
            decay_t = decay_per_pos.transpose(1, 0, 2)
            _, acc_stack = jax.lax.scan(scan_fn, init_state, (stack_t, decay_t))
            acc_stack = acc_stack.transpose(1, 0, 2)  # (B, L, 2*flat_size)

            # Unpack
            re_acc_flat, im_acc_flat = jnp.split(acc_stack, [flat_size], axis=-1)
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

        state_re_g = state_re.reshape(B, L, self.K_q, self.n_rep, H, self.M)
        state_im_g = state_im.reshape(B, L, self.K_q, self.n_rep, H, self.M)

        match_re = state_re_g * q_re + state_im_g * q_im
        match_im = state_im_g * q_re - state_re_g * q_im

        out_re_g = jnp.sum(match_re * w_int, axis=-1)
        out_im_g = jnp.sum(match_im * w_int, axis=-1)

        out_re = out_re_g.reshape(B, L, self.K, H).astype(self.compute_dtype)
        out_im = out_im_g.reshape(B, L, self.K, H).astype(self.compute_dtype)

        out_complex = jnp.concatenate([out_re, out_im], axis=-1)

        # ======================================================================
        # 5. FUSION FINALE with GatedRMSNorm
        # ======================================================================
        W_readout = self.param(
            "W_readout",
            nn.initializers.glorot_uniform(),
            (self.K, 2 * H, dim_swiglu_head),
        )

        # Apply GatedRMSNorm to spectral output (Mamba2 style)
        # gate_logits acts as the residual gate
        out_complex_flat = out_complex.reshape(B, L, -1)
        gate_for_norm = nn.Dense(
            out_complex_flat.shape[-1], use_bias=False, name="gate_proj"
        )(gate_logits)
        out_normed = GatedRMSNorm(dtype=self.compute_dtype, name="gated_norm")(
            out_complex_flat, gate_for_norm
        )
        out_complex = out_normed.reshape(B, L, self.K, 2 * H)

        y_spec_raw = jnp.einsum("blkf,kfn->blkn", out_complex, W_readout)

        # Skip connection: low-rank or full
        if self.skip_low_rank:
            y_skip_raw = nn.Dense(dim_swiglu_total, use_bias=False, name="skip_up")(
                c_skip
            )
        else:
            y_skip_raw = c_skip  # Already dim_swiglu_total
        y_skip = y_skip_raw.reshape(B, L, self.K, dim_swiglu_head)

        y_spec_val, y_spec_gate = jnp.split(y_spec_raw, 2, axis=-1)
        y_skip_val, y_skip_gate = jnp.split(y_skip, 2, axis=-1)

        highway_scale = self.param(
            "highway_scale", nn.initializers.constant(1.0), (1, 1, self.K, 1)
        )

        y_val = y_spec_val + (y_skip_val * highway_scale)
        y_gate = y_spec_gate + (y_skip_gate * highway_scale)

        y_act = y_val * jax.nn.sigmoid(y_gate)

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

        # Skip dimension: low-rank (D//4) or full (dim_swiglu_total)
        dim_skip = D // 4 if self.skip_low_rank else dim_swiglu_total
        dim_gate = self.K

        # Unpack state (now with single fused conv buffer)
        den_acc, re_acc, im_acc, pos, conv_buffer = state

        # ======================================================================
        # FUSED INPUT PROJECTION + CONV (matches __call__)
        # ======================================================================
        dim_mem_total = dim_memory + self.K
        dim_conv_total = dim_mem_total + dim_query_total
        dim_total = dim_conv_total + dim_skip + dim_gate

        z_all = nn.Dense(dim_total, use_bias=False, name="in_proj")(x_t)
        z_all = z_all.astype(self.compute_dtype)

        # Split: conv_branch vs non-conv branches
        z_conv = z_all[..., :dim_conv_total]
        c_skip = z_all[..., dim_conv_total : dim_conv_total + dim_skip]
        gate_logits = z_all[..., dim_conv_total + dim_skip :]

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
        k_val = nn.RMSNorm(dtype=self.compute_dtype, name="k_norm")(k_val)
        s_raw = z_mem[..., dim_memory:]

        # Process query branch (matches __call__)
        q_raw = nn.RMSNorm(dtype=self.compute_dtype, name="q_norm")(q_raw)
        q_raw = q_raw.reshape(B, self.K_q, 1, H, self.M, 2)
        q_re, q_im = q_raw[..., 0], q_raw[..., 1]  # (B, K_q, 1, H, M)

        # Theta grid
        theta_min, theta_max = 0.001, 3.0

        if self.M == 1:

            def init_theta_m1(key, shape):
                grid = np.geomspace(theta_min, theta_max, self.K).reshape(
                    1, 1, self.K, 1, 1
                )
                base = np.tile(grid, (1, 1, 1, H, 1))
                u = (base - theta_min) / max(theta_max - theta_min, 1e-6)
                u = np.clip(u, 1e-4, 1 - 1e-4)
                raw = np.log(u) - np.log(1 - u)
                return jnp.array(raw, dtype=jnp.float32)

            theta_raw = self.param("theta_raw", init_theta_m1, (1, 1, self.K, H, 1))
            theta = theta_min + (theta_max - theta_min) * jax.nn.sigmoid(
                theta_raw
            ).astype(jnp.float32)
        else:

            def init_theta_deltas(key, shape):
                grid_m = np.geomspace(theta_min, theta_max, self.M)
                base = np.tile(grid_m.reshape(1, 1, 1, 1, self.M), (1, 1, self.K, H, 1))
                return jnp.log(jnp.exp(base) - 1.0 + 1e-4)

            theta_d_raw = self.param(
                "theta_d_raw", init_theta_deltas, (1, 1, self.K, H, self.M)
            )
            theta_d = jax.nn.softplus(theta_d_raw).astype(jnp.float32) + 1e-4
            theta_accum = jnp.cumsum(theta_d, axis=-1)

            scale_range = theta_max - theta_min
            total_sum = theta_accum[..., -1:]
            theta = theta_min + (theta_accum / total_sum) * scale_range

            # Compute w_int from theta_accum (reuse for later)
            dtheta_raw = theta_accum[..., 1:] - theta_accum[..., :-1]
            dtheta = dtheta_raw * (scale_range / total_sum)
            w0 = dtheta[..., :1] * 0.5
            w_mid = 0.5 * (dtheta[..., :-1] + dtheta[..., 1:])
            wL = dtheta[..., -1:] * 0.5
            w_int_m = jnp.concatenate([w0, w_mid, wL], axis=-1)
            w_int_m = w_int_m.reshape(1, 1, self.K_q, self.n_rep, H, self.M)

        theta = theta[0, 0]  # (K, H, M)

        # dt: content-based "delta time" (like Mamba)
        # Controls BOTH contribution weight AND decay rate
        # Must match __call__ initialization
        dt_scale = self.param("dt_scale", nn.initializers.ones, (self.K,))
        dt_bias = self.param(
            "dt_bias",
            lambda k, s: jnp.log(
                jnp.expm1(jnp.array(np.geomspace(0.001, 0.1, s[0])))
            ).astype(jnp.float32),
            (self.K,),
        )
        dt_raw = dt_scale[None, :] * s_raw.astype(jnp.float32) + dt_bias[None, :]
        dt_raw = jnp.clip(dt_raw, -20.0, 20.0)

        # Softplus for positive dt (like Mamba)
        dt = jax.nn.softplus(dt_raw)
        dt = jnp.clip(dt, 0.001, 0.1)  # (B, K)

        # A: base decay rate per head (like Mamba)
        # Must match __call__ initialization
        log_A = self.param(
            "log_A",
            lambda k, s: jnp.log(jnp.array(np.geomspace(1.0, 16.0, s[0]))).astype(
                jnp.float32
            ),
            (self.K,),
        )
        A = -jnp.exp(log_A)  # (K,) - negative for decay

        # A_disc = A * dt (discretized decay, like Mamba)
        A_disc = A[None, :] * dt  # (B, K)
        A_disc = jnp.clip(A_disc, -20.0, 0.0)

        # Modulation
        k_f32 = k_val[..., None].astype(jnp.float32)  # (B, K, H, 1)
        dt_b = dt[..., None, None]  # (B, K, 1, 1)

        phase_scale = self.param("phase_scale", nn.initializers.ones, (self.K,))
        phase_scale_b = phase_scale.reshape(1, self.K, 1, 1)
        k_scaled = k_f32 * phase_scale_b
        phi = (k_scaled / (1.0 + jnp.abs(k_scaled))) * theta  # softsign (B, K, H, M)

        # x_disc = x * dt (discretized input, like Mamba)
        kvw = k_f32 * dt_b

        re = kvw * jnp.cos(phi)
        im = kvw * jnp.sin(phi)

        # Decay for this step: exp(A_disc)
        decay_rate = jnp.exp(A_disc)  # (B, K)
        decay_rate = decay_rate[..., None, None]  # (B, K, 1, 1) for broadcasting

        # Update accumulation with Mamba-style decay
        # state_new = exp(A * dt) * state_old + x * dt
        den_acc_new = den_acc + dt  # Keep track of cumulative dt (optional)
        re_acc_new = decay_rate * re_acc + re
        im_acc_new = decay_rate * im_acc + im

        state_re = re_acc_new
        state_im = im_acc_new

        # Group state for GQA: (B, K, H, M) -> (B, K_q, n_rep, H, M)
        state_re_grouped = state_re.reshape(B, self.K_q, self.n_rep, H, self.M)
        state_im_grouped = state_im.reshape(B, self.K_q, self.n_rep, H, self.M)

        # w_int for integration (must match __call__)
        if self.M == 1:
            w_int_raw = self.param(
                "w_int_raw", nn.initializers.zeros, (1, 1, self.K_q, self.n_rep, H, 1)
            )
            w_int_raw_clipped = jnp.clip(w_int_raw, -5.0, 5.0)
            w_int = jnp.exp(w_int_raw_clipped).astype(jnp.float32)
        else:
            # w_int_m was computed above in the theta block
            w_int = w_int_m

        # Remove batch/seq dims from w_int for step
        w_int_step = w_int[0, 0]  # (K_q, n_rep, H, M)

        # Compute complex multiplication (matching)
        # q_re, q_im: (B, K_q, 1, H, M) from reshape at line 461
        # state_re_grouped, state_im_grouped: (B, K_q, n_rep, H, M)
        match_re = state_re_grouped * q_re + state_im_grouped * q_im
        match_im = state_im_grouped * q_re - state_re_grouped * q_im

        # Integrate over M (frequencies) with w_int
        out_re_g = jnp.sum(match_re * w_int_step, axis=-1)  # (B, K_q, n_rep, H)
        out_im_g = jnp.sum(match_im * w_int_step, axis=-1)

        # Reshape to (B, K, H)
        out_re = out_re_g.reshape(B, self.K, H).astype(self.compute_dtype)
        out_im = out_im_g.reshape(B, self.K, H).astype(self.compute_dtype)

        # Concatenate (not stack!) to match __call__
        out_complex = jnp.concatenate([out_re, out_im], axis=-1)  # (B, K, 2*H)

        # Flatten and apply GatedRMSNorm
        out_complex_flat = out_complex.reshape(B, -1)
        out_complex_flat = out_complex_flat.astype(self.compute_dtype)

        # GatedRMSNorm (matches __call__)
        gate_for_norm = nn.Dense(
            out_complex_flat.shape[-1], use_bias=False, name="gate_proj"
        )(gate_logits)
        out_normed = GatedRMSNorm(dtype=self.compute_dtype, name="gated_norm")(
            out_complex_flat, gate_for_norm
        )
        out_complex = out_normed.reshape(B, self.K, 2 * H)

        # Readout
        W_readout = self.param(
            "W_readout",
            nn.initializers.glorot_uniform(),
            (self.K, 2 * H, dim_swiglu_head),
        )
        y_spec_raw = jnp.einsum("bkf,kfn->bkn", out_complex, W_readout)

        # Skip connection
        if self.skip_low_rank:
            y_skip_raw = nn.Dense(dim_swiglu_total, use_bias=False, name="skip_up")(
                c_skip
            )
        else:
            y_skip_raw = c_skip
        y_skip = y_skip_raw.reshape(B, self.K, dim_swiglu_head)

        y_spec_val, y_spec_gate = jnp.split(y_spec_raw, 2, axis=-1)
        y_skip_val, y_skip_gate = jnp.split(y_skip, 2, axis=-1)

        highway_scale = self.param(
            "highway_scale", nn.initializers.constant(1.0), (1, 1, self.K, 1)
        )
        highway_scale_step = highway_scale[0, 0]  # (K, 1)

        y_val = y_spec_val + (y_skip_val * highway_scale_step)
        y_gate = y_spec_gate + (y_skip_gate * highway_scale_step)

        y_act = y_val * jax.nn.sigmoid(y_gate)

        # Output
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

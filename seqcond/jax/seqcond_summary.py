from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


class SeqCondAttention(nn.Module):
    """
    Simplified SeqCond: No queries.
    Memory conv -> characteristic function summary -> flatten -> SwiGLU with skip.
    """

    num_heads: int = 12
    num_anchor_heads: int = 0
    num_thetas: int = 1

    expand_factor: float = 1.0
    out_expand_factor: int = 3
    conv_kernel_size: int = 4

    dropout: float = 0.0
    maxlen: Optional[int] = None
    use_square_matrix: bool = False

    compute_dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.K = self.num_heads
        self.M = self.num_thetas
        self.num_decay_heads = self.K - self.num_anchor_heads

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        B, L, D = x.shape
        d_inner = int(D * self.expand_factor)

        H = max(1, d_inner // (self.K * self.M))
        dim_memory = self.K * H

        dim_expand = H * self.out_expand_factor
        dim_swiglu_head = dim_expand * 2
        dim_swiglu_total = self.K * dim_swiglu_head

        # Fused projection (optimization 1)
        dim_fused = dim_memory + self.K + dim_swiglu_total
        z_fused = nn.Dense(dim_fused, use_bias=False, name="in_proj_fused")(x)

        z_mem = z_fused[..., : dim_memory + self.K].astype(self.compute_dtype)
        skip_direct = z_fused[..., dim_memory + self.K :]

        # Memory conv
        z_mem = nn.Conv(
            features=z_mem.shape[-1],
            kernel_size=(self.conv_kernel_size,),
            padding=((self.conv_kernel_size - 1, 0),),
            feature_group_count=z_mem.shape[-1],
            use_bias=False,
            name="conv_mem",
        )(z_mem)
        z_mem = jax.nn.silu(z_mem)

        k_val = z_mem[..., :dim_memory].reshape(B, L, self.K, H)
        k_val = nn.RMSNorm(dtype=self.compute_dtype, name="k_norm")(k_val)
        s_raw = z_mem[..., dim_memory:]

        if mask is not None:
            m = mask.astype(x.dtype)[:, :, None, None]
            s_raw = s_raw * mask.astype(x.dtype)[..., None]
            k_val = k_val * m

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

        # Decay weights
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
        log_p = jnp.clip(
            score_scale[None, None, :] * s_raw.astype(jnp.float32), -10.0, 10.0
        )
        p_w = jnp.exp(log_p + log_time_weight)

        # Modulation
        k_f32 = k_val[..., None].astype(
            jnp.float32
        )  # Optimization 5: cast after unsqueeze
        p_w_b = p_w[..., None, None]

        tanh_scale = self.param("tanh_scale", nn.initializers.ones, (self.K,))
        tanh_scale_b = tanh_scale.reshape(
            1, 1, self.K, 1, 1
        )  # Optimization 3: reshape instead of indexing
        phi = jnp.tanh(k_f32 * tanh_scale_b) * theta
        kvw = k_f32 * p_w_b

        re = kvw * jnp.cos(phi)
        im = kvw * jnp.sin(phi)

        # Accumulation
        if self.use_square_matrix:
            mask_idx = jnp.arange(L)[:, None] >= jnp.arange(L)[None, :]
            causal_mask = mask_idx.astype(self.compute_dtype)

            den_acc = jnp.einsum(
                "ts,bsk->btk", causal_mask, p_w.astype(self.compute_dtype)
            )

            stack_ri = jnp.stack([re, im], axis=-1).astype(self.compute_dtype)
            acc_ri = jnp.einsum("ts,bskhmc->btkhmc", causal_mask, stack_ri)

            re_acc = acc_ri[..., 0]
            im_acc = acc_ri[..., 1]
        else:
            flat_size = self.K * H * self.M
            re_flat = re.reshape(B, L, flat_size)
            im_flat = im.reshape(B, L, flat_size)
            den_flat = p_w

            stack = jnp.concatenate([den_flat, re_flat, im_flat], axis=-1)
            cumsum = jnp.cumsum(stack, axis=1)

            den_acc, re_acc_flat, im_acc_flat = jnp.split(
                cumsum, [self.K, self.K + flat_size], axis=-1
            )

            re_acc = re_acc_flat.reshape(B, L, self.K, H, self.M)
            im_acc = im_acc_flat.reshape(B, L, self.K, H, self.M)

        # Normalize
        inv_den = 1.0 / jnp.maximum(den_acc, 1e-4)
        inv_den = inv_den[..., None, None]

        state_re = re_acc * inv_den
        state_im = im_acc * inv_den

        # Flatten summary (no queries)
        # Shape: (B, L, K, H, M) -> (B, L, K*H*M*2)
        # Optimization 2: stack + reshape instead of concatenate
        summary_flat = jnp.stack([state_re, state_im], axis=-1).reshape(B, L, -1)
        summary_flat = summary_flat.astype(self.compute_dtype)

        # SwiGLU with skip
        y_spectral = nn.Dense(dim_swiglu_total, use_bias=False, name="swiglu_proj")(
            summary_flat
        )
        y_spectral = y_spectral.reshape(B, L, self.K, dim_swiglu_head)
        y_skip = skip_direct.reshape(B, L, self.K, dim_swiglu_head)

        # Fusion
        y_spec_val, y_spec_gate = jnp.split(y_spectral, 2, axis=-1)
        y_skip_val, y_skip_gate = jnp.split(y_skip, 2, axis=-1)

        y_val = y_spec_val + y_skip_val
        y_gate = y_spec_gate + y_skip_gate

        y_act = y_val * jax.nn.silu(y_gate)

        # Output projection
        y_flat = y_act.reshape(B, L, -1)
        out = nn.Dense(D, use_bias=False, name="out_proj")(y_flat)

        if self.dropout > 0:
            out = nn.Dropout(self.dropout)(out, deterministic=deterministic)

        return out

    @nn.compact
    def step(self, x_t, state, deterministic=True):
        """
        O(1) autoregressive decoding step.

        Args:
            x_t: Input token embedding (B, D)
            state: Tuple of (den_acc, re_acc, im_acc, pos, conv_buffer) where:
                - den_acc: (B, K) - accumulated denominator
                - re_acc: (B, K, H, M) - accumulated real part
                - im_acc: (B, K, H, M) - accumulated imaginary part
                - pos: (B,) - current position
                - conv_buffer: (B, conv_kernel_size-1, dim_memory+K) - conv history
            deterministic: Whether to use dropout

        Returns:
            out: (B, D) - output for this step
            new_state: Updated state tuple
        """
        B, D = x_t.shape
        d_inner = int(D * self.expand_factor)

        H = max(1, d_inner // (self.K * self.M))
        dim_memory = self.K * H

        dim_expand = H * self.out_expand_factor
        dim_swiglu_head = dim_expand * 2
        dim_swiglu_total = self.K * dim_swiglu_head

        # Unpack state
        den_acc, re_acc, im_acc, pos, conv_buffer = state

        # Fused projection
        dim_fused = dim_memory + self.K + dim_swiglu_total
        z_fused = nn.Dense(dim_fused, use_bias=False, name="in_proj_fused")(x_t)

        z_mem = z_fused[..., : dim_memory + self.K].astype(self.compute_dtype)
        skip_direct = z_fused[..., dim_memory + self.K :]

        # Memory conv with buffer
        # Concatenate buffer with current input
        z_mem_expanded = z_mem[:, None, :]  # (B, 1, dim_memory+K)
        conv_input = jnp.concatenate(
            [conv_buffer, z_mem_expanded], axis=1
        )  # (B, conv_kernel_size, dim_memory+K)

        # Apply conv
        z_mem_conv = nn.Conv(
            features=conv_input.shape[-1],
            kernel_size=(self.conv_kernel_size,),
            padding="VALID",
            feature_group_count=conv_input.shape[-1],
            use_bias=False,
            name="conv_mem",
        )(
            conv_input
        )  # (B, 1, dim_memory+K)

        z_mem = jax.nn.silu(z_mem_conv[:, 0, :])  # (B, dim_memory+K)

        # Update conv buffer (shift and append)
        conv_buffer_new = jnp.concatenate(
            [conv_buffer[:, 1:, :], z_mem_expanded], axis=1
        )

        k_val = z_mem[..., :dim_memory].reshape(B, self.K, H)
        k_val = nn.RMSNorm(dtype=self.compute_dtype, name="k_norm")(k_val)
        s_raw = z_mem[..., dim_memory:]

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
        log_p = jnp.clip(score_scale[None, :] * s_raw.astype(jnp.float32), -10.0, 10.0)
        p_w = jnp.exp(log_p + log_time_weight)

        # Modulation
        k_f32 = k_val[..., None].astype(jnp.float32)  # (B, K, H, 1)
        p_w_b = p_w[..., None, None]  # (B, K, 1, 1)

        tanh_scale = self.param("tanh_scale", nn.initializers.ones, (self.K,))
        tanh_scale_b = tanh_scale.reshape(1, self.K, 1, 1)
        phi = jnp.tanh(k_f32 * tanh_scale_b) * theta  # (B, K, H, M)
        kvw = k_f32 * p_w_b

        re = kvw * jnp.cos(phi)
        im = kvw * jnp.sin(phi)

        # Update accumulation (O(1) update)
        den_acc_new = den_acc + p_w  # (B, K)
        re_acc_new = re_acc + re  # (B, K, H, M)
        im_acc_new = im_acc + im  # (B, K, H, M)

        # Normalize
        inv_den = 1.0 / jnp.maximum(den_acc_new, 1e-4)
        inv_den = inv_den[..., None, None]  # (B, K, 1, 1)

        state_re = re_acc_new * inv_den
        state_im = im_acc_new * inv_den

        # Flatten summary
        summary_flat = jnp.stack([state_re, state_im], axis=-1).reshape(B, -1)
        summary_flat = summary_flat.astype(self.compute_dtype)

        # SwiGLU with skip
        y_spectral = nn.Dense(dim_swiglu_total, use_bias=False, name="swiglu_proj")(
            summary_flat
        )
        y_spectral = y_spectral.reshape(B, self.K, dim_swiglu_head)
        y_skip = skip_direct.reshape(B, self.K, dim_swiglu_head)

        # Fusion
        y_spec_val, y_spec_gate = jnp.split(y_spectral, 2, axis=-1)
        y_skip_val, y_skip_gate = jnp.split(y_skip, 2, axis=-1)

        y_val = y_spec_val + y_skip_val
        y_gate = y_spec_gate + y_skip_gate

        y_act = y_val * jax.nn.silu(y_gate)

        # Output projection
        y_flat = y_act.reshape(B, -1)
        out = nn.Dense(D, use_bias=False, name="out_proj")(y_flat)

        if self.dropout > 0:
            out = nn.Dropout(self.dropout)(out, deterministic=deterministic)

        # Update position
        pos_new = pos + 1

        return out, (den_acc_new, re_acc_new, im_acc_new, pos_new, conv_buffer_new)


class SeqCondBlock(nn.Module):
    num_heads: int = 32
    expand_factor: float = 1.0
    num_thetas: int = 1
    num_anchor_heads: int = 0
    dropout: float = 0.0
    norm_eps: float = 1e-5
    maxlen: Optional[int] = None
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
            expand_factor=self.expand_factor,
            num_thetas=self.num_thetas,
            num_anchor_heads=self.num_anchor_heads,
            dropout=self.dropout,
            maxlen=self.maxlen,
            use_square_matrix=self.use_square_matrix,
            compute_dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )(h, mask=mask, deterministic=deterministic)
        return x + h

"""SeqCond attention layer for Keras 3 (ported from seqcond_fast.py)."""

from typing import Optional

import numpy as np
import keras
from keras import ops, layers

from .norm import RMSNorm, gated_rmsnorm


class SeqCondAttention(layers.Layer):
    """SeqCond spectral-temporal attention (training forward pass only).

    supports_masking: we handle masking explicitly via the `mask` argument.

    Implements the full SeqCond attention mechanism with:
    - Fused input projection + causal depthwise conv
    - Learnable spectral grid (theta) with softsign phase modulation
    - Temporal decay weighting
    - Cumulative-sum (linear) or matrix-multiply (quadratic) scan
    - GQA-style grouped query matching
    - GatedRMSNorm + SwiGLU output fusion
    """

    def __init__(
        self,
        num_heads=12,
        num_query_heads=6,
        num_anchor_heads=0,
        num_thetas=1,
        conv_kernel_size=4,
        expand_factor=1,
        out_expand_factor=3,
        dropout=0.0,
        maxlen=None,
        use_square_matrix=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_query_heads = num_query_heads
        self.num_anchor_heads = num_anchor_heads
        self.num_thetas = num_thetas
        self.conv_kernel_size = conv_kernel_size
        self.expand_factor = expand_factor
        self.out_expand_factor = out_expand_factor
        self.dropout_rate = dropout
        self.maxlen = maxlen
        self.use_square_matrix = use_square_matrix

        self.K = num_heads
        self.K_q = num_query_heads
        self.M = num_thetas
        self.num_decay_heads = self.K - num_anchor_heads
        assert self.K % self.K_q == 0
        self.n_rep = self.K // self.K_q
        self.supports_masking = True

    def build(self, input_shape):
        D = input_shape[-1]
        d_inner = int(D * self.expand_factor)
        K, K_q, M = self.K, self.K_q, self.M
        H = max(1, d_inner // (K * M))
        self.H = H

        dim_memory = K * H
        dim_query_head = H * M * 2
        dim_query_total = K_q * dim_query_head
        dim_expand = H * self.out_expand_factor
        dim_swiglu_head = dim_expand * 2

        dim_mem_total = dim_memory + K
        dim_conv_total = dim_mem_total + dim_query_total

        self.dim_memory = dim_memory
        self.dim_mem_total = dim_mem_total
        self.dim_swiglu_head = dim_swiglu_head

        # ── Sub-layers ──────────────────────────────────────────────────
        self.in_proj = layers.Dense(dim_conv_total, use_bias=False, name="in_proj")
        self.conv = layers.Conv1D(
            filters=dim_conv_total,
            kernel_size=self.conv_kernel_size,
            padding="causal",
            groups=dim_conv_total,
            use_bias=False,
            name="conv",
        )
        out_complex_dim = K * 2 * H
        self.gate_proj = layers.Dense(out_complex_dim, use_bias=False, name="gate_proj")
        self.out_proj = layers.Dense(D, use_bias=False, name="out_proj")
        if self.dropout_rate > 0:
            self.drop = layers.Dropout(self.dropout_rate)

        # ── Learnable parameters ────────────────────────────────────────
        # Spectral grid (theta)
        if M == 1:
            init_val = self._init_theta_raw_m1(K, H)
            self.theta_raw = self.add_weight(
                name="theta_raw",
                shape=(1, 1, K, H, 1),
                initializer=keras.initializers.Constant(init_val),
            )
        else:
            init_val = self._init_theta_d_raw(K, H, M)
            self.theta_d_raw = self.add_weight(
                name="theta_d_raw",
                shape=(1, 1, K, H, M),
                initializer=keras.initializers.Constant(init_val),
            )

        # Integration weights
        self.w_int_raw = self.add_weight(
            name="w_int_raw",
            shape=(1, 1, K_q, self.n_rep, H, M),
            initializer="ones",
        )

        # Temporal decay
        if self.num_decay_heads > 0:
            decay_init = np.log(
                np.exp(np.geomspace(0.001, 0.1, self.num_decay_heads)) - 1
            ).astype(np.float32)
            self.decay_slopes = self.add_weight(
                name="decay_slopes",
                shape=(self.num_decay_heads,),
                initializer=keras.initializers.Constant(decay_init),
            )
        if self.num_anchor_heads > 0:
            anchor_init = np.log(
                np.exp(np.geomspace(0.01, 0.1, self.num_anchor_heads)) - 1
            ).astype(np.float32)
            self.anchor_slopes = self.add_weight(
                name="anchor_slopes",
                shape=(self.num_anchor_heads,),
                initializer=keras.initializers.Constant(anchor_init),
            )

        # Score scale / bias
        self.score_scale = self.add_weight(
            name="score_scale", shape=(K,), initializer="ones"
        )
        self.score_bias = self.add_weight(
            name="score_bias", shape=(K,), initializer="zeros"
        )

        # Phase scale
        self.phase_scale = self.add_weight(
            name="phase_scale", shape=(K,), initializer="ones"
        )

        # GatedRMSNorm weight
        self.gated_norm_weight = self.add_weight(
            name="gated_norm_weight",
            shape=(out_complex_dim,),
            initializer="ones",
        )

        # Readout matrix
        self.W_readout = self.add_weight(
            name="W_readout",
            shape=(K, 2 * H, dim_swiglu_head),
            initializer="glorot_uniform",
        )

        super().build(input_shape)

    # ── Initializers ────────────────────────────────────────────────────
    @staticmethod
    def _init_theta_raw_m1(K, H):
        """Logit initialiser for theta when M == 1 (geomspace in [0.001, 3])."""
        geom = np.geomspace(0.001, 3.0, K).reshape(1, 1, K, 1, 1)
        geom = np.tile(geom, (1, 1, 1, H, 1))
        normalized = np.clip((geom - 0.001) / 2.999, 1e-4, 1 - 1e-4)
        return (np.log(normalized) - np.log(1 - normalized)).astype(np.float32)

    @staticmethod
    def _init_theta_d_raw(K, H, M):
        """Softplus-inverse initialiser for theta deltas when M > 1."""
        geom = np.geomspace(0.001, 3.0, M).reshape(1, 1, 1, 1, M)
        geom = np.tile(geom, (1, 1, K, H, 1))
        return np.log(np.exp(geom) - 1.0 + 1e-4).astype(np.float32)

    # ── Forward pass ────────────────────────────────────────────────────
    def call(self, x, mask=None, training=False):
        B = ops.shape(x)[0]
        L = ops.shape(x)[1]
        K, K_q, M, H = self.K, self.K_q, self.M, self.H

        # ==================================================================
        # 1. FUSED INPUT PROJECTION + CONV
        # ==================================================================
        z_conv = self.in_proj(x)
        z_conv = self.conv(z_conv)
        z_conv = ops.silu(z_conv)

        z_mem = z_conv[..., : self.dim_mem_total]
        q_raw = z_conv[..., self.dim_mem_total :]

        k_val = ops.reshape(z_mem[..., : self.dim_memory], (B, L, K, H))
        s_raw = z_mem[..., self.dim_memory :]

        if mask is not None:
            m = ops.cast(mask, x.dtype)
            s_raw = s_raw * m[..., None]
            k_val = k_val * m[:, :, None, None]

        q_raw = ops.reshape(q_raw, (B, L, K_q, 1, H, M, 2))
        q_re = q_raw[..., 0]
        q_im = q_raw[..., 1]

        # ==================================================================
        # 2. SPECTRAL GRID (THETA & W_INT)
        # ==================================================================
        if self.M == 1:
            theta = 0.001 + 2.999 * ops.sigmoid(ops.cast(self.theta_raw, "float32"))
        else:
            theta_d = ops.softplus(ops.cast(self.theta_d_raw, "float32")) + 1e-4
            theta_accum = ops.cumsum(theta_d, axis=-1)
            total_sum = theta_accum[..., -1:]
            theta = 0.001 + (theta_accum / total_sum) * 2.999

        w_int = ops.exp(ops.cast(self.w_int_raw, "float32"))
        w_int = w_int / (ops.sum(w_int, axis=-1, keepdims=True) + 1e-6)

        # ==================================================================
        # 3. MODULATION & SCAN
        # ==================================================================
        pos = ops.cast(ops.arange(L), "float32")
        log_w_parts = []
        if self.num_decay_heads > 0:
            slopes = ops.reshape(ops.softplus(self.decay_slopes), (1, 1, -1))
            maxlen_f = ops.cast((self.maxlen or L) - 1, "float32")
            dist = ops.maximum(maxlen_f - pos, 0.0)
            log_w_parts.append(-slopes * dist[None, :, None])
        if self.num_anchor_heads > 0:
            slopes_a = ops.reshape(ops.softplus(self.anchor_slopes), (1, 1, -1))
            log_w_parts.append(-slopes_a * pos[None, :, None])

        if log_w_parts:
            log_time_weight = ops.concatenate(log_w_parts, axis=2)
        else:
            log_time_weight = ops.zeros((1, L, K))

        score_raw = (
            self.score_scale[None, None, :] * ops.cast(s_raw, "float32")
            + self.score_bias[None, None, :]
        )
        p_w_content = ops.softplus(score_raw)
        temporal_weight = ops.exp(log_time_weight)
        p_w = p_w_content * temporal_weight
        p_w = ops.clip(p_w, 1e-4, 5000.0)

        # Modulation
        k_f32 = ops.cast(k_val, "float32")[..., None]  # (B, L, K, H, 1)
        p_w_b = p_w[..., None, None]  # (B, L, K, 1, 1)

        phase_scale_b = self.phase_scale[None, None, :, None, None]
        k_scaled = k_f32 * phase_scale_b
        phi = (k_scaled / (1.0 + ops.abs(k_scaled))) * theta
        kvw = k_f32 * p_w_b

        re = kvw * ops.cos(phi)
        im = kvw * ops.sin(phi)

        # ==================================================================
        # 4. SCAN (MATRIX or LINEAR)
        # ==================================================================
        if self.use_square_matrix:
            # O(L²) matrix path
            idx = ops.arange(L)
            causal = ops.cast(idx[:, None] >= idx[None, :], "float32")
            den_acc = ops.einsum("ts,bsk->btk", causal, ops.cast(p_w, "float32"))
            stack_ri = ops.cast(ops.stack([re, im], axis=-1), "float32")
            acc_ri = ops.einsum("ts,bskhmc->btkhmc", causal, stack_ri)
            re_acc = acc_ri[..., 0]
            im_acc = acc_ri[..., 1]
        else:
            # O(L) cumsum path
            flat_size = K * H * M
            re_flat = ops.reshape(ops.cast(re, "float32"), (B, L, flat_size))
            im_flat = ops.reshape(ops.cast(im, "float32"), (B, L, flat_size))
            den_flat = ops.cast(p_w, "float32")

            stack = ops.concatenate([den_flat, re_flat, im_flat], axis=-1)
            cumsum = ops.cumsum(stack, axis=1)

            den_acc = cumsum[..., :K]
            re_acc = ops.reshape(cumsum[..., K : K + flat_size], (B, L, K, H, M))
            im_acc = ops.reshape(cumsum[..., K + flat_size :], (B, L, K, H, M))

        inv_den = 1.0 / ops.maximum(den_acc, 1e-4)
        inv_den = inv_den[..., None, None]

        state_re = re_acc * inv_den
        state_im = im_acc * inv_den

        # ==================================================================
        # 5. READOUT & GQA & INTEGRATION
        # ==================================================================
        state_re_g = ops.reshape(state_re, (B, L, K_q, self.n_rep, H, M))
        state_im_g = ops.reshape(state_im, (B, L, K_q, self.n_rep, H, M))

        scale = ops.rsqrt(ops.cast(H, "float32"))
        match_re = (state_re_g * q_re + state_im_g * q_im) * scale
        match_im = (state_im_g * q_re - state_re_g * q_im) * scale

        out_re_g = ops.sum(match_re * w_int, axis=-1)
        out_im_g = ops.sum(match_im * w_int, axis=-1)

        out_re = ops.reshape(out_re_g, (B, L, K, H))
        out_im = ops.reshape(out_im_g, (B, L, K, H))
        out_complex = ops.concatenate([out_re, out_im], axis=-1)

        # ==================================================================
        # 6. FUSION: GatedRMSNorm + SwiGLU + Output Projection
        # ==================================================================
        out_flat = ops.reshape(out_complex, (B, L, -1))
        gate = self.gate_proj(x)
        out_normed = gated_rmsnorm(out_flat, gate, self.gated_norm_weight)
        out_complex = ops.reshape(out_normed, (B, L, K, 2 * H))

        y_raw = ops.einsum("blkf,kfn->blkn", out_complex, self.W_readout)
        y_val, y_gate = ops.split(y_raw, 2, axis=-1)
        y_act = y_val * ops.sigmoid(y_gate)

        y_flat = ops.reshape(y_act, (B, L, -1))
        out = self.out_proj(y_flat)

        if self.dropout_rate > 0:
            out = self.drop(out, training=training)

        return out

    def step(self, x_t, state):
        """Single autoregressive step.

        Args:
            x_t: (B, D) current token embedding
            state: (den_acc, re_acc, im_acc, pos, conv_buffer)
                - den_acc: (B, K) accumulated denominator
                - re_acc: (B, K, H, M) accumulated real part
                - im_acc: (B, K, H, M) accumulated imaginary part
                - pos: (B,) current position
                - conv_buffer: (B, conv_kernel_size-1, dim_conv_total) conv history

        Returns:
            out: (B, D) output
            new_state: updated state tuple
        """
        den_acc, re_acc, im_acc, pos, conv_buffer = state
        B = ops.shape(x_t)[0]

        # Input projection
        z_conv = self.in_proj(x_t)  # (B, dim_conv_total)

        # Depthwise conv using buffer
        # Build input: [buffer, new_token]
        z_conv_expanded = ops.expand_dims(z_conv, 1)  # (B, 1, C)
        conv_input = ops.concatenate(
            [conv_buffer, z_conv_expanded], axis=1
        )  # (B, K, C)

        # Manual depthwise conv (groups=C)
        conv_kernel = self.conv.kernel  # (K, 1, C)
        conv_kernel_t = ops.transpose(ops.squeeze(conv_kernel, 1))  # (C, K)
        z_conv_out = ops.sum(
            conv_input * ops.transpose(conv_kernel_t), axis=1
        )  # (B, C)
        z_conv_act = ops.silu(z_conv_out)

        # Split into memory and query
        z_mem = z_conv_act[..., : self.dim_mem_total]
        q_raw = z_conv_act[..., self.dim_mem_total :]

        k_val = ops.reshape(z_mem[..., : self.dim_memory], (B, self.K, self.H))
        s_raw = z_mem[..., self.dim_memory :]

        q_raw = ops.reshape(q_raw, (B, self.K_q, 1, self.H, self.M, 2))
        q_re, q_im = q_raw[..., 0], q_raw[..., 1]

        # Compute theta and w_int (must match call() float32 casts)
        if self.M == 1:
            theta = (
                0.001 + 2.999 * ops.sigmoid(ops.cast(self.theta_raw, "float32"))[0, 0]
            )
        else:
            theta_d = ops.softplus(ops.cast(self.theta_d_raw, "float32")) + 1e-4
            theta_accum = ops.cumsum(theta_d, axis=-1)
            total_sum = theta_accum[..., -1:]
            theta = (0.001 + (theta_accum / total_sum) * 2.999)[0, 0]

        w_int_full = ops.exp(ops.cast(self.w_int_raw, "float32"))
        w_int = (w_int_full / (ops.sum(w_int_full, axis=-1, keepdims=True) + 1e-6))[
            0, 0
        ]

        # Temporal weighting
        log_w_list = []
        if self.num_decay_heads > 0:
            decay_slopes = ops.softplus(self.decay_slopes)
            dist = (self.maxlen or 2048) - 1 - ops.expand_dims(pos, -1)
            dist = ops.maximum(dist, 0.0)
            log_w_list.append(-ops.expand_dims(decay_slopes, 0) * dist)
        if self.num_anchor_heads > 0:
            anchor_slopes = ops.softplus(self.anchor_slopes)
            log_w_list.append(
                -ops.expand_dims(anchor_slopes, 0) * ops.expand_dims(pos, -1)
            )

        if log_w_list:
            log_time_weight = ops.concatenate(log_w_list, axis=1)
        else:
            log_time_weight = ops.zeros((B, self.K))

        # Compute p_w (cast s_raw to float32 to match call())
        score_raw = ops.expand_dims(self.score_scale, 0) * ops.cast(
            s_raw, "float32"
        ) + ops.expand_dims(self.score_bias, 0)
        p_w = ops.clip(ops.softplus(score_raw) * ops.exp(log_time_weight), 1e-4, 5000.0)

        # Complex accumulation
        k_f32 = ops.cast(ops.expand_dims(k_val, -1), "float32")
        p_w_b = ops.expand_dims(ops.expand_dims(p_w, -1), -1)
        phase_scale_b = ops.reshape(self.phase_scale, (1, self.K, 1, 1))
        k_scaled = k_f32 * phase_scale_b
        phi = (k_scaled / (1.0 + ops.abs(k_scaled))) * theta
        kvw = k_f32 * p_w_b
        re = kvw * ops.cos(phi)
        im = kvw * ops.sin(phi)

        # Update accumulators
        den_acc = den_acc + p_w
        re_acc = re_acc + re
        im_acc = im_acc + im

        # Normalize
        inv_den = 1.0 / ops.maximum(den_acc, 1e-4)
        inv_den = ops.expand_dims(ops.expand_dims(inv_den, -1), -1)
        state_re = re_acc * inv_den
        state_im = im_acc * inv_den

        # Query matching
        state_re_g = ops.reshape(state_re, (B, self.K_q, self.n_rep, self.H, self.M))
        state_im_g = ops.reshape(state_im, (B, self.K_q, self.n_rep, self.H, self.M))

        scale = 1.0 / (self.H**0.5)
        q_re_sq = ops.squeeze(q_re, 2)  # (B, K_q, H, M)
        q_im_sq = ops.squeeze(q_im, 2)
        q_re_exp = ops.expand_dims(q_re_sq, 2)  # (B, K_q, 1, H, M)
        q_im_exp = ops.expand_dims(q_im_sq, 2)

        match_re = ops.cast(
            (state_re_g * q_re_exp + state_im_g * q_im_exp) * scale, "float32"
        )
        match_im = ops.cast(
            (state_im_g * q_re_exp - state_re_g * q_im_exp) * scale, "float32"
        )

        w_int_f32 = ops.cast(w_int, "float32")
        out_re_g = ops.sum(match_re * w_int_f32, axis=-1)
        out_im_g = ops.sum(match_im * w_int_f32, axis=-1)

        out_re = ops.cast(ops.reshape(out_re_g, (B, self.K, self.H)), x_t.dtype)
        out_im = ops.cast(ops.reshape(out_im_g, (B, self.K, self.H)), x_t.dtype)
        out_complex = ops.concatenate([out_re, out_im], axis=-1)
        out_complex = ops.reshape(out_complex, (B, self.K, 2 * self.H))

        # GatedRMSNorm
        out_complex_flat = ops.reshape(out_complex, (B, -1))
        gate_for_norm = self.gate_proj(x_t)
        out_normed = gated_rmsnorm(
            out_complex_flat, gate_for_norm, self.gated_norm_weight
        )
        out_complex = ops.reshape(out_normed, (B, self.K, 2 * self.H))

        # W_readout + SwiGLU
        y_spec_raw = ops.einsum("bkf,kfn->bkn", out_complex, self.W_readout)
        y_val, y_gate = ops.split(y_spec_raw, 2, axis=-1)
        y_act = y_val * ops.sigmoid(y_gate)

        out = self.out_proj(ops.reshape(y_act, (B, -1)))

        # Update position and conv_buffer
        pos = pos + 1
        pos = ops.minimum(pos, (self.maxlen or 2048) - 1)

        if self.conv_kernel_size > 1:
            # Shift buffer left and append new value
            if self.conv_kernel_size > 2:
                new_buffer = ops.concatenate(
                    [conv_buffer[:, 1:, :], z_conv_expanded], axis=1
                )
            else:
                new_buffer = z_conv_expanded
        else:
            new_buffer = conv_buffer

        new_state = (den_acc, re_acc, im_acc, pos, new_buffer)
        return out, new_state


class SeqCondBlock(layers.Layer):
    """Pre-norm residual wrapper around SeqCondAttention."""

    supports_masking = True

    def __init__(
        self,
        num_heads=32,
        num_query_heads=6,
        expand_factor=1.0,
        out_expand_factor=3,
        num_thetas=1,
        num_anchor_heads=0,
        conv_kernel_size=4,
        dropout=0.0,
        norm_eps=1e-5,
        maxlen=None,
        use_square_matrix=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pre_norm = RMSNorm(epsilon=norm_eps, name="pre_norm")
        self.attn = SeqCondAttention(
            num_heads=num_heads,
            num_query_heads=num_query_heads,
            expand_factor=expand_factor,
            out_expand_factor=out_expand_factor,
            num_thetas=num_thetas,
            num_anchor_heads=num_anchor_heads,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            maxlen=maxlen,
            use_square_matrix=use_square_matrix,
            name="attn",
        )

    def call(self, x, mask=None, training=False):
        h = self.pre_norm(x)
        h = self.attn(h, mask=mask, training=training)
        return x + h

    def step(self, x_t, state):
        """Single autoregressive step with residual."""
        h = self.pre_norm(x_t)
        h, new_state = self.attn.step(h, state)
        return x_t + h, new_state

"""
SeqCondAttention - mirrors JAX seqcond_fast.py exactly.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .norm import RMSNorm, GatedRMSNorm

# Triton kernels (optional)
try:
    from .triton_kernels import seqcond_step_triton, TRITON_AVAILABLE
except ImportError:
    TRITON_AVAILABLE = False
    seqcond_step_triton = None


class SeqCondAttention(nn.Module):
    """SeqCond attention - matches JAX exactly."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 12,
        num_query_heads: int = 6,
        num_anchor_heads: int = 0,
        num_thetas: int = 1,
        conv_kernel_size: int = 4,
        expand_factor: int = 1,
        out_expand_factor: int = 3,
        dropout: float = 0.0,
        maxlen: Optional[int] = None,
        **kwargs,  # Ignore skip_low_rank for backward compat
    ):
        super().__init__()
        assert num_heads % num_query_heads == 0

        self.d_model = d_model
        self.K = num_heads
        self.K_q = num_query_heads
        self.n_rep = num_heads // num_query_heads
        self.M = num_thetas
        self.num_decay_heads = num_heads - num_anchor_heads
        self.num_anchor_heads = num_anchor_heads
        self.conv_kernel_size = conv_kernel_size
        self.dropout_rate = dropout
        self.maxlen = maxlen

        d_inner = int(d_model * expand_factor)
        self.H = max(1, d_inner // (self.K * self.M))
        self.dim_memory = self.K * self.H
        self.dim_query_head = self.H * self.M * 2
        self.dim_query_total = self.K_q * self.dim_query_head
        self.dim_expand = self.H * out_expand_factor
        self.dim_swiglu_head = self.dim_expand * 2
        self.dim_swiglu_total = self.K * self.dim_swiglu_head
        self.dim_mem_total = self.dim_memory + self.K
        self.dim_conv_total = self.dim_mem_total + self.dim_query_total

        # Input projection (only conv_branch, gate comes from x directly)
        self.in_proj = nn.Linear(d_model, self.dim_conv_total, bias=False)
        # Depthwise conv (matches JAX nn.Conv with feature_group_count)
        self.conv_weight = nn.Parameter(
            torch.empty(self.dim_conv_total, 1, conv_kernel_size)
        )
        nn.init.kaiming_normal_(self.conv_weight)
        # Register buffers for cached computations (computed lazily in step)
        self.register_buffer("_conv_kernel_t", None)
        self.register_buffer("_theta_cached", None)
        self.register_buffer("_w_int_cached", None)
        self.register_buffer("_decay_slopes_cached", None)
        self.register_buffer("_anchor_slopes_cached", None)
        self.register_buffer("_phase_scale_b", None)
        self.register_buffer("_score_scale_b", None)
        self.register_buffer("_score_bias_b", None)
        # k_norm and q_norm removed (not needed after simplification)
        # self.k_norm = RMSNorm(self.H)
        # self.q_norm = RMSNorm(self.dim_query_total)

        if self.M == 1:
            init_theta = np.geomspace(0.001, 3.0, self.K).reshape(1, 1, self.K, 1, 1)
            init_theta = np.tile(init_theta, (1, 1, 1, self.H, 1))
            x = np.clip((init_theta - 0.001) / 2.999, 1e-4, 1 - 1e-4)
            theta_raw_init = np.log(x) - np.log(1 - x)
            self.theta_raw = nn.Parameter(
                torch.from_numpy(theta_raw_init.astype(np.float32))
            )
            self.w_int_raw = nn.Parameter(
                torch.zeros(1, 1, self.K_q, self.n_rep, self.H, 1)
            )
        else:
            init_vals = np.geomspace(0.001, 3.0, self.M).reshape(1, 1, 1, 1, self.M)
            init_vals = np.tile(init_vals, (1, 1, self.K, self.H, 1))
            self.theta_d_raw = nn.Parameter(
                torch.from_numpy(
                    np.log(np.exp(init_vals) - 1.0 + 1e-4).astype(np.float32)
                )
            )
            # Learnable w_int (independent of theta)
            self.w_int_raw = nn.Parameter(
                torch.zeros(1, 1, self.K_q, self.n_rep, self.H, self.M)
            )

        if self.num_decay_heads > 0:
            self.decay_slopes = nn.Parameter(
                torch.from_numpy(
                    np.log(
                        np.exp(np.geomspace(0.001, 0.1, self.num_decay_heads)) - 1
                    ).astype(np.float32)
                )
            )
        if self.num_anchor_heads > 0:
            self.anchor_slopes = nn.Parameter(
                torch.from_numpy(
                    np.log(
                        np.exp(np.geomspace(0.01, 0.1, self.num_anchor_heads)) - 1
                    ).astype(np.float32)
                )
            )

        self.score_scale = nn.Parameter(torch.ones(self.K))
        self.score_bias = nn.Parameter(torch.zeros(self.K))
        self.phase_scale = nn.Parameter(torch.ones(self.K))
        # GatedRMSNorm with gate from x
        self.gate_proj = nn.Linear(d_model, self.K * 2 * self.H, bias=False)
        self.gated_norm = GatedRMSNorm(self.K * 2 * self.H)

        self.W_readout = nn.Parameter(
            torch.empty(self.K, 2 * self.H, self.dim_swiglu_head)
        )
        nn.init.xavier_uniform_(self.W_readout)
        self.out_proj = nn.Linear(self.dim_swiglu_total // 2, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        B, L, D = x.shape
        z_conv = self.in_proj(x)

        z_conv_t = z_conv.transpose(1, 2)
        z_conv_t = F.pad(z_conv_t, (self.conv_kernel_size - 1, 0))
        z_conv_t = F.conv1d(z_conv_t, self.conv_weight, groups=self.dim_conv_total)
        z_conv = F.silu(z_conv_t.transpose(1, 2))

        z_mem = z_conv[..., : self.dim_mem_total]
        q_raw = z_conv[..., self.dim_mem_total :]
        k_val = z_mem[..., : self.dim_memory].reshape(B, L, self.K, self.H)
        # k_val = self.k_norm(k_val)  # Removed
        s_raw = z_mem[..., self.dim_memory :]

        # q_raw = self.q_norm(q_raw)  # Removed
        q_raw = q_raw.reshape(B, L, self.K_q, 1, self.H, self.M, 2)
        q_re, q_im = q_raw[..., 0], q_raw[..., 1]

        if self.M == 1:
            theta = 0.001 + 2.999 * torch.sigmoid(self.theta_raw)
        else:
            theta_d = F.softplus(self.theta_d_raw) + 1e-4
            theta_accum = torch.cumsum(theta_d, dim=-1)
            total_sum = theta_accum[..., -1:]
            theta = 0.001 + (theta_accum / total_sum) * 2.999

        # Learnable w_int with exp + normalize (matches JAX)
        w_int = torch.exp(self.w_int_raw)
        w_int = w_int / (w_int.sum(dim=-1, keepdim=True) + 1e-6)

        pos = torch.arange(L, dtype=torch.float32, device=x.device)
        log_w_list = []
        if self.num_decay_heads > 0:
            slopes = F.softplus(self.decay_slopes).view(1, 1, -1)
            dist = torch.clamp((self.maxlen or L) - 1 - pos, min=0.0).view(1, L, 1)
            log_w_list.append(-slopes * dist)
        if self.num_anchor_heads > 0:
            slopes_a = F.softplus(self.anchor_slopes).view(1, 1, -1)
            log_w_list.append(-slopes_a * pos.view(1, L, 1))
        log_time_weight = (
            torch.cat(log_w_list, dim=2)
            if log_w_list
            else torch.zeros(1, L, self.K, device=x.device)
        )

        score_raw = self.score_scale.view(
            1, 1, -1
        ) * s_raw.float() + self.score_bias.view(1, 1, -1)
        p_w_content = F.softplus(score_raw)
        p_w = (p_w_content * torch.exp(log_time_weight)).clamp(1e-6, 1000.0)

        k_f32 = k_val.float().unsqueeze(-1)
        p_w_b = p_w.unsqueeze(-1).unsqueeze(-1)
        phase_scale_b = self.phase_scale.view(1, 1, self.K, 1, 1)
        k_scaled = k_f32 * phase_scale_b
        phi = (k_scaled / (1.0 + k_scaled.abs())) * theta
        kvw = k_f32 * p_w_b
        re = kvw * torch.cos(phi)
        im = kvw * torch.sin(phi)

        # Accumulate with denominator
        flat_size = self.K * self.H * self.M
        den_flat = p_w.float()  # (B, L, K)
        stack = torch.cat(
            [den_flat, re.reshape(B, L, -1), im.reshape(B, L, -1)], dim=-1
        )
        cumsum = torch.cumsum(stack, dim=1)
        den_acc = cumsum[..., : self.K]
        re_acc = cumsum[..., self.K : self.K + flat_size].reshape(
            B, L, self.K, self.H, self.M
        )
        im_acc = cumsum[..., self.K + flat_size :].reshape(B, L, self.K, self.H, self.M)

        # Normalize by accumulated denominator
        inv_den = 1.0 / torch.clamp(den_acc, min=1e-4)
        inv_den = inv_den.unsqueeze(-1).unsqueeze(-1)  # (B, L, K, 1, 1)
        state_re = re_acc * inv_den
        state_im = im_acc * inv_den

        state_re_g = state_re.reshape(B, L, self.K_q, self.n_rep, self.H, self.M)
        state_im_g = state_im.reshape(B, L, self.K_q, self.n_rep, self.H, self.M)

        # Scale by 1/sqrt(H) (matches JAX)
        scale = 1.0 / (self.H**0.5)
        match_re = ((state_re_g * q_re + state_im_g * q_im) * scale).float()
        match_im = ((state_im_g * q_re - state_re_g * q_im) * scale).float()
        out_re_g = (match_re * w_int.float()).sum(dim=-1)
        out_im_g = (match_im * w_int.float()).sum(dim=-1)
        out_re = out_re_g.reshape(B, L, self.K, self.H).to(x.dtype)
        out_im = out_im_g.reshape(B, L, self.K, self.H).to(x.dtype)
        out_complex = torch.cat([out_re, out_im], dim=-1)

        # GatedRMSNorm with gate from x (matches JAX)
        out_complex_flat = out_complex.reshape(B, L, -1)
        gate_for_norm = self.gate_proj(x)
        out_normed = self.gated_norm(out_complex_flat, gate_for_norm)
        out_complex = out_normed.reshape(B, L, self.K, 2 * self.H)

        # W_readout -> SwiGLU (no skip)
        y_spec_raw = torch.einsum(
            "blkf,kfn->blkn", out_complex, self.W_readout.to(out_complex.dtype)
        )
        y_val, y_gate = y_spec_raw.chunk(2, dim=-1)
        y_act = y_val * torch.sigmoid(y_gate)

        output = self.out_proj(y_act.reshape(B, L, -1))

        if return_state:
            # Extract final state for continuing with step()
            # den_acc: sum of p_w over sequence
            den_acc_final = p_w.sum(dim=1)  # (B, K)
            # re_acc, im_acc: final cumsum values
            re_acc_final = re_acc[:, -1]  # (B, K, H, M)
            im_acc_final = im_acc[:, -1]  # (B, K, H, M)
            # pos: sequence length
            pos_final = torch.full((B,), L, dtype=torch.float32, device=x.device)
            # conv_buffer: last (kernel_size-1) values of z_conv (before silu)
            z_conv_pre_silu = self.in_proj(x)  # Need pre-silu values
            buffer_size = self.conv_kernel_size - 1
            if L >= buffer_size:
                conv_buffer_final = z_conv_pre_silu[:, -buffer_size:, :]
            else:
                pad_size = buffer_size - L
                padding = torch.zeros(
                    B,
                    pad_size,
                    self.dim_conv_total,
                    device=x.device,
                    dtype=z_conv_pre_silu.dtype,
                )
                conv_buffer_final = torch.cat([padding, z_conv_pre_silu], dim=1)
            state = (
                den_acc_final,
                re_acc_final,
                im_acc_final,
                pos_final,
                conv_buffer_final,
            )
            return output, state

        return output

    def step(
        self, x_t: torch.Tensor, state: Tuple, use_triton: bool = False
    ) -> Tuple[torch.Tensor, Tuple]:
        B, D = x_t.shape
        den_acc, re_acc, im_acc, pos, conv_buffer = state

        z_conv = self.in_proj(x_t)

        # Cache transposed kernel on first call
        if self._conv_kernel_t is None or self._conv_kernel_t.device != z_conv.device:
            self._conv_kernel_t = self.conv_weight[:, 0, :].t().contiguous()  # (K, C)

        # Build conv input: [buffer, new_token]
        z_conv_expanded = z_conv.unsqueeze(1)
        conv_input = torch.cat([conv_buffer, z_conv_expanded], dim=1)

        # Depthwise conv via matmul (faster than einsum for this shape)
        # conv_input: (B, K, C), kernel: (K, C) -> output: (B, C)
        z_conv_out = (conv_input * self._conv_kernel_t).sum(dim=1)
        z_conv_act = F.silu(z_conv_out)

        # conv_buffer will be updated in-place at the end of step()

        z_mem = z_conv_act[..., : self.dim_mem_total]
        q_raw = z_conv_act[..., self.dim_mem_total :]
        k_val = z_mem[..., : self.dim_memory].reshape(B, self.K, self.H)
        # k_val = self.k_norm(k_val)
        s_raw = z_mem[..., self.dim_memory :]

        # q_raw = self.q_norm(q_raw)
        q_raw = q_raw.reshape(B, self.K_q, 1, self.H, self.M, 2)
        q_re, q_im = q_raw[..., 0], q_raw[..., 1]

        # Cache theta and w_int on first call (they only depend on parameters)
        if self._theta_cached is None:
            if self.M == 1:
                self._theta_cached = (0.001 + 2.999 * torch.sigmoid(self.theta_raw))[
                    0, 0
                ]
            else:
                theta_d = F.softplus(self.theta_d_raw) + 1e-4
                theta_accum = torch.cumsum(theta_d, dim=-1)
                total_sum = theta_accum[..., -1:]
                self._theta_cached = (0.001 + (theta_accum / total_sum) * 2.999)[0, 0]
            # w_int: exp + normalize (matches JAX)
            w_int_full = torch.exp(self.w_int_raw)
            w_int_full = w_int_full / (w_int_full.sum(dim=-1, keepdim=True) + 1e-6)
            self._w_int_cached = w_int_full[0, 0]
        theta = self._theta_cached
        w_int = self._w_int_cached

        # Cache slopes on first call
        if self._decay_slopes_cached is None and self.num_decay_heads > 0:
            self._decay_slopes_cached = F.softplus(self.decay_slopes).view(1, -1)
        if self._anchor_slopes_cached is None and self.num_anchor_heads > 0:
            self._anchor_slopes_cached = F.softplus(self.anchor_slopes).view(1, -1)

        log_w_list = []
        if self.num_decay_heads > 0:
            dist = (self.maxlen or 2048) - 1 - pos.unsqueeze(-1)
            log_w_list.append(-self._decay_slopes_cached * dist.clamp(min=0.0))
        if self.num_anchor_heads > 0:
            log_w_list.append(-self._anchor_slopes_cached * pos.unsqueeze(-1))
        log_time_weight = (
            torch.cat(log_w_list, dim=1)
            if log_w_list
            else torch.zeros(B, self.K, device=x_t.device)
        )

        # Cache views on first call
        if self._score_scale_b is None:
            self._score_scale_b = self.score_scale.view(1, -1)
            self._score_bias_b = self.score_bias.view(1, -1)
            self._phase_scale_b = self.phase_scale.view(1, self.K, 1, 1)

        # Use Triton kernels if requested and available
        if use_triton and TRITON_AVAILABLE and seqcond_step_triton is not None:
            out_re, out_im = seqcond_step_triton(
                k_val.contiguous(),
                s_raw.contiguous(),
                q_re.squeeze(2).contiguous(),
                q_im.squeeze(2).contiguous(),
                re_acc,
                im_acc,
                den_acc,
                theta.contiguous(),
                w_int.contiguous(),
                self.phase_scale,
                self.score_scale,
                self.score_bias,
                log_time_weight.contiguous(),
            )
            out_complex = torch.cat([out_re, out_im], dim=-1)
        else:
            # Standard PyTorch path
            score_raw = self._score_scale_b * s_raw.float() + self._score_bias_b
            # p_w = (F.relu(score_raw) ** 2 * torch.exp(log_time_weight)).clamp(
            #     1e-6, 1000.0
            # )
            p_w = (F.softplus(score_raw) * torch.exp(log_time_weight)).clamp(
                1e-6, 1000.0
            )

            k_f32 = k_val.float().unsqueeze(-1)
            p_w_b = p_w.unsqueeze(-1).unsqueeze(-1)
            k_scaled = k_f32 * self._phase_scale_b
            phi = (k_scaled / (1.0 + k_scaled.abs())) * theta
            kvw = k_f32 * p_w_b
            re = kvw * torch.cos(phi)
            im = kvw * torch.sin(phi)

            # Update accumulators in-place for CUDA graph compatibility
            den_acc.add_(p_w)
            re_acc.add_(re)
            im_acc.add_(im)

            # Normalize by accumulated denominator
            inv_den = 1.0 / torch.clamp(den_acc, min=1e-4)
            inv_den = inv_den.unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
            state_re = re_acc * inv_den
            state_im = im_acc * inv_den

            state_re_g = state_re.reshape(B, self.K_q, self.n_rep, self.H, self.M)
            state_im_g = state_im.reshape(B, self.K_q, self.n_rep, self.H, self.M)
            # Scale by 1/sqrt(H) (matches JAX)
            scale = 1.0 / (self.H**0.5)
            match_re = ((state_re_g * q_re + state_im_g * q_im) * scale).float()
            match_im = ((state_im_g * q_re - state_re_g * q_im) * scale).float()
            out_re_g = (match_re * w_int.float()).sum(dim=-1)
            out_im_g = (match_im * w_int.float()).sum(dim=-1)
            out_re = out_re_g.reshape(B, self.K, self.H).to(x_t.dtype)
            out_im = out_im_g.reshape(B, self.K, self.H).to(x_t.dtype)
            out_complex = torch.cat([out_re, out_im], dim=-1)

        out_complex = out_complex.reshape(B, self.K, 2 * self.H)

        # GatedRMSNorm with gate from x_t (matches JAX)
        out_complex_flat = out_complex.reshape(B, -1)
        gate_for_norm = self.gate_proj(x_t)
        out_normed = self.gated_norm(out_complex_flat, gate_for_norm)
        out_complex = out_normed.reshape(B, self.K, 2 * self.H)

        # W_readout -> SwiGLU (no skip)
        y_spec_raw = torch.einsum(
            "bkf,kfn->bkn", out_complex, self.W_readout.to(out_complex.dtype)
        )
        y_val, y_gate = y_spec_raw.chunk(2, dim=-1)
        y_act = y_val * torch.sigmoid(y_gate)

        out = self.out_proj(y_act.reshape(B, -1))

        # Update position in-place for CUDA graph compatibility
        pos.add_(1)

        # Update conv_buffer in-place (shift left and insert new value)
        if self.conv_kernel_size > 1:
            if self.conv_kernel_size > 2:
                conv_buffer[:, :-1, :].copy_(conv_buffer[:, 1:, :].clone())
            conv_buffer[:, -1, :].copy_(z_conv)

        return out, (den_acc, re_acc, im_acc, pos, conv_buffer)


class SeqCondBlock(nn.Module):
    """SeqCond block with residual - matches JAX SeqCondBlock."""

    def __init__(self, d_model: int, norm_eps: float = 1e-6, **kwargs):
        super().__init__()
        self.norm = RMSNorm(d_model, epsilon=norm_eps)
        self.attn = SeqCondAttention(d_model=d_model, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple]]:
        if return_state:
            out, state = self.attn(self.norm(x), mask=mask, return_state=True)
            return x + out, state
        return x + self.attn(self.norm(x), mask=mask)

    def step(
        self, x_t: torch.Tensor, state: Tuple, use_triton: bool = False
    ) -> Tuple[torch.Tensor, Tuple]:
        out, new_state = self.attn.step(self.norm(x_t), state, use_triton=use_triton)
        return x_t + out, new_state

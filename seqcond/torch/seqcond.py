"""
SeqCondAttention - mirrors JAX seqcond_fast.py exactly.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .norm import RMSNorm, GatedRMSNorm


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
        skip_low_rank: bool = True,
        dropout: float = 0.0,
        maxlen: Optional[int] = None,
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
        self.skip_low_rank = skip_low_rank
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
        self.dim_skip = d_model // 4 if skip_low_rank else self.dim_swiglu_total
        self.dim_gate = self.K
        self.dim_mem_total = self.dim_memory + self.K
        self.dim_conv_total = self.dim_mem_total + self.dim_query_total
        self.dim_total = self.dim_conv_total + self.dim_skip + self.dim_gate

        self.in_proj = nn.Linear(d_model, self.dim_total, bias=False)
        self.conv_weight = nn.Parameter(
            torch.empty(self.dim_conv_total, 1, conv_kernel_size)
        )
        nn.init.kaiming_normal_(self.conv_weight)
        self.k_norm = RMSNorm(self.H)
        self.q_norm = RMSNorm(self.dim_query_total)

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
        self.W_readout = nn.Parameter(
            torch.empty(self.K, 2 * self.H, self.dim_swiglu_head)
        )
        nn.init.xavier_uniform_(self.W_readout)
        self.gate_proj = nn.Linear(self.dim_gate, self.K * 2 * self.H, bias=False)
        self.gated_norm = GatedRMSNorm(self.K * 2 * self.H)
        if skip_low_rank:
            self.skip_up = nn.Linear(self.dim_skip, self.dim_swiglu_total, bias=False)
        self.highway_scale = nn.Parameter(torch.ones(1, 1, self.K, 1))
        self.out_proj = nn.Linear(self.dim_swiglu_total // 2, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, D = x.shape
        z_all = self.in_proj(x)
        z_conv = z_all[..., : self.dim_conv_total]
        c_skip = z_all[..., self.dim_conv_total : self.dim_conv_total + self.dim_skip]
        gate_logits = z_all[..., self.dim_conv_total + self.dim_skip :]

        z_conv_t = z_conv.transpose(1, 2)
        z_conv_t = F.pad(z_conv_t, (self.conv_kernel_size - 1, 0))
        z_conv_t = F.conv1d(z_conv_t, self.conv_weight, groups=self.dim_conv_total)
        z_conv = F.silu(z_conv_t.transpose(1, 2))

        z_mem = z_conv[..., : self.dim_mem_total]
        q_raw = z_conv[..., self.dim_mem_total :]
        k_val = z_mem[..., : self.dim_memory].reshape(B, L, self.K, self.H)
        k_val = self.k_norm(k_val)
        s_raw = z_mem[..., self.dim_memory :]

        q_raw = self.q_norm(q_raw)
        q_raw = q_raw.reshape(B, L, self.K_q, 1, self.H, self.M, 2)
        q_re, q_im = q_raw[..., 0], q_raw[..., 1]

        if self.M == 1:
            theta = 0.001 + 2.999 * torch.sigmoid(self.theta_raw)
            w_int = torch.exp(self.w_int_raw.clamp(-5.0, 5.0))
        else:
            theta_d = F.softplus(self.theta_d_raw) + 1e-4
            theta_accum = torch.cumsum(theta_d, dim=-1)
            total_sum = theta_accum[..., -1:]
            theta = 0.001 + (theta_accum / total_sum) * 2.999
            dtheta_raw = theta_accum[..., 1:] - theta_accum[..., :-1]
            dtheta = dtheta_raw * (2.999 / total_sum)
            w_int = torch.cat(
                [
                    dtheta[..., :1] * 0.5,
                    0.5 * (dtheta[..., :-1] + dtheta[..., 1:]),
                    dtheta[..., -1:] * 0.5,
                ],
                dim=-1,
            )
            w_int = w_int.reshape(1, 1, self.K_q, self.n_rep, self.H, self.M)

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
        p_w_content = F.relu(score_raw) ** 2
        p_w = (p_w_content * torch.exp(log_time_weight)).clamp(1e-6, 1000.0)

        k_f32 = k_val.float().unsqueeze(-1)
        p_w_b = p_w.unsqueeze(-1).unsqueeze(-1)
        phase_scale_b = self.phase_scale.view(1, 1, self.K, 1, 1)
        k_scaled = k_f32 * phase_scale_b
        phi = (k_scaled / (1.0 + k_scaled.abs())) * theta
        kvw = k_f32 * p_w_b
        re = kvw * torch.cos(phi)
        im = kvw * torch.sin(phi)

        stack = torch.cat([re.reshape(B, L, -1), im.reshape(B, L, -1)], dim=-1)
        cumsum = torch.cumsum(stack, dim=1)
        flat_size = self.K * self.H * self.M
        re_acc = cumsum[..., :flat_size].reshape(B, L, self.K, self.H, self.M)
        im_acc = cumsum[..., flat_size:].reshape(B, L, self.K, self.H, self.M)

        state_re_g = re_acc.reshape(B, L, self.K_q, self.n_rep, self.H, self.M)
        state_im_g = im_acc.reshape(B, L, self.K_q, self.n_rep, self.H, self.M)
        match_re = (state_re_g * q_re + state_im_g * q_im).float()
        match_im = (state_im_g * q_re - state_re_g * q_im).float()
        out_re_g = (match_re * w_int.float()).sum(dim=-1)
        out_im_g = (match_im * w_int.float()).sum(dim=-1)
        out_re = out_re_g.reshape(B, L, self.K, self.H)
        out_im = out_im_g.reshape(B, L, self.K, self.H)
        out_complex = torch.cat([out_re, out_im], dim=-1)

        out_flat = out_complex.reshape(B, L, -1)
        gate_for_norm = self.gate_proj(gate_logits)
        out_normed = self.gated_norm(out_flat, gate_for_norm)
        out_complex = out_normed.reshape(B, L, self.K, 2 * self.H)

        y_spec_raw = torch.einsum("blkf,kfn->blkn", out_complex, self.W_readout)
        y_skip_raw = self.skip_up(c_skip) if self.skip_low_rank else c_skip
        y_skip = y_skip_raw.reshape(B, L, self.K, self.dim_swiglu_head)

        y_spec_val, y_spec_gate = y_spec_raw.chunk(2, dim=-1)
        y_skip_val, y_skip_gate = y_skip.chunk(2, dim=-1)
        y_val = y_spec_val + y_skip_val * self.highway_scale
        y_gate = y_spec_gate + y_skip_gate * self.highway_scale
        y_act = y_val * torch.sigmoid(y_gate)

        return self.out_proj(y_act.reshape(B, L, -1))

    def step(self, x_t: torch.Tensor, state: Tuple) -> Tuple[torch.Tensor, Tuple]:
        B, D = x_t.shape
        den_acc, re_acc, im_acc, pos, conv_buffer = state

        z_all = self.in_proj(x_t)
        z_conv = z_all[..., : self.dim_conv_total]
        c_skip = z_all[..., self.dim_conv_total : self.dim_conv_total + self.dim_skip]
        gate_logits = z_all[..., self.dim_conv_total + self.dim_skip :]

        z_conv_expanded = z_conv.unsqueeze(1)
        conv_input = torch.cat([conv_buffer, z_conv_expanded], dim=1)
        # conv_weight is (C, 1, K) in PyTorch format, need (K, C) for einsum
        conv_kernel = self.conv_weight[:, 0, :].t()  # (K, C)
        z_conv_out = torch.einsum("bkc,kc->bc", conv_input, conv_kernel)
        z_conv_act = F.silu(z_conv_out)
        conv_buffer_new = torch.cat([conv_buffer[:, 1:, :], z_conv_expanded], dim=1)

        z_mem = z_conv_act[..., : self.dim_mem_total]
        q_raw = z_conv_act[..., self.dim_mem_total :]
        k_val = z_mem[..., : self.dim_memory].reshape(B, self.K, self.H)
        k_val = self.k_norm(k_val)
        s_raw = z_mem[..., self.dim_memory :]

        q_raw = self.q_norm(q_raw)
        q_raw = q_raw.reshape(B, self.K_q, 1, self.H, self.M, 2)
        q_re, q_im = q_raw[..., 0], q_raw[..., 1]

        if self.M == 1:
            theta = (0.001 + 2.999 * torch.sigmoid(self.theta_raw))[0, 0]
            w_int = torch.exp(self.w_int_raw.clamp(-5.0, 5.0))[0, 0]
        else:
            theta_d = F.softplus(self.theta_d_raw) + 1e-4
            theta_accum = torch.cumsum(theta_d, dim=-1)
            total_sum = theta_accum[..., -1:]
            theta = (0.001 + (theta_accum / total_sum) * 2.999)[0, 0]
            dtheta_raw = theta_accum[..., 1:] - theta_accum[..., :-1]
            dtheta = dtheta_raw * (2.999 / total_sum)
            w_int = torch.cat(
                [
                    dtheta[..., :1] * 0.5,
                    0.5 * (dtheta[..., :-1] + dtheta[..., 1:]),
                    dtheta[..., -1:] * 0.5,
                ],
                dim=-1,
            )
            w_int = w_int.reshape(1, 1, self.K_q, self.n_rep, self.H, self.M)[0, 0]

        log_w_list = []
        if self.num_decay_heads > 0:
            slopes = F.softplus(self.decay_slopes).view(1, -1)
            dist = torch.clamp((self.maxlen or 2048) - 1 - pos.unsqueeze(-1), min=0.0)
            log_w_list.append(-slopes * dist)
        if self.num_anchor_heads > 0:
            slopes_a = F.softplus(self.anchor_slopes).view(1, -1)
            log_w_list.append(-slopes_a * pos.unsqueeze(-1))
        log_time_weight = (
            torch.cat(log_w_list, dim=1)
            if log_w_list
            else torch.zeros(B, self.K, device=x_t.device)
        )

        score_raw = self.score_scale.view(1, -1) * s_raw.float() + self.score_bias.view(
            1, -1
        )
        p_w = (F.relu(score_raw) ** 2 * torch.exp(log_time_weight)).clamp(1e-6, 1000.0)

        k_f32 = k_val.float().unsqueeze(-1)
        p_w_b = p_w.unsqueeze(-1).unsqueeze(-1)
        phase_scale_b = self.phase_scale.view(1, self.K, 1, 1)
        k_scaled = k_f32 * phase_scale_b
        phi = (k_scaled / (1.0 + k_scaled.abs())) * theta
        kvw = k_f32 * p_w_b
        re = kvw * torch.cos(phi)
        im = kvw * torch.sin(phi)

        den_acc_new = den_acc + p_w
        re_acc_new = re_acc + re
        im_acc_new = im_acc + im

        state_re_g = re_acc_new.reshape(B, self.K_q, self.n_rep, self.H, self.M)
        state_im_g = im_acc_new.reshape(B, self.K_q, self.n_rep, self.H, self.M)
        match_re = (state_re_g * q_re + state_im_g * q_im).float()
        match_im = (state_im_g * q_re - state_re_g * q_im).float()
        out_re_g = (match_re * w_int.float()).sum(dim=-1)
        out_im_g = (match_im * w_int.float()).sum(dim=-1)
        out_re = out_re_g.reshape(B, self.K, self.H)
        out_im = out_im_g.reshape(B, self.K, self.H)
        out_complex = torch.cat([out_re, out_im], dim=-1)

        out_flat = out_complex.reshape(B, -1)
        gate_for_norm = self.gate_proj(gate_logits)
        out_normed = self.gated_norm(out_flat, gate_for_norm)
        out_complex = out_normed.reshape(B, self.K, 2 * self.H)

        y_spec_raw = torch.einsum("bkf,kfn->bkn", out_complex, self.W_readout)
        y_skip_raw = self.skip_up(c_skip) if self.skip_low_rank else c_skip
        y_skip = y_skip_raw.reshape(B, self.K, self.dim_swiglu_head)

        y_spec_val, y_spec_gate = y_spec_raw.chunk(2, dim=-1)
        y_skip_val, y_skip_gate = y_skip.chunk(2, dim=-1)
        h_scale = self.highway_scale[0, 0]
        y_val = y_spec_val + y_skip_val * h_scale
        y_gate = y_spec_gate + y_skip_gate * h_scale
        y_act = y_val * torch.sigmoid(y_gate)

        out = self.out_proj(y_act.reshape(B, -1))
        pos_new = pos + 1
        return out, (den_acc_new, re_acc_new, im_acc_new, pos_new, conv_buffer_new)


class SeqCondBlock(nn.Module):
    """SeqCond block with residual - matches JAX SeqCondBlock."""

    def __init__(self, d_model: int, norm_eps: float = 1e-6, **kwargs):
        super().__init__()
        self.norm = RMSNorm(d_model, epsilon=norm_eps)
        self.attn = SeqCondAttention(d_model=d_model, **kwargs)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return x + self.attn(self.norm(x), mask=mask)

    def step(self, x_t: torch.Tensor, state: Tuple) -> Tuple[torch.Tensor, Tuple]:
        out, new_state = self.attn.step(self.norm(x_t), state)
        return x_t + out, new_state

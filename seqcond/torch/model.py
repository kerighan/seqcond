import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class GatedRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, residual):
        # x: (B, D) - flattened
        # residual: (B, D) - flattened

        x = x.float()
        res = residual.float()

        # Gate with silu FIRST
        x = x * F.silu(res)

        # RMSNorm
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x * self.weight

        return x


class RotaryEmbedding(nn.Module):
    """RoPE implementation matching the specific configuration used."""

    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: int = 10000
    ):
        super().__init__()
        # dim is head_dim, we compute frequencies for half of it
        half_dim = dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        self.half_dim = half_dim

        # Precompute cos/sin: (maxlen, half_dim)
        pos = torch.arange(self.max_seq_len_cached).float()
        angles = pos[:, None] * inv_freq[None, :]
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        # Return (seq_len, half_dim) tensors
        if seq_len > self.max_seq_len_cached:
            pos = torch.arange(seq_len, device=self.inv_freq.device).float()
            angles = pos[:, None] * self.inv_freq[None, :]
            return angles.cos(), angles.sin()
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE to query and key tensors.

    q, k: [B, H, L, D] where D is head_dim
    cos, sin: [B, L, H, D//2]
    """
    dim = q.shape[-1] // 2
    # Crop cos/sin to match tensor dimension
    cos = cos[..., :dim]
    sin = sin[..., :dim]

    # Split into two halves
    q1, q2 = q[..., :dim], q[..., dim:]
    k1, k2 = k[..., :dim], k[..., dim:]

    # Apply rotation
    q_embed = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_embed = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_embed, k_embed


class TransformerDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = self.d_model // self.num_heads
        self.num_kv_heads = config.num_kv_heads or self.num_heads
        self.n_rep = self.num_heads // self.num_kv_heads

        self.norm = RMSNorm(self.d_model, eps=config.qk_norm_eps)
        self.q_proj = nn.Linear(
            self.d_model, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.d_model, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.d_model, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.d_model, bias=False
        )

        self.ffn_norm = RMSNorm(self.d_model, eps=config.qk_norm_eps)
        # Transformer FFN has bias in Flax checkpoint
        self.ffn_gate = nn.Linear(self.d_model, config.d_ff, bias=True)
        self.ffn_up = nn.Linear(self.d_model, config.d_ff, bias=True)
        self.ffn_down = nn.Linear(config.d_ff, self.d_model, bias=True)

        # qk_norm for transformer blocks
        self.qk_norm = config.qk_norm
        self.qk_norm_eps = config.qk_norm_eps

    def step(self, x_t, state, cos_t, sin_t):
        # x_t: [B, D]
        # state: (k_cache, v_cache, pos)
        # cos_t, sin_t: [B, 1, num_heads, half_dim]
        k_cache, v_cache, pos = state
        B = x_t.size(0)

        h = self.norm(x_t)
        q = self.q_proj(h).view(B, self.num_heads, 1, self.head_dim)
        k = self.k_proj(h).view(B, self.num_kv_heads, 1, self.head_dim)
        v = self.v_proj(h).view(B, self.num_kv_heads, 1, self.head_dim)

        # Apply RoPE - transpose cos/sin to match q/k format [B, H, L, D//2]
        cos_t_rope = cos_t.permute(0, 2, 1, 3)  # [B, num_heads, 1, half_dim]
        sin_t_rope = sin_t.permute(0, 2, 1, 3)

        # For KV heads, slice if needed
        if self.num_kv_heads < self.num_heads:
            cos_t_kv = cos_t_rope[:, : self.num_kv_heads, :, :]
            sin_t_kv = sin_t_rope[:, : self.num_kv_heads, :, :]
        else:
            cos_t_kv = cos_t_rope
            sin_t_kv = sin_t_rope

        q, _ = apply_rotary_pos_emb(q, q, cos_t_rope, sin_t_rope)
        k, _ = apply_rotary_pos_emb(k, k, cos_t_kv, sin_t_kv)

        # QK Norm (stateless)
        if self.qk_norm:
            q_f32 = q.float()
            k_f32 = k.float()
            q_ms = q_f32.pow(2).mean(-1, keepdim=True)
            k_ms = k_f32.pow(2).mean(-1, keepdim=True)
            q = (q_f32 * torch.rsqrt(q_ms + self.qk_norm_eps)).to(q.dtype)
            k = (k_f32 * torch.rsqrt(k_ms + self.qk_norm_eps)).to(k.dtype)

        # Update KV cache (In-place & Static Shape friendly)
        # We assume positions are synchronized across batch for this implementation
        # k: (B, H, 1, D)
        # k_cache: (B, H, L, D)
        # pos: (B,)

        # Use index_copy_ for CUDA Graph compatibility
        # We assume all batch elements are at the same position 'pos[0]'
        # This is true for standard generation.
        # We need pos to be 1D tensor of indices to write to on dim 2.
        # Since we write 1 element on dim 2, we need an index tensor of size (1,).

        # Take the first element of pos as the index for everyone
        idx_tensor = pos[0:1]  # (1,)

        k_cache.index_copy_(2, idx_tensor, k)
        v_cache.index_copy_(2, idx_tensor, v)

        # Attention with Static Shape (Masking instead of Slicing)
        # k_cache, v_cache are full size (L).

        # Prepare causal mask
        # We want to attend to [0, ..., pos].
        # Mask should be True for positions > pos.
        all_pos = torch.arange(
            self.d_model // self.num_heads * 0 + k_cache.size(2), device=x_t.device
        )  # range(L)
        # (1, 1, 1, L) compared to (B, 1, 1, 1)
        # mask: (B, 1, 1, L)
        mask = all_pos.reshape(1, 1, 1, -1) > pos.reshape(B, 1, 1, 1)

        # GQA repeat (Virtual, use expand instead of repeat to save memory?)
        # SDPA handles GQA if we pass correct shapes?
        # Manual attention:
        k_c = k_cache
        v_c = v_cache

        if self.n_rep > 1:
            # repeat_interleave creates copy. usage of expand is preferred if possible but matmul requires matching dims.
            k_c = k_c.repeat_interleave(self.n_rep, dim=1)
            v_c = v_c.repeat_interleave(self.n_rep, dim=1)

        attn_weights = torch.matmul(q, k_c.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )  # (B, H, 1, L)

        # Apply mask
        attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_probs, v_c)  # (B, H, 1, D)

        attn_out = attn_out.reshape(B, self.d_model)
        h = x_t + self.o_proj(attn_out)

        # FFN
        f = self.ffn_norm(h)
        # JAX SwiGLU: u, v = split(ff_in(y)); out = swish(v) * u
        # Here: ffn_gate is u (first half), ffn_up is v (second half)
        # So we want: self.ffn_gate(f) * F.silu(self.ffn_up(f))
        ffn_out = self.ffn_down(self.ffn_gate(f) * F.silu(self.ffn_up(f)))
        out = h + ffn_out

        # Update pos in-place
        pos.add_(1)

        return out, (k_cache, v_cache, pos)


class SeqCondAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.seqcond_heads  # K = 30
        self.num_query_heads = self.num_heads  # Kq = 30 (from checkpoint analysis)
        self.n_rep = self.num_heads // self.num_query_heads  # 1
        self.num_anchor_heads = config.num_anchor_heads
        self.num_decay_heads = self.num_heads - self.num_anchor_heads
        self.num_thetas = config.num_thetas  # M = 4
        self.conv_kernel_size = config.conv_kernel_size
        self.expand_factor = config.expand_factor
        self.out_expand_factor = config.out_expand_factor
        self.skip_low_rank = False  # Matches JAX checkpoint configuration
        self.maxlen = config.maxlen or 2048

        d_model = config.d_model
        # H = 16 (from k_norm/scale analysis)
        self.H = 16

        self.dim_memory = self.num_heads * self.H  # 30 * 16 = 480
        self.dim_query_head = self.H * self.num_thetas * 2  # 16 * 4 * 2 = 128
        self.dim_query_total = (
            self.num_query_heads * self.dim_query_head
        )  # 30 * 128 = 3840

        self.dim_mem_total = self.dim_memory + self.num_heads  # 480 + 30 = 510
        self.dim_conv_total = (
            self.dim_mem_total + self.dim_query_total
        )  # 510 + 3840 = 4350
        self.dim_skip = 1920  # Matches checkpoint
        self.dim_gate = self.num_heads  # 30 (from checkpoint gate_proj shape)
        self.dim_total = (
            self.dim_conv_total + self.dim_skip + self.dim_gate
        )  # 4350 + 1920 + 30 = 6300
        self.dim_swiglu_head = 64
        self.dim_swiglu_total = self.num_heads * self.dim_swiglu_head  # 30 * 64 = 1920

        self.in_proj = nn.Linear(d_model, self.dim_total, bias=False)
        self.conv_weight = nn.Parameter(
            torch.randn(self.dim_conv_total, 1, self.conv_kernel_size)
        )

        self.q_norm = RMSNorm(self.dim_query_total, eps=config.qk_norm_eps)
        # k_norm is per-head: RMSNorm on H=16, not on full dim_memory=480
        self.k_norm = RMSNorm(self.H, eps=config.qk_norm_eps)

        self.score_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.score_scale = nn.Parameter(torch.ones(self.num_heads))
        self.phase_scale = nn.Parameter(torch.ones(self.num_heads))

        self.decay_slopes = nn.Parameter(torch.empty(self.num_decay_heads))
        if self.num_anchor_heads > 0:
            self.anchor_slopes = nn.Parameter(torch.empty(self.num_anchor_heads))

        # theta_d_raw: (1, 1, K, H, M) for multi-theta frequency integration
        self.theta_d_raw = nn.Parameter(
            torch.empty(1, 1, self.num_heads, self.H, self.num_thetas)
        )

        # gate_proj: (30, 960) in checkpoint -> input=num_heads, output=960
        self.gate_proj = nn.Linear(self.num_heads, 960, bias=False)
        self.gated_norm = GatedRMSNorm(960, eps=1e-5)  # JAX uses 1e-5 by default

        self.W_readout = nn.Parameter(
            torch.empty(self.num_heads, self.H * 2, self.dim_swiglu_head)
        )

        self.highway_scale = nn.Parameter(torch.ones(1, 1, self.num_heads, 1))
        self.out_proj = nn.Linear(960, d_model, bias=False)

    def step(self, x_t, state):
        den_acc, re_acc, im_acc, pos, conv_buffer = state
        B = x_t.size(0)

        z_all = self.in_proj(x_t)  # (B, 6300)

        z_conv = z_all[:, : self.dim_conv_total]  # (B, 4350)
        c_skip = z_all[
            :, self.dim_conv_total : self.dim_conv_total + self.dim_skip
        ]  # (B, 1920)
        gate_logits = z_all[:, self.dim_conv_total + self.dim_skip :]  # (B, 30)

        # Conv step - match JAX layout: (B, K, C)
        z_conv_expanded = z_conv.unsqueeze(1)  # (B, 1, dim_conv_total)

        # Prepare input for convolution: buffer + new input
        # We need a temporary tensor for the convolution input since we can't easily do it in-place
        # without potentially overwriting what we need if we are not careful.
        # However, for CUDA graphs, temporary allocations inside the graph are fine as long as they are deterministic size.
        # But `torch.cat` allocates new memory.

        # conv_input = torch.cat([conv_buffer, z_conv_expanded], dim=1)
        # Optimization: Avoid allocating conv_input if possible, or pre-allocate.
        # But `conv_buffer` is (K-1). `conv_input` is (K).
        # Let's allocate conv_input (allocator will handle it efficiently in graph capture).
        conv_input = torch.cat([conv_buffer, z_conv_expanded], dim=1)

        # Update conv_buffer in-place for next step
        # Shift left: buffer[:, :-1] = buffer[:, 1:]
        # clone() is needed because input and output overlap in memory
        if self.conv_kernel_size > 1:
            if self.conv_kernel_size > 2:
                conv_buffer[:, :-1, :].copy_(conv_buffer[:, 1:, :].clone())
            conv_buffer[:, -1, :].copy_(z_conv_expanded.squeeze(1))

        # Manual depthwise conv: (B, K, C) * (K, C) -> (B, C)
        # conv_weight is (C, 1, K), need (K, C) for einsum
        conv_kernel = self.conv_weight.squeeze(1).t()  # (K, C)
        z_conv_out = torch.einsum("bkc,kc->bc", conv_input, conv_kernel)
        z_conv_act = F.silu(z_conv_out)

        # Split z_conv_act: dim_mem_total = 510, dim_query_total = 3840
        z_mem = z_conv_act[:, : self.dim_mem_total]  # (B, 510)
        q_raw = z_conv_act[:, self.dim_mem_total :]  # (B, 3840)

        k_val = z_mem[:, : self.dim_memory].reshape(
            B, self.num_heads, self.H
        )  # (B, 30, 16)
        k_val = self.k_norm(k_val)
        s_raw = z_mem[:, self.dim_memory :]  # (B, 30)

        q_raw_norm = self.q_norm(q_raw)  # (B, 3840)
        q_view = q_raw_norm.view(B, self.num_query_heads, 1, self.H, self.num_thetas, 2)
        q_re, q_im = q_view[..., 0], q_view[..., 1]  # (B, 30, 1, 16, 4)

        if self.num_thetas == 1:
            theta = 0.001 + 2.999 * torch.sigmoid(self.theta_raw)
            # theta: (1, 1, K, H, 1) -> (K, H, 1)
            theta = theta[0, 0]
            w_int = torch.exp(self.w_int_raw.clamp(-5.0, 5.0))
        else:
            # theta_d_raw: (1, 1, K, H, M)
            theta_d = F.softplus(self.theta_d_raw) + 1e-4
            theta_accum = theta_d.cumsum(dim=-1)
            total_sum = theta_accum[..., -1:]

            scale_range = 2.999
            theta = 0.001 + (theta_accum / total_sum) * scale_range

            # Compute w_int from theta_accum (matches JAX trapz integration)
            dtheta_raw = theta_accum[..., 1:] - theta_accum[..., :-1]
            dtheta = dtheta_raw * (scale_range / total_sum)
            w0 = dtheta[..., :1] * 0.5
            w_mid = 0.5 * (dtheta[..., :-1] + dtheta[..., 1:])
            wL = dtheta[..., -1:] * 0.5
            w_int = torch.cat([w0, w_mid, wL], dim=-1)
            # w_int: (1, 1, K, H, M) -> reshape to (1, 1, K, 1, H, M) for GQA compat
            w_int = w_int.view(
                1, 1, self.num_query_heads, self.n_rep, self.H, self.num_thetas
            )

            theta = theta[0, 0]  # (K, H, M)

        log_w_list = []
        if self.num_decay_heads > 0:
            slopes = F.softplus(self.decay_slopes).view(1, -1)
            # Match JAX: dist = maxlen - 1 - pos
            dist = (float(self.maxlen) - 1.0 - pos.float()).clamp(min=0.0).view(-1, 1)
            log_w_list.append(-slopes * dist)
        if self.num_anchor_heads > 0:
            slopes_a = F.softplus(self.anchor_slopes).view(1, -1)
            log_w_list.append(-slopes_a * pos.float().view(-1, 1))

        log_time_weight = (
            torch.cat(log_w_list, dim=1)
            if log_w_list
            else torch.zeros((B, self.num_heads), device=x_t.device)
        )

        score_raw = self.score_scale * s_raw + self.score_bias
        p_w_content = F.relu(score_raw).pow(2)
        p_w = (p_w_content * torch.exp(log_time_weight)).clamp(1e-6, 1000.0)

        k_scaled = k_val.unsqueeze(-1) * self.phase_scale.view(1, self.num_heads, 1, 1)
        phi = (k_scaled / (1.0 + torch.abs(k_scaled))) * theta  # (B, K, H, M)
        kvw = k_val.unsqueeze(-1) * p_w.view(B, self.num_heads, 1, 1)

        re = kvw * torch.cos(phi)
        im = kvw * torch.sin(phi)

        # Update accumulators in-place
        den_acc.add_(p_w)
        re_acc.add_(re)
        im_acc.add_(im)

        # Integration matching JAX: match_re = state_re * q_re + state_im * q_im
        # Group state for GQA: (B, K, H, M) -> (B, K_q, n_rep, H, M)
        state_re_grouped = re_acc.view(
            B, self.num_query_heads, self.n_rep, self.H, self.num_thetas
        )
        state_im_grouped = im_acc.view(
            B, self.num_query_heads, self.n_rep, self.H, self.num_thetas
        )

        # q_re, q_im: (B, K_q, 1, H, M)
        # state_re_grouped, state_im_grouped: (B, K_q, n_rep, H, M)
        match_re = (state_re_grouped * q_re + state_im_grouped * q_im).float()
        match_im = (state_im_grouped * q_re - state_re_grouped * q_im).float()

        # Sum over M (integration)
        # w_int is (1, 1, K_q, n_rep, H, M), get w_int_step = (K_q, n_rep, H, M)
        w_int_step = w_int[0, 0]  # (K_q, n_rep, H, M)
        out_re_g = (match_re * w_int_step).sum(dim=-1)  # (B, K_q, n_rep, H)
        out_im_g = (match_im * w_int_step).sum(dim=-1)

        # Reshape to (B, K, H)
        out_re_g = out_re_g.view(B, self.num_heads, self.H)
        out_im_g = out_im_g.view(B, self.num_heads, self.H)

        out_re = out_re_g.to(x_t.dtype)
        out_im = out_im_g.to(x_t.dtype)

        # out_complex: (B, K, 2H)
        out_complex = torch.cat([out_re, out_im], dim=-1)

        # Gated Norm & Readout
        gate_for_norm = self.gate_proj(gate_logits)  # (B, 960)
        out_normed = self.gated_norm(
            out_complex.reshape(B, -1), gate_for_norm
        )  # (B, 960)
        out_normed = out_normed.view(B, self.num_heads, self.H * 2)

        # Spectral branch output: (B, K, dim_swiglu_head)
        y_spec = torch.einsum("bkf,kfn->bkn", out_normed, self.W_readout)

        # Skip connection: (B, K, dim_swiglu_head)
        y_skip = c_skip.view(B, self.num_heads, self.dim_swiglu_head)

        # Split into val/gate for SwiGLU fusion
        y_spec_val, y_spec_gate = y_spec.chunk(2, dim=-1)
        y_skip_val, y_skip_gate = y_skip.chunk(2, dim=-1)

        # Apply highway_scale (K, 1)
        # self.highway_scale is (1, 1, K, 1)
        h_scale = self.highway_scale.view(self.num_heads, 1)

        y_val = y_spec_val + (y_skip_val * h_scale)
        y_gate = y_spec_gate + (y_skip_gate * h_scale)

        # SwiGLU activation
        y_act = y_val * torch.sigmoid(y_gate)

        # Final output projection
        out = self.out_proj(y_act.reshape(B, -1))

        # Update pos in-place
        pos.add_(1)

        return out, (den_acc, re_acc, im_acc, pos, conv_buffer)


class SeqCondBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # JAX uses norm_eps=1e-5 by default for SeqCondBlock
        self.norm = RMSNorm(config.d_model, eps=1e-5)
        self.attn = SeqCondAttention(config)

    def step(self, x_t, state):
        h = self.norm(x_t)
        h, new_state = self.attn.step(h, state)
        return x_t + h, new_state


class SeqCondModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size
        self.maxlen = config.maxlen or 2048
        self.num_heads = config.num_heads

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.rotary_emb = RotaryEmbedding(
            self.d_model // self.num_heads, max_position_embeddings=self.maxlen
        )

        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            if (i + 1) % (config.seqcond_ratio + 1) == 0:
                self.blocks.append(TransformerDecoderBlock(config))
            else:
                self.blocks.append(SeqCondBlock(config))

        self.output_projection = nn.Linear(self.d_model, self.vocab_size, bias=False)
        if config.tie_weights:
            self.output_projection.weight = self.embedding.weight

    def step(self, token_id, states):
        # token_id: [B, 1]
        B = token_id.size(0)
        x = self.embedding(token_id).squeeze(1)  # [B, D]

        # Position from first block
        if isinstance(states[0], tuple) and len(states[0]) == 5:  # SeqCond
            pos = states[0][3]
        else:  # Transformer
            pos = states[0][2]

        # Get RoPE for current position - match JAX format (B, 1, num_heads, half_dim)
        cos_all, sin_all = self.rotary_emb(x, seq_len=self.maxlen)  # (maxlen, half_dim)

        # Use tensor indexing to avoid CPU sync (.item()) for CUDA Graphs
        # pos is (B,). cos_all is (L, D). We want (B, D).
        cos_t = cos_all[pos].unsqueeze(1)  # (B, 1, half_dim)
        sin_t = sin_all[pos].unsqueeze(1)

        # Expand to (B, 1, num_heads, half_dim)
        cos_t = cos_t.unsqueeze(2).expand(B, 1, self.num_heads, -1)
        sin_t = sin_t.unsqueeze(2).expand(B, 1, self.num_heads, -1)

        new_states = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, TransformerDecoderBlock):
                x, new_s = block.step(x, states[i], cos_t, sin_t)
            else:
                x, new_s = block.step(x, states[i])
            new_states.append(new_s)

        logits = self.output_projection(x)
        return logits, new_states

    def init_state(self, batch_size: int = 1, device="cuda"):
        states = []
        for block in self.blocks:
            if isinstance(block, TransformerDecoderBlock):
                k_cache = torch.zeros(
                    batch_size,
                    block.num_kv_heads,
                    self.maxlen,
                    block.head_dim,
                    device=device,
                )
                v_cache = torch.zeros(
                    batch_size,
                    block.num_kv_heads,
                    self.maxlen,
                    block.head_dim,
                    device=device,
                )
                pos = torch.zeros(batch_size, dtype=torch.long, device=device)
                states.append((k_cache, v_cache, pos))
            else:
                attn = block.attn
                den_acc = torch.zeros(batch_size, attn.num_heads, device=device)
                re_acc = torch.zeros(
                    batch_size, attn.num_heads, attn.H, attn.num_thetas, device=device
                )
                im_acc = torch.zeros(
                    batch_size, attn.num_heads, attn.H, attn.num_thetas, device=device
                )
                pos = torch.zeros(batch_size, dtype=torch.long, device=device)
                # conv_buffer: (B, kernel_size-1, dim_conv_total) to match JAX
                conv_buffer = torch.zeros(
                    batch_size,
                    attn.conv_kernel_size - 1,
                    attn.dim_conv_total,
                    device=device,
                )
                states.append((den_acc, re_acc, im_acc, pos, conv_buffer))
        return states

"""
Triton kernels for SeqCond acceleration.

These kernels fuse multiple operations in the SeqCond attention step.
"""

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            # Single head per block - good for small H
            triton.Config({"BLOCK_M": 32, "BLOCK_H": 1}, num_warps=2, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_H": 1}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_H": 1}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_H": 1}, num_warps=8, num_stages=2),
            # Multiple heads per block - better parallelism
            triton.Config({"BLOCK_M": 32, "BLOCK_H": 2}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_H": 2}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_H": 4}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_H": 4}, num_warps=8, num_stages=2),
            # Larger M blocks for better memory coalescing
            triton.Config({"BLOCK_M": 64, "BLOCK_H": 1}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_M": 128, "BLOCK_H": 1}, num_warps=4, num_stages=3),
            # All heads at once if H is small
            triton.Config({"BLOCK_M": 32, "BLOCK_H": 8}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 64, "BLOCK_H": 8}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_H": 16}, num_warps=8, num_stages=2),
        ],
        key=["M", "H"],
    )
    @triton.jit
    def _seqcond_fully_fused_kernel(
        # Inputs
        k_ptr,  # (B, K, H)
        s_raw_ptr,  # (B, K)
        q_re_ptr,  # (B, K_q, H, M)
        q_im_ptr,  # (B, K_q, H, M)
        # Accumulators (in-place update)
        re_acc_ptr,  # (B, K, H, M)
        im_acc_ptr,  # (B, K, H, M)
        den_acc_ptr,  # (B, K)
        # Parameters
        theta_ptr,  # (K, H, M)
        w_int_ptr,  # (K_q, H, M)
        phase_scale_ptr,  # (K,)
        score_scale_ptr,  # (K,)
        score_bias_ptr,  # (K,)
        log_tw_ptr,  # (B, K)
        # Outputs
        out_re_ptr,  # (B, K, H)
        out_im_ptr,  # (B, K, H)
        # Dimensions
        K: tl.constexpr,
        H: tl.constexpr,
        M: tl.constexpr,
        # Strides for k
        stride_k_b,
        stride_k_k,
        stride_k_h,
        # Strides for accumulators
        stride_acc_b,
        stride_acc_k,
        stride_acc_h,
        stride_acc_m,
        # Strides for theta
        stride_theta_k,
        stride_theta_h,
        stride_theta_m,
        # Strides for query
        stride_q_b,
        stride_q_k,
        stride_q_h,
        stride_q_m,
        # Strides for w_int
        stride_w_k,
        stride_w_h,
        stride_w_m,
        # Strides for output
        stride_out_b,
        stride_out_k,
        stride_out_h,
        # Block sizes
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        """
        Fully fused single-kernel: computes p_w, phi_base, updates accumulators, and does query matching.
        Grid: (B * K * ceil(H/BLOCK_H),)
        Each program handles one (batch, k) and BLOCK_H heads, processing all M.
        """
        pid = tl.program_id(0)
        num_h_blocks = (H + BLOCK_H - 1) // BLOCK_H

        # Decode indices
        b = pid // (K * num_h_blocks)
        rem = pid % (K * num_h_blocks)
        k = rem // num_h_blocks
        h_block = rem % num_h_blocks
        h_start = h_block * BLOCK_H

        # === Compute p_w once per (b, k) - only first h_block does den_acc update ===
        s_raw = tl.load(s_raw_ptr + b * K + k)
        score_scale = tl.load(score_scale_ptr + k)
        score_bias = tl.load(score_bias_ptr + k)
        log_tw = tl.load(log_tw_ptr + b * K + k)
        phase_scale = tl.load(phase_scale_ptr + k)

        score = score_scale * s_raw + score_bias
        score = tl.minimum(tl.maximum(score, -20.0), 20.0)
        # Softplus: log(1 + exp(x)) - more stable than ReLU^2
        p_w_content = tl.log(1.0 + tl.exp(score))
        p_w = p_w_content * tl.exp(log_tw)
        p_w = tl.minimum(tl.maximum(p_w, 1e-6), 1000.0)

        # Update den_acc (only once per (b, k) - use h_block == 0)
        if h_block == 0:
            old_den = tl.load(den_acc_ptr + b * K + k)
            tl.store(den_acc_ptr + b * K + k, old_den + p_w)

        # Process BLOCK_H heads
        offs_h = tl.arange(0, BLOCK_H)
        h_idx = h_start + offs_h
        h_mask = h_idx < H

        # Load k_val for all heads in this block
        k_val = tl.load(
            k_ptr + b * stride_k_b + k * stride_k_k + h_idx * stride_k_h,
            mask=h_mask,
            other=0.0,
        )

        # Compute phi_base for all heads
        k_scaled = k_val * phase_scale
        phi_base = k_scaled / (1.0 + tl.abs(k_scaled))

        # kvw for accumulator update (BLOCK_H,)
        kvw = k_val * p_w

        # Accumulators for output sum over M (BLOCK_H,)
        sum_re = tl.zeros((BLOCK_H,), dtype=tl.float32)
        sum_im = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Process M dimension in blocks
        offs_m = tl.arange(0, BLOCK_M)

        for m_start in range(0, M, BLOCK_M):
            m_idx = m_start + offs_m
            m_mask = m_idx < M

            # Load theta: shape (BLOCK_H, BLOCK_M)
            theta_base = k * stride_theta_k
            theta_vals = tl.load(
                theta_ptr
                + theta_base
                + h_idx[:, None] * stride_theta_h
                + m_idx[None, :] * stride_theta_m,
                mask=h_mask[:, None] & m_mask[None, :],
                other=0.0,
            )

            # Compute phi and sin/cos: (BLOCK_H, BLOCK_M)
            # Keep sin/cos separate to allow GPU dual-issue on SFU
            phi = phi_base[:, None] * theta_vals
            cos_phi = tl.cos(phi)
            sin_phi = tl.sin(phi)

            # Load accumulators: (BLOCK_H, BLOCK_M)
            acc_base = b * stride_acc_b + k * stride_acc_k
            old_re = tl.load(
                re_acc_ptr
                + acc_base
                + h_idx[:, None] * stride_acc_h
                + m_idx[None, :] * stride_acc_m,
                mask=h_mask[:, None] & m_mask[None, :],
                other=0.0,
            )
            old_im = tl.load(
                im_acc_ptr
                + acc_base
                + h_idx[:, None] * stride_acc_h
                + m_idx[None, :] * stride_acc_m,
                mask=h_mask[:, None] & m_mask[None, :],
                other=0.0,
            )

            # Update accumulators: (BLOCK_H, BLOCK_M)
            new_re = old_re + kvw[:, None] * cos_phi
            new_im = old_im + kvw[:, None] * sin_phi

            tl.store(
                re_acc_ptr
                + acc_base
                + h_idx[:, None] * stride_acc_h
                + m_idx[None, :] * stride_acc_m,
                new_re,
                mask=h_mask[:, None] & m_mask[None, :],
            )
            tl.store(
                im_acc_ptr
                + acc_base
                + h_idx[:, None] * stride_acc_h
                + m_idx[None, :] * stride_acc_m,
                new_im,
                mask=h_mask[:, None] & m_mask[None, :],
            )

            # Load query: (BLOCK_H, BLOCK_M)
            q_base = b * stride_q_b + k * stride_q_k
            q_re_vals = tl.load(
                q_re_ptr
                + q_base
                + h_idx[:, None] * stride_q_h
                + m_idx[None, :] * stride_q_m,
                mask=h_mask[:, None] & m_mask[None, :],
                other=0.0,
            )
            q_im_vals = tl.load(
                q_im_ptr
                + q_base
                + h_idx[:, None] * stride_q_h
                + m_idx[None, :] * stride_q_m,
                mask=h_mask[:, None] & m_mask[None, :],
                other=0.0,
            )

            # Load w_int: (BLOCK_H, BLOCK_M)
            w_base = k * stride_w_k
            w_vals = tl.load(
                w_int_ptr
                + w_base
                + h_idx[:, None] * stride_w_h
                + m_idx[None, :] * stride_w_m,
                mask=h_mask[:, None] & m_mask[None, :],
                other=0.0,
            )

            # Normalize by accumulated denominator (load updated den_acc)
            new_den = tl.load(den_acc_ptr + b * K + k)
            inv_den = 1.0 / tl.maximum(new_den, 1e-4)
            state_re = new_re * inv_den
            state_im = new_im * inv_den

            # Complex multiplication with 1/sqrt(H) scaling: (BLOCK_H, BLOCK_M)
            scale = 1.0 / tl.sqrt(float(H))
            match_re = (state_re * q_re_vals + state_im * q_im_vals) * scale
            match_im = (state_im * q_re_vals - state_re * q_im_vals) * scale

            # Sum over M dimension: (BLOCK_H,)
            sum_re += tl.sum(match_re * w_vals, axis=1)
            sum_im += tl.sum(match_im * w_vals, axis=1)

        # Store outputs for each head
        out_base = b * stride_out_b + k * stride_out_k
        tl.store(
            out_re_ptr + out_base + h_idx * stride_out_h,
            sum_re,
            mask=h_mask,
        )
        tl.store(
            out_im_ptr + out_base + h_idx * stride_out_h,
            sum_im,
            mask=h_mask,
        )


def seqcond_step_triton(
    k_val: torch.Tensor,  # (B, K, H)
    s_raw: torch.Tensor,  # (B, K)
    q_re: torch.Tensor,  # (B, K_q, H, M) - already squeezed
    q_im: torch.Tensor,  # (B, K_q, H, M)
    re_acc: torch.Tensor,  # (B, K, H, M) - modified in-place
    im_acc: torch.Tensor,  # (B, K, H, M) - modified in-place
    den_acc: torch.Tensor,  # (B, K) - modified in-place
    theta: torch.Tensor,  # (K, H, M)
    w_int: torch.Tensor,  # (K_q, H, M) - already squeezed
    phase_scale: torch.Tensor,  # (K,)
    score_scale: torch.Tensor,  # (K,)
    score_bias: torch.Tensor,  # (K,)
    log_time_weight: torch.Tensor,  # (B, K)
) -> tuple:
    """
    Fully fused Triton implementation of SeqCond step.
    Single kernel launch that computes everything:
    - p_w and phi_base computation
    - Accumulator updates
    - Query matching
    - den_acc update

    Returns:
        out_re: (B, K, H) - real part of output
        out_im: (B, K, H) - imaginary part of output
    """
    B, K, H = k_val.shape
    M = theta.shape[2]

    # Ensure contiguous and float32
    k_val = k_val.contiguous().float()
    s_raw = s_raw.contiguous().float()
    q_re = q_re.contiguous().float()
    q_im = q_im.contiguous().float()
    theta = theta.contiguous().float()
    phase_scale = phase_scale.contiguous().float()
    score_scale = score_scale.contiguous().float()
    score_bias = score_bias.contiguous().float()
    log_time_weight = log_time_weight.contiguous().float()

    # Squeeze w_int from (K_q, n_rep, H, M) to (K_q, H, M) if needed
    if w_int.dim() == 4:
        w_int = w_int.squeeze(1)
    w_int = w_int.contiguous().float()

    # Allocate outputs
    out_re = torch.empty(B, K, H, device=k_val.device, dtype=torch.float32)
    out_im = torch.empty(B, K, H, device=k_val.device, dtype=torch.float32)

    # Single fully fused kernel launch
    # Grid: B * K * ceil(H / BLOCK_H) - autotuner will pick BLOCK_H
    # We use H as the upper bound for grid calculation
    grid = lambda meta: (B * K * ((H + meta["BLOCK_H"] - 1) // meta["BLOCK_H"]),)

    _seqcond_fully_fused_kernel[grid](
        k_val,
        s_raw,
        q_re,
        q_im,
        re_acc,
        im_acc,
        den_acc,
        theta,
        w_int,
        phase_scale,
        score_scale,
        score_bias,
        log_time_weight,
        out_re,
        out_im,
        K,
        H,
        M,
        k_val.stride(0),
        k_val.stride(1),
        k_val.stride(2),
        re_acc.stride(0),
        re_acc.stride(1),
        re_acc.stride(2),
        re_acc.stride(3),
        theta.stride(0),
        theta.stride(1),
        theta.stride(2),
        q_re.stride(0),
        q_re.stride(1),
        q_re.stride(2),
        q_re.stride(3),
        w_int.stride(0),
        w_int.stride(1),
        w_int.stride(2),
        out_re.stride(0),
        out_re.stride(1),
        out_re.stride(2),
    )

    return out_re, out_im

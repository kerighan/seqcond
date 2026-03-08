"""
Functional Model Upscaling via Learned Deconvolution Kernels

Approche: pour chaque layer, on apprend de petits noyaux de déconvolution qui
transforment W_small -> W_large, puis on vérifie fonctionnellement que
downsample(block_large(upsample(x))) ≈ block_small(x) sur des inputs aléatoires.

Le weight sharing des noyaux de déconv = très peu de paramètres à optimiser,
donc des inputs aléatoires suffisent.

Usage:
    python upscale_functional.py \
        --checkpoint checkpoints/seqcond_torch_610k.pt \
        --target-config target_config_xlarge.json \
        --output checkpoints/seqcond_upscaled_functional.pt \
        --steps-per-layer 200 \
        --device cuda
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses
import json
import argparse
from typing import Dict, Tuple, Optional
from tqdm import tqdm
from pathlib import Path

from seqcond.config import ModelConfig
from seqcond.torch.model import SeqCondModel
from seqcond.torch.rope import precompute_freqs


class DeconvKernel(nn.Module):
    """
    Learned deconvolution kernel that derives W_large from W_small.

    For a 2D weight (out, in): applies a transposed convolution per dimension.
    For a 1D weight (d,): applies a 1D transposed convolution.
    For N-D: applies transposed conv on each changed dimension.

    The number of learnable parameters is O(kernel_size^2) per dimension,
    NOT O(out * in) like direct weight optimization.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        kernel_size: int = 4,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_size = kernel_size

        # For each dimension that changes, create a 1D transposed conv kernel
        self.dim_kernels = nn.ParameterDict()
        self.dim_scales = {}

        for dim in range(len(input_shape)):
            src = input_shape[dim]
            tgt = output_shape[dim]
            if src == tgt:
                continue
            self.dim_scales[dim] = tgt / src
            # Transposed conv kernel: (in_channels=1, out_channels=1, kernel_size)
            # Initialized to approximate bilinear interpolation
            kernel = nn.Parameter(torch.zeros(1, 1, kernel_size))
            # Initialize as a smooth interpolation filter
            with torch.no_grad():
                t = torch.linspace(-1, 1, kernel_size)
                kernel.data[0, 0] = (1 - t.abs()).clamp(min=0)
                kernel.data /= kernel.data.sum() + 1e-8
            self.dim_kernels[str(dim)] = kernel

    def forward(self, w_small: torch.Tensor) -> torch.Tensor:
        """
        Apply learned deconvolution to produce W_large from W_small.
        Processes each changed dimension sequentially via transposed conv.
        """
        result = w_small
        ndim = len(self.input_shape)
        current_shape = list(w_small.shape)

        for dim in range(ndim):
            if str(dim) not in self.dim_kernels:
                continue

            tgt_size = self.output_shape[dim]
            kernel = self.dim_kernels[str(dim)]
            src_size = current_shape[dim]

            # Compute stride and padding for ConvTranspose1d to hit target size
            # output_size = (input_size - 1) * stride - 2 * padding + kernel_size
            # We want output_size = tgt_size, input_size = src_size
            # Use stride = ceil(tgt_size / src_size), then crop
            stride = max(1, round(tgt_size / src_size))

            # Move target dim to last, flatten others
            perm = list(range(ndim))
            perm.remove(dim)
            perm.append(dim)
            result = result.permute(*perm)
            flat_shape = result.shape
            result = result.reshape(-1, 1, flat_shape[-1])  # (N, 1, src_size)

            # Apply transposed conv
            result = F.conv_transpose1d(
                result, kernel, stride=stride, padding=self.kernel_size // 2
            )
            # Crop or pad to exact target size
            if result.shape[-1] > tgt_size:
                result = result[..., :tgt_size]
            elif result.shape[-1] < tgt_size:
                result = F.interpolate(
                    result, size=tgt_size, mode="linear", align_corners=True
                )

            # Reshape back
            new_flat_shape = list(flat_shape[:-1]) + [tgt_size]
            result = result.reshape(new_flat_shape)
            inv_perm = [0] * ndim
            for i, p in enumerate(perm):
                inv_perm[p] = i
            result = result.permute(*inv_perm)
            current_shape[dim] = tgt_size

        return result

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def upsample_activation(x: torch.Tensor, target_d: int) -> torch.Tensor:
    """Upsample activation from (B, L, d_small) to (B, L, d_large) via repeat.
    Each feature is duplicated: [a, b, c] -> [a, a, b, b, c, c].
    This is mathematically consistent with W_large = repeat(W_small) / 2."""
    B, L, D = x.shape
    if D == target_d:
        return x
    assert target_d % D == 0, f"target_d={target_d} must be multiple of D={D}"
    return x.repeat_interleave(target_d // D, dim=-1)


def downsample_activation(x: torch.Tensor, target_d: int) -> torch.Tensor:
    """Downsample activation from (B, L, d_large) to (B, L, d_small) via averaging.
    Inverse of repeat_interleave: average groups of consecutive features."""
    B, L, D = x.shape
    if D == target_d:
        return x
    assert D % target_d == 0, f"D={D} must be multiple of target_d={target_d}"
    factor = D // target_d
    return x.reshape(B, L, target_d, factor).mean(dim=-1)


def head_aware_init_weights(
    source_weights: Dict[str, torch.Tensor],
    target_shapes: Dict[str, Tuple[int, ...]],
    block_type: str,
    source_config: ModelConfig,
    target_config: ModelConfig,
) -> Dict[str, torch.Tensor]:
    """
    Initialize target weights from source using head-aware duplication.

    For structured per-head params (SeqCond/Transformer), each head's weights
    are duplicated rather than interpolated. For d_model input dimensions,
    columns are duplicated and divided by the factor to preserve output magnitude.

    This gives PERFECT reconstruction: loss = 0 at initialization.
    """
    K_src = source_config.seqcond_heads
    K_tgt = target_config.seqcond_heads
    head_factor = K_tgt // K_src  # 2 for x2 scaling

    d_src = source_config.d_model
    d_tgt = target_config.d_model
    d_factor = d_tgt // d_src  # 2 for x2 scaling

    # SeqCond internal dims (source)
    M = source_config.num_thetas
    expand = source_config.expand_factor
    d_inner = int(d_src * expand)
    H = max(1, d_inner // (K_src * M))
    K_q_src = source_config.num_query_heads
    dim_memory = K_src * H
    dim_mem_total = dim_memory + K_src
    dim_query_head = H * M * 2
    dim_expand = H * source_config.out_expand_factor

    # Transformer dims (source)
    num_heads_src = source_config.num_heads
    num_heads_tgt = target_config.num_heads
    head_dim_src = d_src // num_heads_src
    num_kv_src = source_config.num_kv_heads or num_heads_src
    num_kv_tgt = target_config.num_kv_heads or num_heads_tgt
    kv_factor = num_kv_tgt // num_kv_src
    heads_factor_t = num_heads_tgt // num_heads_src
    d_ff_src = source_config.d_ff

    result = {}
    for name, src_w in source_weights.items():
        tgt_shape = target_shapes[name]
        if tuple(src_w.shape) == tgt_shape:
            result[name] = src_w.clone()
            continue

        if block_type == "seqcond":
            result[name] = _init_seqcond_param(
                name,
                src_w,
                tgt_shape,
                K_src,
                head_factor,
                d_factor,
                H,
                M,
                K_q_src,
                dim_memory,
                dim_mem_total,
                dim_query_head,
                dim_expand,
            )
        else:  # transformer
            result[name] = _init_transformer_param(
                name,
                src_w,
                tgt_shape,
                num_heads_src,
                heads_factor_t,
                d_factor,
                num_kv_src,
                kv_factor,
                head_dim_src,
                d_ff_src,
            )

    return result


def _init_seqcond_param(
    name,
    src_w,
    tgt_shape,
    K,
    head_factor,
    d_factor,
    H,
    M,
    K_q,
    dim_memory,
    dim_mem_total,
    dim_query_head,
    dim_expand,
):
    """Head-aware init for a single SeqCond parameter."""
    if name == "attn.in_proj.weight":
        # Output: [K*H memory | K scores | K_q*query_head_dim queries]
        mem = src_w[:dim_memory].reshape(K, H, -1)
        mem = mem.repeat_interleave(head_factor, dim=0).reshape(K * head_factor * H, -1)
        score = src_w[dim_memory:dim_mem_total].repeat_interleave(head_factor, dim=0)
        query = src_w[dim_mem_total:].reshape(K_q, dim_query_head, -1)
        query = query.repeat_interleave(head_factor, dim=0).reshape(
            K_q * head_factor * dim_query_head, -1
        )
        out = torch.cat([mem, score, query], dim=0)
        return out.repeat_interleave(d_factor, dim=1) / d_factor

    if name == "attn.conv_weight":
        # (dim_conv_total, 1, kernel): same head structure as in_proj output
        mem = src_w[:dim_memory].reshape(K, H, 1, -1)
        mem = mem.repeat_interleave(head_factor, dim=0).reshape(
            K * head_factor * H, 1, -1
        )
        score = src_w[dim_memory:dim_mem_total].repeat_interleave(head_factor, dim=0)
        query = src_w[dim_mem_total:].reshape(K_q, dim_query_head, 1, -1)
        query = query.repeat_interleave(head_factor, dim=0).reshape(
            K_q * head_factor * dim_query_head, 1, -1
        )
        return torch.cat([mem, score, query], dim=0)

    if name == "attn.gate_proj.weight":
        # (K * 2 * H, d_model)
        heads = src_w.reshape(K, 2 * H, -1)
        out = heads.repeat_interleave(head_factor, dim=0).reshape(
            K * head_factor * 2 * H, -1
        )
        return out.repeat_interleave(d_factor, dim=1) / d_factor

    if name == "attn.out_proj.weight":
        # (d_model, K * dim_expand)
        inp_heads = src_w.reshape(-1, K, dim_expand)
        inp_up = inp_heads.repeat_interleave(head_factor, dim=1).reshape(
            -1, K * head_factor * dim_expand
        )
        return inp_up.repeat_interleave(d_factor, dim=0) / d_factor

    if name in (
        "attn.decay_slopes",
        "attn.score_scale",
        "attn.score_bias",
        "attn.phase_scale",
    ):
        return src_w.repeat_interleave(head_factor)

    if name == "attn.gated_norm.weight":
        return src_w.reshape(K, 2 * H).repeat_interleave(head_factor, dim=0).reshape(-1)

    if name == "attn.W_readout":
        return src_w.repeat_interleave(head_factor, dim=0)

    if name == "attn.theta_d_raw":
        return src_w.repeat_interleave(head_factor, dim=2)

    if name == "attn.w_int_raw":
        return src_w.repeat_interleave(head_factor, dim=2)

    if name == "norm.scale":
        return src_w.repeat_interleave(d_factor)

    # Fallback: repeat_interleave on each changed dim
    result = src_w
    for dim in range(len(src_w.shape)):
        if src_w.shape[dim] != tgt_shape[dim]:
            factor = tgt_shape[dim] // src_w.shape[dim]
            result = result.repeat_interleave(factor, dim=dim)
            if dim == len(src_w.shape) - 1:  # input dim: divide
                result = result / factor
    return result


def _init_transformer_param(
    name,
    src_w,
    tgt_shape,
    num_heads,
    heads_factor,
    d_factor,
    num_kv,
    kv_factor,
    head_dim,
    d_ff,
):
    """Head-aware init for a single Transformer parameter."""
    if name == "attn.q_proj.weight":
        # (num_heads * head_dim, d_model) -> duplicate heads on dim 0, repeat input dim 1
        heads = src_w.reshape(num_heads, head_dim, -1)  # (num_heads, head_dim, d_model)
        out = heads.repeat_interleave(
            heads_factor, dim=0
        )  # (2*num_heads, head_dim, d_model)
        out = out.reshape(
            num_heads * heads_factor * head_dim, -1
        )  # (2*num_heads*head_dim, d_model)
        return out.repeat_interleave(d_factor, dim=1) / d_factor

    if name == "attn.out_proj.weight":
        # (d_model, num_heads * head_dim) -> repeat output dim 0, duplicate heads on dim 1
        inp_heads = src_w.reshape(
            -1, num_heads, head_dim
        )  # (d_model, num_heads, head_dim)
        out = inp_heads.repeat_interleave(
            heads_factor, dim=1
        )  # (d_model, 2*num_heads, head_dim)
        out = out.reshape(
            -1, num_heads * heads_factor * head_dim
        )  # (d_model, 2*num_heads*head_dim)
        return out.repeat_interleave(d_factor, dim=0) / d_factor

    if name in ("attn.k_proj.weight", "attn.v_proj.weight"):
        # (num_kv * head_dim, d_model) -> duplicate kv heads on dim 0
        heads = src_w.reshape(num_kv, head_dim, -1)
        out = heads.repeat_interleave(kv_factor, dim=0).reshape(
            num_kv * kv_factor * head_dim, -1
        )
        return out.repeat_interleave(d_factor, dim=1) / d_factor

    if name == "ff_in.weight":
        # (2 * d_ff, d_model) -> repeat_interleave both dims
        out = src_w.repeat_interleave(d_factor, dim=0)
        return out.repeat_interleave(d_factor, dim=1) / d_factor

    if name == "ff_in.bias":
        return src_w.repeat_interleave(d_factor)

    if name == "ff_out.weight":
        # (d_model, d_ff)
        out = src_w.repeat_interleave(d_factor, dim=0) / d_factor
        return out.repeat_interleave(d_factor, dim=1)

    if name == "ff_out.bias":
        return src_w.repeat_interleave(d_factor)

    if name in ("norm1.scale", "norm2.scale"):
        return src_w.repeat_interleave(d_factor)

    # Fallback
    result = src_w
    for dim in range(len(src_w.shape)):
        if src_w.shape[dim] != tgt_shape[dim]:
            factor = tgt_shape[dim] // src_w.shape[dim]
            result = result.repeat_interleave(factor, dim=dim)
    return result


def build_block_from_state(
    block_type: str,
    state_dict: Dict[str, torch.Tensor],
    config: ModelConfig,
    prefix: str,
) -> nn.Module:
    """Instantiate a single block and load its weights."""
    if block_type == "transformer":
        from seqcond.torch.rope import TransformerDecoderBlock

        block = TransformerDecoderBlock(
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            num_kv_heads=config.num_kv_heads,
            qk_norm=config.qk_norm,
        )
    else:
        from seqcond.torch.seqcond import SeqCondBlock

        block = SeqCondBlock(
            d_model=config.d_model,
            num_heads=config.seqcond_heads,
            num_query_heads=config.num_query_heads,
            num_thetas=config.num_thetas,
            conv_kernel_size=config.conv_kernel_size,
            expand_factor=config.expand_factor,
            out_expand_factor=config.out_expand_factor,
            maxlen=config.maxlen,
        )

    # Load weights with prefix stripping
    block_sd = {}
    for k, v in state_dict.items():
        if k.startswith(prefix + "."):
            local_key = k[len(prefix) + 1 :]
            block_sd[local_key] = v

    block.load_state_dict(block_sd, strict=True)
    return block


def run_source_block(block, block_type, x, cos=None, sin=None):
    """Forward pass through a source block."""
    with torch.no_grad():
        if block_type == "transformer":
            return block(x, cos, sin)
        else:
            return block(x)


def functional_forward(target_block, block_type, params_dict, x, cos=None, sin=None):
    """
    Forward pass using functional_call to preserve gradient flow.
    params_dict maps param names to tensors (possibly with grad).
    """
    if block_type == "transformer":
        return torch.func.functional_call(target_block, params_dict, (x, cos, sin))
    else:
        return torch.func.functional_call(target_block, params_dict, (x,))


def learn_layer_functional(
    source_block: nn.Module,
    target_block: nn.Module,
    block_type: str,
    source_weights: Dict[str, torch.Tensor],
    target_shapes: Dict[str, Tuple[int, ...]],
    source_config: ModelConfig,
    target_config: ModelConfig,
    num_steps: int = 200,
    lr: float = 1e-2,
    batch_size: int = 4,
    seq_len: int = 128,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Learn deconv kernels for one layer via functional distillation.

    Returns the final upscaled weights for this layer.
    """
    source_block = source_block.to(device).eval()
    target_block = target_block.to(device).train()

    # Freeze source block
    for p in source_block.parameters():
        p.requires_grad_(False)

    # Create deconv kernels for weights that change shape
    deconv_kernels = nn.ModuleDict()
    for name, src_w in source_weights.items():
        tgt_shape = target_shapes[name]
        if tuple(src_w.shape) != tgt_shape:
            deconv_kernels[name.replace(".", "_")] = DeconvKernel(
                tuple(src_w.shape), tgt_shape, kernel_size=4
            )

    # Map clean names to module dict names
    name_to_key = {
        name: name.replace(".", "_")
        for name in source_weights
        if tuple(source_weights[name].shape) != target_shapes[name]
    }

    deconv_kernels = deconv_kernels.to(device)
    source_weights_dev = {k: v.to(device) for k, v in source_weights.items()}

    total_deconv_params = sum(p.numel() for p in deconv_kernels.parameters())
    total_weight_params = sum(
        v.numel()
        for k, v in source_weights.items()
        if tuple(v.shape) != target_shapes[k]
    )
    print(
        f"    Deconv params: {total_deconv_params:,} "
        f"(vs {total_weight_params:,} weight params, "
        f"compression: {total_weight_params / max(total_deconv_params, 1):.0f}x)"
    )

    optimizer = torch.optim.Adam(deconv_kernels.parameters(), lr=lr)

    # Precompute RoPE if needed
    cos_src, sin_src, cos_tgt, sin_tgt = None, None, None, None
    if block_type == "transformer":
        head_dim_src = source_config.d_model // source_config.num_heads
        cos_s, sin_s = precompute_freqs(source_config.maxlen, head_dim_src)
        cos_src = (
            cos_s[:seq_len]
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(batch_size, seq_len, source_config.num_heads, -1)
            .to(device)
        )
        sin_src = (
            sin_s[:seq_len]
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(batch_size, seq_len, source_config.num_heads, -1)
            .to(device)
        )

        head_dim_tgt = target_config.d_model // target_config.num_heads
        cos_t, sin_t = precompute_freqs(target_config.maxlen, head_dim_tgt)
        cos_tgt = (
            cos_t[:seq_len]
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(batch_size, seq_len, target_config.num_heads, -1)
            .to(device)
        )
        sin_tgt = (
            sin_t[:seq_len]
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(batch_size, seq_len, target_config.num_heads, -1)
            .to(device)
        )

    # Also freeze target block's own parameters (we inject weights via functional_call)
    for p in target_block.parameters():
        p.requires_grad_(False)

    pbar = tqdm(range(num_steps), desc="    Functional distillation")
    for step in pbar:
        optimizer.zero_grad()

        # Generate random input
        x_small = (
            torch.randn(batch_size, seq_len, source_config.d_model, device=device) * 0.1
        )

        # Source forward (frozen)
        y_small = run_source_block(source_block, block_type, x_small, cos_src, sin_src)

        # Upsample input for target block
        x_large = upsample_activation(x_small, target_config.d_model)

        # Build deconv-derived weights dict (gradients flow through deconv kernels)
        derived_params = {}
        for name, src_w in source_weights_dev.items():
            if name in name_to_key:
                derived_params[name] = deconv_kernels[name_to_key[name]](src_w)
            else:
                derived_params[name] = src_w

        # Forward through target block using functional_call (preserves grad graph)
        y_large = functional_forward(
            target_block,
            block_type,
            derived_params,
            x_large,
            cos_tgt,
            sin_tgt,
        )

        # Downsample target output and compare
        y_large_down = downsample_activation(y_large, source_config.d_model)

        loss = F.mse_loss(y_large_down, y_small.detach())

        loss.backward()
        optimizer.step()

        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    # Extract final upscaled weights
    final_weights = {}
    with torch.no_grad():
        for name, src_w in source_weights_dev.items():
            if name in name_to_key:
                final_weights[name] = deconv_kernels[name_to_key[name]](src_w).cpu()
            else:
                final_weights[name] = src_w.cpu()

    # Cleanup GPU
    del source_block, target_block, deconv_kernels, source_weights_dev
    torch.cuda.empty_cache()

    return final_weights


def upscale_model_functional(
    checkpoint_path: str,
    output_path: str,
    target_config: ModelConfig,
    num_steps_per_layer: int = 200,
    batch_size: int = 4,
    seq_len: int = 128,
    device: str = "cuda",
):
    """
    Upscale a model using functional distillation with learned deconv kernels.
    Processes one layer at a time to fit on small GPUs.
    """
    from upscale_model import compute_target_shapes

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}
    if "config" in checkpoint:
        config_dict = {
            k: v for k, v in checkpoint["config"].items() if k in valid_fields
        }
        source_config = ModelConfig(**config_dict)
    else:
        source_config = ModelConfig.large()

    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

    print(
        f"\nSource: d_model={source_config.d_model}, "
        f"layers={source_config.num_layers}, "
        f"seqcond_heads={source_config.seqcond_heads}"
    )
    print(
        f"Target: d_model={target_config.d_model}, "
        f"layers={target_config.num_layers}, "
        f"seqcond_heads={target_config.seqcond_heads}"
    )

    # Compute target shapes
    target_shapes = compute_target_shapes(state_dict, target_config)

    # Determine block types
    block_types = []
    for i in range(source_config.num_layers):
        if (i + 1) % (source_config.seqcond_ratio + 1) == 0:
            block_types.append("transformer")
        else:
            block_types.append("seqcond")

    print(
        f"Block types: {''.join('T' if t == 'transformer' else 'S' for t in block_types)}"
    )

    n_changed = sum(
        1 for n, p in state_dict.items() if tuple(p.shape) != target_shapes[n]
    )
    print(f"Parameters to upscale: {n_changed}/{len(state_dict)}")

    upscaled_state_dict = {}

    # Process embedding (simple interpolation, no functional distillation needed)
    print("\n[Embedding] Interpolating...")
    from upscale_model import initialize_upscaled_weights

    for name in ["embedding.weight"]:
        if name in state_dict:
            src_w = state_dict[name]
            tgt_shape = target_shapes[name]
            if tuple(src_w.shape) != tgt_shape:
                upscaled_state_dict[name] = initialize_upscaled_weights(
                    src_w, tgt_shape
                )
                print(f"  {name}: {tuple(src_w.shape)} -> {tgt_shape}")
            else:
                upscaled_state_dict[name] = src_w

    # Process final_norm (simple interpolation)
    for name in list(state_dict.keys()):
        if name.startswith("final_norm"):
            src_w = state_dict[name]
            tgt_shape = target_shapes[name]
            if tuple(src_w.shape) != tgt_shape:
                upscaled_state_dict[name] = initialize_upscaled_weights(
                    src_w, tgt_shape
                )
                print(f"  {name}: {tuple(src_w.shape)} -> {tgt_shape}")
            else:
                upscaled_state_dict[name] = src_w

    # Process blocks layer by layer
    for layer_idx in range(source_config.num_layers):
        prefix = f"blocks.{layer_idx}"
        btype = block_types[layer_idx]

        # Collect source weights for this block
        source_weights = {}
        layer_target_shapes = {}
        for k, v in state_dict.items():
            if k.startswith(prefix + "."):
                local_key = k[len(prefix) + 1 :]
                source_weights[local_key] = v
                layer_target_shapes[local_key] = target_shapes[k]

        changes = [
            k
            for k, v in source_weights.items()
            if tuple(v.shape) != layer_target_shapes[k]
        ]
        btype_label = "Transformer" if btype == "transformer" else "SeqCond"
        print(
            f"\n[Block {layer_idx}] {btype_label} ({len(source_weights)} params, {len(changes)} to upscale)"
        )

        if not changes:
            # No changes, copy as-is
            for local_key, w in source_weights.items():
                upscaled_state_dict[f"{prefix}.{local_key}"] = w
            continue

        for k in changes:
            print(
                f"  {k}: {tuple(source_weights[k].shape)} -> {layer_target_shapes[k]}"
            )

        # Build source and target blocks
        source_block = build_block_from_state(btype, state_dict, source_config, prefix)
        target_block = build_block_from_state(
            btype,
            # Initialize target block with random weights (will be overwritten)
            {
                f"{prefix}.{k}": torch.randn(layer_target_shapes[k])
                for k in source_weights
            },
            target_config,
            prefix,
        )

        # Learn deconv kernels via functional distillation
        upscaled_weights = learn_layer_functional(
            source_block=source_block,
            target_block=target_block,
            block_type=btype,
            source_weights=source_weights,
            target_shapes=layer_target_shapes,
            source_config=source_config,
            target_config=target_config,
            num_steps=num_steps_per_layer,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )

        # Store with full key names
        for local_key, w in upscaled_weights.items():
            upscaled_state_dict[f"{prefix}.{local_key}"] = w

    # lm_head is tied with embedding
    if "lm_head.weight" in state_dict:
        upscaled_state_dict["lm_head.weight"] = upscaled_state_dict["embedding.weight"]

    # Save
    print(f"\nSaving to {output_path}")
    torch.save(
        {
            "model": upscaled_state_dict,
            "config": target_config.to_dict(),
            "original_checkpoint": checkpoint_path,
            "original_config": source_config.to_dict(),
            "method": "functional_deconv",
        },
        output_path,
    )

    original_params = sum(p.numel() for p in state_dict.values())
    upscaled_params = sum(p.numel() for p in upscaled_state_dict.values())
    print(f"\nParameter count:")
    print(f"  Original: {original_params:,} ({original_params / 1e6:.1f}M)")
    print(f"  Upscaled: {upscaled_params:,} ({upscaled_params / 1e6:.1f}M)")
    print(f"  Ratio: {upscaled_params / original_params:.2f}x")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upscale model via functional distillation with deconv kernels"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/seqcond_torch_610k.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/seqcond_upscaled_functional.pt",
    )
    parser.add_argument(
        "--target-config",
        type=str,
        required=True,
        help="Path to target config JSON",
    )
    parser.add_argument("--steps-per-layer", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    with open(args.target_config) as f:
        target_config = ModelConfig(**json.load(f))

    upscale_model_functional(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        target_config=target_config,
        num_steps_per_layer=args.steps_per_layer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=args.device,
    )

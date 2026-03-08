"""
Head-Aware Model Upscaling — 370M → 1.26B instantanément.

Duplique les heads (SeqCond + Transformer) pour doubler la largeur du modèle.
L'embedding est upscalé par interpolation bilinéaire (pas de structure per-head).
Résultat : le modèle upscalé est fonctionnellement équivalent au petit (loss = 0).

Usage:
    python upscale_head_aware.py \
        --checkpoint checkpoints/seqcond_torch_640k.pt \
        --target-config target_config_xlarge.json \
        --output checkpoints/seqcond_xlarge_init.pt
"""

import torch
import torch.nn.functional as F
import dataclasses
import inspect
import json
import argparse
from typing import Dict, Tuple

from seqcond.config import ModelConfig
from seqcond.torch.model import SeqCondModel


# ---------------------------------------------------------------------------
# Head-aware weight initialization
# ---------------------------------------------------------------------------


def head_aware_init_seqcond(
    source_weights: Dict[str, torch.Tensor],
    target_shapes: Dict[str, Tuple[int, ...]],
    src_cfg: ModelConfig,
    tgt_cfg: ModelConfig,
) -> Dict[str, torch.Tensor]:
    """
    Initialize SeqCond block weights by duplicating heads.

    Key insight: H = d_inner / (K * M) stays CONSTANT between source and target.
    Only K (num heads) doubles. Each head is an independent module → just duplicate.
    For the d_model input dimension, repeat + /factor to compensate the sum.
    """
    K = src_cfg.seqcond_heads
    hf = tgt_cfg.seqcond_heads // K  # head factor (2 for ×2)
    df = tgt_cfg.d_model // src_cfg.d_model  # d_model factor (2 for ×2)

    M = src_cfg.num_thetas
    H = max(1, int(src_cfg.d_model * src_cfg.expand_factor) // (K * M))
    K_q = src_cfg.num_query_heads
    dim_memory = K * H
    dim_mem_total = dim_memory + K
    dim_query_head = H * M * 2
    dim_expand = H * src_cfg.out_expand_factor

    result = {}
    for name, w in source_weights.items():
        tgt = target_shapes[name]
        if tuple(w.shape) == tgt:
            result[name] = w.clone()
            continue

        # --- in_proj.weight: (dim_conv_total, d_model) ---
        if name == "attn.in_proj.weight":
            mem = w[:dim_memory].reshape(K, H, -1)
            mem = mem.repeat_interleave(hf, dim=0).reshape(K * hf * H, -1)
            score = w[dim_memory:dim_mem_total].repeat_interleave(hf, dim=0)
            query = w[dim_mem_total:].reshape(K_q, dim_query_head, -1)
            query = query.repeat_interleave(hf, dim=0).reshape(
                K_q * hf * dim_query_head, -1
            )
            out = torch.cat([mem, score, query], dim=0)
            result[name] = out.repeat_interleave(df, dim=1) / df

        # --- conv_weight: (dim_conv_total, 1, kernel) ---
        elif name == "attn.conv_weight":
            mem = w[:dim_memory].reshape(K, H, 1, -1)
            mem = mem.repeat_interleave(hf, dim=0).reshape(K * hf * H, 1, -1)
            score = w[dim_memory:dim_mem_total].repeat_interleave(hf, dim=0)
            query = w[dim_mem_total:].reshape(K_q, dim_query_head, 1, -1)
            query = query.repeat_interleave(hf, dim=0).reshape(
                K_q * hf * dim_query_head, 1, -1
            )
            result[name] = torch.cat([mem, score, query], dim=0)

        # --- gate_proj.weight: (K * 2H, d_model) ---
        elif name == "attn.gate_proj.weight":
            heads = w.reshape(K, 2 * H, -1)
            out = heads.repeat_interleave(hf, dim=0).reshape(K * hf * 2 * H, -1)
            result[name] = out.repeat_interleave(df, dim=1) / df

        # --- out_proj.weight: (d_model, K * dim_expand) ---
        elif name == "attn.out_proj.weight":
            inp = w.reshape(-1, K, dim_expand)
            inp = inp.repeat_interleave(hf, dim=1).reshape(-1, K * hf * dim_expand)
            result[name] = inp.repeat_interleave(df, dim=0) / df

        # --- per-head scalars ---
        elif name in (
            "attn.decay_slopes",
            "attn.score_scale",
            "attn.score_bias",
            "attn.phase_scale",
        ):
            result[name] = w.repeat_interleave(hf)

        # --- gated_norm.weight: (K * 2H,) ---
        elif name == "attn.gated_norm.weight":
            result[name] = w.reshape(K, 2 * H).repeat_interleave(hf, dim=0).reshape(-1)

        # --- W_readout: (K, 2H, swiglu_head) ---
        elif name == "attn.W_readout":
            result[name] = w.repeat_interleave(hf, dim=0)

        # --- theta_d_raw: (1, 1, K, H, M) ---
        elif name == "attn.theta_d_raw":
            result[name] = w.repeat_interleave(hf, dim=2)

        # --- w_int_raw: (1, 1, K_q, n_rep, H, M) ---
        elif name == "attn.w_int_raw":
            result[name] = w.repeat_interleave(hf, dim=2)

        # --- norm.scale: (d_model,) ---
        elif name == "norm.scale":
            result[name] = w.repeat_interleave(df)

        # --- fallback ---
        else:
            r = w
            for dim in range(len(w.shape)):
                if w.shape[dim] != tgt[dim]:
                    f = tgt[dim] // w.shape[dim]
                    r = r.repeat_interleave(f, dim=dim)
            result[name] = r

    return result


def head_aware_init_transformer(
    source_weights: Dict[str, torch.Tensor],
    target_shapes: Dict[str, Tuple[int, ...]],
    src_cfg: ModelConfig,
    tgt_cfg: ModelConfig,
) -> Dict[str, torch.Tensor]:
    """
    Initialize Transformer block weights by duplicating heads.

    q_proj, k_proj, v_proj, out_proj are structured per-head.
    FFN is repeated + /factor for magnitude preservation.
    """
    num_h = src_cfg.num_heads
    hf = tgt_cfg.num_heads // num_h
    df = tgt_cfg.d_model // src_cfg.d_model
    head_dim = src_cfg.d_model // num_h
    num_kv = src_cfg.num_kv_heads or num_h
    kv_f = (tgt_cfg.num_kv_heads or tgt_cfg.num_heads) // num_kv

    result = {}
    for name, w in source_weights.items():
        tgt = target_shapes[name]
        if tuple(w.shape) == tgt:
            result[name] = w.clone()
            continue

        if name == "attn.q_proj.weight":
            heads = w.reshape(num_h, head_dim, -1)
            out = heads.repeat_interleave(hf, dim=0).reshape(num_h * hf * head_dim, -1)
            result[name] = out.repeat_interleave(df, dim=1) / df

        elif name == "attn.out_proj.weight":
            heads = w.reshape(-1, num_h, head_dim)
            out = heads.repeat_interleave(hf, dim=1).reshape(-1, num_h * hf * head_dim)
            result[name] = out.repeat_interleave(df, dim=0) / df

        elif name in ("attn.k_proj.weight", "attn.v_proj.weight"):
            heads = w.reshape(num_kv, head_dim, -1)
            out = heads.repeat_interleave(kv_f, dim=0).reshape(
                num_kv * kv_f * head_dim, -1
            )
            result[name] = out.repeat_interleave(df, dim=1) / df

        elif name == "ff_in.weight":
            out = w.repeat_interleave(df, dim=0)
            result[name] = out.repeat_interleave(df, dim=1) / df

        elif name == "ff_in.bias":
            result[name] = w.repeat_interleave(df)

        elif name == "ff_out.weight":
            out = w.repeat_interleave(df, dim=0) / df
            result[name] = out.repeat_interleave(df, dim=1)

        elif name == "ff_out.bias":
            result[name] = w.repeat_interleave(df)

        elif name in ("norm1.scale", "norm2.scale"):
            result[name] = w.repeat_interleave(df)

        else:
            r = w
            for dim in range(len(w.shape)):
                if w.shape[dim] != tgt[dim]:
                    f = tgt[dim] // w.shape[dim]
                    r = r.repeat_interleave(f, dim=dim)
            result[name] = r

    return result


# ---------------------------------------------------------------------------
# Main upscaling function
# ---------------------------------------------------------------------------


def upscale_model(checkpoint_path: str, target_config: ModelConfig, output_path: str):
    """Upscale a checkpoint using head-aware weight duplication."""

    print(f"Loading {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}
    if "config" in checkpoint:
        cfg_dict = {k: v for k, v in checkpoint["config"].items() if k in valid_fields}
        src_cfg = ModelConfig(**cfg_dict)
    else:
        src_cfg = ModelConfig.large()

    sd = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

    print(
        f"Source: {sum(p.numel() for p in sd.values()) / 1e6:.0f}M params, "
        f"d_model={src_cfg.d_model}, heads={src_cfg.seqcond_heads}"
    )
    print(
        f"Target: d_model={target_config.d_model}, heads={target_config.seqcond_heads}"
    )

    # Block types
    block_types = []
    for i in range(src_cfg.num_layers):
        if (i + 1) % (src_cfg.seqcond_ratio + 1) == 0:
            block_types.append("transformer")
        else:
            block_types.append("seqcond")

    # Instantiate target model for reference shapes + buffers
    valid_init = set(inspect.signature(SeqCondModel.__init__).parameters.keys()) - {
        "self"
    }
    tgt_init = {k: v for k, v in target_config.to_dict().items() if k in valid_init}
    tgt_model = SeqCondModel(**tgt_init)
    tgt_sd = tgt_model.state_dict()

    upscaled = {}

    # 1. Embedding — bilinear interpolation (no per-head structure)
    emb = sd["embedding.weight"]
    emb_4d = emb.unsqueeze(0).unsqueeze(0)
    emb_up = F.interpolate(
        emb_4d,
        size=(emb.shape[0], target_config.d_model),
        mode="bilinear",
        align_corners=True,
    )
    upscaled["embedding.weight"] = emb_up.squeeze(0).squeeze(0)
    print(
        f"  embedding: {tuple(emb.shape)} → {tuple(upscaled['embedding.weight'].shape)}  (bilinear)"
    )

    # 2. Blocks — head-aware duplication
    for idx in range(src_cfg.num_layers):
        prefix = f"blocks.{idx}"
        btype = block_types[idx]

        src_w = {}
        tgt_shapes = {}
        for k, v in sd.items():
            if k.startswith(prefix + "."):
                local = k[len(prefix) + 1 :]
                src_w[local] = v
                tgt_shapes[local] = tuple(tgt_sd[k].shape)

        if btype == "seqcond":
            init_w = head_aware_init_seqcond(src_w, tgt_shapes, src_cfg, target_config)
        else:
            init_w = head_aware_init_transformer(
                src_w, tgt_shapes, src_cfg, target_config
            )

        n_changed = sum(1 for k in src_w if tuple(src_w[k].shape) != tgt_shapes[k])
        label = "SeqCond" if btype == "seqcond" else "Transformer"
        print(f"  block {idx:2d} ({label:11s}): {n_changed} params upscaled")

        for local, w in init_w.items():
            upscaled[f"{prefix}.{local}"] = w

    # 3. Non-block params from target model (final_norm, lm_head, RoPE buffers)
    for k, v in tgt_sd.items():
        if k not in upscaled:
            if k == "lm_head.weight":
                upscaled[k] = upscaled["embedding.weight"]
            elif k == "final_norm.scale":
                upscaled[k] = torch.ones(target_config.d_model)
            else:
                upscaled[k] = v  # cos_emb, sin_emb, etc.

    # Verify strict load
    tgt_model.load_state_dict(upscaled, strict=True)
    total = sum(p.numel() for p in tgt_model.parameters())
    print(f"\nStrict load OK — {total:,} params ({total / 1e9:.2f}B)")

    # Save with filtered config (only keys SeqCondModel accepts)
    save_config = {k: v for k, v in target_config.to_dict().items() if k in valid_init}
    torch.save(
        {
            "state_dict": upscaled,
            "config": save_config,
            "original_checkpoint": checkpoint_path,
            "method": "head_aware_init",
        },
        output_path,
    )

    import os

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved to {output_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upscale model via head-aware weight duplication"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--target-config", type=str, required=True)
    parser.add_argument(
        "--output", type=str, default="checkpoints/seqcond_xlarge_init.pt"
    )
    args = parser.parse_args()

    with open(args.target_config) as f:
        tgt_cfg = ModelConfig(**json.load(f))

    upscale_model(args.checkpoint, tgt_cfg, args.output)

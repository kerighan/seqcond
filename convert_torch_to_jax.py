"""
Convert PyTorch checkpoint (.pt) to JAX checkpoint (.pkl).
Inverse of convert_jax_to_torch.py.

Usage:
    python convert_torch_to_jax.py \
        --input checkpoints/seqcond_xlarge_init.pt \
        --output checkpoints/seqcond_xlarge_init.pkl
"""

import argparse
import pickle
import torch
import numpy as np


def convert_torch_to_jax(torch_path: str, jax_path: str):
    print(f"Loading PyTorch checkpoint from {torch_path}...")
    checkpoint = torch.load(torch_path, map_location="cpu")

    config = checkpoint["config"]
    sd = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

    seqcond_ratio = config.get("seqcond_ratio", 3)
    num_layers = config["num_layers"]
    model_type = config.get("model_type", "seqcond")

    print(f"  d_model={config['d_model']}, num_layers={num_layers}, "
          f"seqcond_ratio={seqcond_ratio}")

    jax_params = {}

    # Embedding: torch embedding.weight -> JAX token_embedding.embedding
    jax_params["token_embedding"] = {
        "embedding": sd["embedding.weight"].numpy()
    }
    print(f"  embedding: {tuple(sd['embedding.weight'].shape)}")

    # Blocks
    transformer_idx = 0
    seqcond_idx = 0

    for i in range(num_layers):
        tp = f"blocks.{i}."

        if model_type == "transformer" or (i + 1) % (seqcond_ratio + 1) == 0:
            # Transformer block
            jax_key = f"transformer_block_{transformer_idx}"
            jax_block = {}

            # Norms
            jax_block["norm1"] = {"scale": sd[tp + "norm1.scale"].numpy()}
            jax_block["norm2"] = {"scale": sd[tp + "norm2.scale"].numpy()}

            # Attention — kernels are transposed (.t()) between torch and JAX
            attn = {}
            attn["q_proj"] = {"kernel": sd[tp + "attn.q_proj.weight"].t().numpy()}
            attn["k_proj"] = {"kernel": sd[tp + "attn.k_proj.weight"].t().numpy()}
            attn["v_proj"] = {"kernel": sd[tp + "attn.v_proj.weight"].t().numpy()}
            attn["out_proj"] = {"kernel": sd[tp + "attn.out_proj.weight"].t().numpy()}
            jax_block["attn"] = attn

            # FFN
            jax_block["ff_in"] = {
                "kernel": sd[tp + "ff_in.weight"].t().numpy(),
                "bias": sd[tp + "ff_in.bias"].numpy(),
            }
            jax_block["ff_out"] = {
                "kernel": sd[tp + "ff_out.weight"].t().numpy(),
                "bias": sd[tp + "ff_out.bias"].numpy(),
            }

            jax_params[jax_key] = jax_block
            transformer_idx += 1
            print(f"  block {i:2d} -> {jax_key}")

        else:
            # SeqCond block
            jax_key = f"seqcond_block_{seqcond_idx}"
            jax_block = {}
            attn = {}

            # Block norm
            jax_block["RMSNorm_0"] = {"scale": sd[tp + "norm.scale"].numpy()}

            # in_proj — transposed
            attn["in_proj"] = {"kernel": sd[tp + "attn.in_proj.weight"].t().numpy()}

            # conv_weight: torch (C, 1, K) -> JAX (K, 1, C)
            conv_w = sd[tp + "attn.conv_weight"].numpy()  # (C, 1, K)
            attn["conv"] = {"kernel": conv_w.transpose(2, 1, 0)}

            # Theta
            if tp + "attn.theta_raw" in sd:
                attn["theta_raw"] = sd[tp + "attn.theta_raw"].numpy()
            if tp + "attn.theta_d_raw" in sd:
                attn["theta_d_raw"] = sd[tp + "attn.theta_d_raw"].numpy()

            # w_int_raw
            if tp + "attn.w_int_raw" in sd:
                attn["w_int_raw"] = sd[tp + "attn.w_int_raw"].numpy()

            # Decay/anchor slopes
            if tp + "attn.decay_slopes" in sd:
                attn["decay_slopes"] = sd[tp + "attn.decay_slopes"].numpy()
            if tp + "attn.anchor_slopes" in sd:
                attn["anchor_slopes"] = sd[tp + "attn.anchor_slopes"].numpy()

            # Score
            attn["score_scale"] = sd[tp + "attn.score_scale"].numpy()
            attn["score_bias"] = sd[tp + "attn.score_bias"].numpy()

            # Phase
            attn["phase_scale"] = sd[tp + "attn.phase_scale"].numpy()

            # Readout
            attn["W_readout"] = sd[tp + "attn.W_readout"].numpy()

            # GatedRMSNorm
            if tp + "attn.gate_proj.weight" in sd:
                attn["gate_proj"] = {"kernel": sd[tp + "attn.gate_proj.weight"].t().numpy()}
            if tp + "attn.gated_norm.weight" in sd:
                attn["gated_norm"] = {"weight": sd[tp + "attn.gated_norm.weight"].numpy()}

            # Skip projection
            if tp + "attn.skip_up.weight" in sd:
                attn["skip_up"] = {"kernel": sd[tp + "attn.skip_up.weight"].t().numpy()}

            # out_proj — transposed
            attn["out_proj"] = {"kernel": sd[tp + "attn.out_proj.weight"].t().numpy()}

            jax_block["SeqCondAttention_0"] = attn
            jax_params[jax_key] = jax_block
            seqcond_idx += 1
            print(f"  block {i:2d} -> {jax_key}")

    # Build full config dict matching training format
    # The training code saves config.to_dict() which is a flat dict
    jax_config = dict(config)
    # Ensure model_type is present
    if "model_type" not in jax_config:
        jax_config["model_type"] = "seqcond"

    data = {
        "params": jax_params,
        "config": jax_config,
        "step": 0,
        "opt_state": None,
    }

    print(f"Saving JAX checkpoint to {jax_path}...")
    with open(jax_path, "wb") as f:
        pickle.dump(data, f)

    import os
    size_mb = os.path.getsize(jax_path) / 1e6
    print(f"Done! ({size_mb:.0f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch .pt to JAX .pkl")
    parser.add_argument("--input", type=str, required=True, help="PyTorch checkpoint path")
    parser.add_argument("--output", type=str, required=True, help="JAX checkpoint output path")
    args = parser.parse_args()

    convert_torch_to_jax(args.input, args.output)

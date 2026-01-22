"""
Convert JAX checkpoint to torch2 format.
Matches JAX parameter names exactly.
"""

import argparse
import pickle
import torch
import numpy as np
from seqcond.torch2.model import SeqCondModel


def convert_jax_to_torch2(jax_path: str, torch_path: str):
    print(f"Loading JAX checkpoint from {jax_path}...")
    with open(jax_path, "rb") as f:
        data = pickle.load(f)

    jax_params = data["params"]
    config = data["config"]["model"]

    print("Converting weights...")
    state_dict = {}

    # Embedding
    state_dict["embedding.weight"] = torch.from_numpy(
        np.array(jax_params["token_embedding"]["embedding"])
    )

    # Final norm is initialized with ones, no weights to copy

    # Blocks
    seqcond_ratio = config.get("seqcond_ratio", 3)
    transformer_idx = 0
    seqcond_idx = 0

    for i in range(config["num_layers"]):
        torch_prefix = f"blocks.{i}."

        if (i + 1) % (seqcond_ratio + 1) == 0:
            # Transformer block
            jax_key = f"transformer_block_{transformer_idx}"
            jax_block = jax_params[jax_key]
            attn = jax_block["attn"]

            # Norms
            state_dict[torch_prefix + "norm1.scale"] = torch.from_numpy(
                np.array(jax_block["norm1"]["scale"])
            )
            state_dict[torch_prefix + "norm2.scale"] = torch.from_numpy(
                np.array(jax_block["norm2"]["scale"])
            )

            # Attention projections
            state_dict[torch_prefix + "attn.q_proj.weight"] = torch.from_numpy(
                np.array(attn["q_proj"]["kernel"])
            ).t()
            state_dict[torch_prefix + "attn.k_proj.weight"] = torch.from_numpy(
                np.array(attn["k_proj"]["kernel"])
            ).t()
            state_dict[torch_prefix + "attn.v_proj.weight"] = torch.from_numpy(
                np.array(attn["v_proj"]["kernel"])
            ).t()
            state_dict[torch_prefix + "attn.out_proj.weight"] = torch.from_numpy(
                np.array(attn["out_proj"]["kernel"])
            ).t()

            # FFN - keep ff_in together (matches JAX)
            state_dict[torch_prefix + "ff_in.weight"] = torch.from_numpy(
                np.array(jax_block["ff_in"]["kernel"])
            ).t()
            state_dict[torch_prefix + "ff_in.bias"] = torch.from_numpy(
                np.array(jax_block["ff_in"]["bias"])
            )
            state_dict[torch_prefix + "ff_out.weight"] = torch.from_numpy(
                np.array(jax_block["ff_out"]["kernel"])
            ).t()
            state_dict[torch_prefix + "ff_out.bias"] = torch.from_numpy(
                np.array(jax_block["ff_out"]["bias"])
            )

            transformer_idx += 1
        else:
            # SeqCond block
            jax_key = f"seqcond_block_{seqcond_idx}"
            jax_block = jax_params[jax_key]
            attn = jax_block["SeqCondAttention_0"]

            # Block norm
            state_dict[torch_prefix + "norm.scale"] = torch.from_numpy(
                np.array(jax_block["RMSNorm_0"]["scale"])
            )

            # Input projection
            state_dict[torch_prefix + "attn.in_proj.weight"] = torch.from_numpy(
                np.array(attn["in_proj"]["kernel"])
            ).t()

            # Conv weight: JAX shape (kernel_size, 1, channels) -> torch (channels, 1, kernel_size)
            conv_kernel = np.array(attn["conv"]["kernel"])  # (K, 1, C)
            state_dict[torch_prefix + "attn.conv_weight"] = torch.from_numpy(
                conv_kernel.transpose(2, 1, 0)
            )

            # Norms
            state_dict[torch_prefix + "attn.k_norm.scale"] = torch.from_numpy(
                np.array(attn["k_norm"]["scale"])
            )
            state_dict[torch_prefix + "attn.q_norm.scale"] = torch.from_numpy(
                np.array(attn["q_norm"]["scale"])
            )

            # Theta parameters
            if "theta_raw" in attn:
                state_dict[torch_prefix + "attn.theta_raw"] = torch.from_numpy(
                    np.array(attn["theta_raw"])
                )
                state_dict[torch_prefix + "attn.w_int_raw"] = torch.from_numpy(
                    np.array(attn["w_int_raw"])
                )
            elif "theta_d_raw" in attn:
                state_dict[torch_prefix + "attn.theta_d_raw"] = torch.from_numpy(
                    np.array(attn["theta_d_raw"])
                )

            # Decay/anchor slopes
            if "decay_slopes" in attn:
                state_dict[torch_prefix + "attn.decay_slopes"] = torch.from_numpy(
                    np.array(attn["decay_slopes"])
                )
            if "anchor_slopes" in attn:
                state_dict[torch_prefix + "attn.anchor_slopes"] = torch.from_numpy(
                    np.array(attn["anchor_slopes"])
                )

            # Score parameters
            state_dict[torch_prefix + "attn.score_scale"] = torch.from_numpy(
                np.array(attn["score_scale"])
            )
            state_dict[torch_prefix + "attn.score_bias"] = torch.from_numpy(
                np.array(attn["score_bias"])
            )

            # Phase scale
            state_dict[torch_prefix + "attn.phase_scale"] = torch.from_numpy(
                np.array(attn["phase_scale"])
            )

            # Readout
            state_dict[torch_prefix + "attn.W_readout"] = torch.from_numpy(
                np.array(attn["W_readout"])
            )

            # Gated norm
            state_dict[torch_prefix + "attn.gate_proj.weight"] = torch.from_numpy(
                np.array(attn["gate_proj"]["kernel"])
            ).t()
            state_dict[torch_prefix + "attn.gated_norm.weight"] = torch.from_numpy(
                np.array(attn["gated_norm"]["weight"])
            )

            # Skip projection
            if "skip_up" in attn:
                state_dict[torch_prefix + "attn.skip_up.weight"] = torch.from_numpy(
                    np.array(attn["skip_up"]["kernel"])
                ).t()

            # Highway scale
            state_dict[torch_prefix + "attn.highway_scale"] = torch.from_numpy(
                np.array(attn["highway_scale"])
            )

            # Output projection
            state_dict[torch_prefix + "attn.out_proj.weight"] = torch.from_numpy(
                np.array(attn["out_proj"]["kernel"])
            ).t()

            seqcond_idx += 1

    # Detect skip_low_rank from checkpoint dimensions
    # If skip_up exists in first seqcond block, skip_low_rank=True
    first_seqcond = jax_params["seqcond_block_0"]["SeqCondAttention_0"]
    skip_low_rank = "skip_up" in first_seqcond

    # Create model config
    model_config = {
        "d_model": config["d_model"],
        "d_ff": config["d_ff"],
        "num_layers": config["num_layers"],
        "vocab_size": config["vocab_size"],
        "maxlen": config["maxlen"],
        "num_heads": config["num_heads"],
        "num_kv_heads": config.get("num_kv_heads"),
        "qk_norm": config.get("qk_norm", True),
        "qk_norm_eps": config.get("qk_norm_eps", 1e-6),
        "seqcond_heads": config.get("seqcond_heads", 32),
        "num_query_heads": config.get("num_query_heads", 6),
        "num_thetas": config.get("num_thetas", 4),
        "conv_kernel_size": config.get("conv_kernel_size", 4),
        "expand_factor": config.get("expand_factor", 1),
        "out_expand_factor": config.get("out_expand_factor", 3),
        "seqcond_ratio": config.get("seqcond_ratio", 3),
        "skip_low_rank": skip_low_rank,
        "num_anchor_heads": config.get("num_anchor_heads", 0),
    }

    print(f"Saving to {torch_path}...")
    torch.save({"config": model_config, "state_dict": state_dict}, torch_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jax_path", help="Path to JAX checkpoint")
    parser.add_argument(
        "--torch_path", default=None, help="Path to save torch checkpoint"
    )
    args = parser.parse_args()

    if args.torch_path is None:
        args.torch_path = args.jax_path.replace(".pkl", "_torch2.pt")

    convert_jax_to_torch2(args.jax_path, args.torch_path)

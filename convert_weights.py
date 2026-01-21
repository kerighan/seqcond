import pickle
import torch
import numpy as np
from typing import Dict, Any


def convert_flax_to_torch(flax_checkpoint_path: str, torch_checkpoint_path: str):
    print(f"Loading Flax checkpoint from {flax_checkpoint_path}...")
    with open(flax_checkpoint_path, "rb") as f:
        data = pickle.load(f)

    flax_params = data["params"]
    config = data["config"]

    torch_state_dict = {}

    # 1. Token Embedding
    # Flax: (vocab_size, d_model) -> Torch: (vocab_size, d_model)
    torch_state_dict["embedding.weight"] = torch.from_numpy(
        np.array(flax_params["token_embedding"]["embedding"])
    )

    # 2. Blocks
    transformer_idx = 0
    seqcond_idx = 0

    # We need to know the block order from the config
    num_layers = config["model"]["num_layers"]
    seqcond_ratio = config["model"]["seqcond_ratio"]

    for i in range(num_layers):
        if (i + 1) % (seqcond_ratio + 1) == 0:
            # Transformer Block
            flax_key = f"transformer_block_{transformer_idx}"
            torch_prefix = f"blocks.{i}."
            f_block = flax_params[flax_key]

            # Attention
            attn = f_block["attn"]
            torch_state_dict[torch_prefix + "q_proj.weight"] = torch.from_numpy(
                np.array(attn["q_proj"]["kernel"])
            ).t()
            torch_state_dict[torch_prefix + "k_proj.weight"] = torch.from_numpy(
                np.array(attn["k_proj"]["kernel"])
            ).t()
            torch_state_dict[torch_prefix + "v_proj.weight"] = torch.from_numpy(
                np.array(attn["v_proj"]["kernel"])
            ).t()
            torch_state_dict[torch_prefix + "o_proj.weight"] = torch.from_numpy(
                np.array(attn["out_proj"]["kernel"])
            ).t()

            torch_state_dict[torch_prefix + "norm.weight"] = torch.from_numpy(
                np.array(f_block["norm1"]["scale"])
            )

            if "q_norm" in attn:
                torch_state_dict[torch_prefix + "q_norm.weight"] = torch.from_numpy(
                    np.array(attn["q_norm"]["scale"])
                )
                torch_state_dict[torch_prefix + "k_norm.weight"] = torch.from_numpy(
                    np.array(attn["k_norm"]["scale"])
                )

            # FFN (SwiGLU)
            torch_state_dict[torch_prefix + "ffn_norm.weight"] = torch.from_numpy(
                np.array(f_block["norm2"]["scale"])
            )

            full_ff_in = np.array(f_block["ff_in"]["kernel"])  # (960, 5120)
            gate_w, up_w = np.split(full_ff_in, 2, axis=-1)
            torch_state_dict[torch_prefix + "ffn_gate.weight"] = torch.from_numpy(
                gate_w
            ).t()
            torch_state_dict[torch_prefix + "ffn_up.weight"] = torch.from_numpy(
                up_w
            ).t()

            full_ff_bias = np.array(f_block["ff_in"]["bias"])
            gate_b, up_b = np.split(full_ff_bias, 2)
            torch_state_dict[torch_prefix + "ffn_gate.bias"] = torch.from_numpy(gate_b)
            torch_state_dict[torch_prefix + "ffn_up.bias"] = torch.from_numpy(up_b)

            torch_state_dict[torch_prefix + "ffn_down.weight"] = torch.from_numpy(
                np.array(f_block["ff_out"]["kernel"])
            ).t()
            torch_state_dict[torch_prefix + "ffn_down.bias"] = torch.from_numpy(
                np.array(f_block["ff_out"]["bias"])
            )

            transformer_idx += 1
        else:
            # SeqCond Block
            flax_key = f"seqcond_block_{seqcond_idx}"
            torch_prefix = f"blocks.{i}."
            f_block = flax_params[flax_key]

            torch_state_dict[torch_prefix + "norm.weight"] = torch.from_numpy(
                np.array(f_block["RMSNorm_0"]["scale"])
            )

            attn = f_block["SeqCondAttention_0"]
            torch_state_dict[torch_prefix + "attn.in_proj.weight"] = torch.from_numpy(
                np.array(attn["in_proj"]["kernel"])
            ).t()

            # Conv weight: Flax (K, 1, C) -> Torch (C, 1, K)
            conv_kernel = np.array(attn["conv"]["kernel"])
            torch_state_dict[torch_prefix + "attn.conv_weight"] = torch.from_numpy(
                conv_kernel
            ).permute(2, 1, 0)

            torch_state_dict[torch_prefix + "attn.k_norm.weight"] = torch.from_numpy(
                np.array(attn["k_norm"]["scale"])
            )
            torch_state_dict[torch_prefix + "attn.q_norm.weight"] = torch.from_numpy(
                np.array(attn["q_norm"]["scale"])
            )

            # Grille spectrale
            if "theta_d_raw" in attn:
                torch_state_dict[torch_prefix + "attn.theta_d_raw"] = torch.from_numpy(
                    np.array(attn["theta_d_raw"])
                )
            elif "theta_raw" in attn:
                torch_state_dict[torch_prefix + "attn.theta_raw"] = torch.from_numpy(
                    np.array(attn["theta_raw"])
                )

            if "w_int_raw" in attn:
                torch_state_dict[torch_prefix + "attn.w_int_raw"] = torch.from_numpy(
                    np.array(attn["w_int_raw"])
                )

            torch_state_dict[torch_prefix + "attn.score_scale"] = torch.from_numpy(
                np.array(attn["score_scale"])
            )
            torch_state_dict[torch_prefix + "attn.score_bias"] = torch.from_numpy(
                np.array(attn["score_bias"])
            )
            torch_state_dict[torch_prefix + "attn.phase_scale"] = torch.from_numpy(
                np.array(attn["phase_scale"])
            )

            torch_state_dict[torch_prefix + "attn.decay_slopes"] = torch.from_numpy(
                np.array(attn["decay_slopes"])
            )
            if "anchor_slopes" in attn:
                torch_state_dict[torch_prefix + "attn.anchor_slopes"] = (
                    torch.from_numpy(np.array(attn["anchor_slopes"]))
                )

            torch_state_dict[torch_prefix + "attn.gate_proj.weight"] = torch.from_numpy(
                np.array(attn["gate_proj"]["kernel"])
            ).t()
            torch_state_dict[torch_prefix + "attn.gated_norm.weight"] = (
                torch.from_numpy(np.array(attn["gated_norm"]["weight"]))
            )

            torch_state_dict[torch_prefix + "attn.W_readout"] = torch.from_numpy(
                np.array(attn["W_readout"])
            )

            if "skip_up" in attn:
                torch_state_dict[torch_prefix + "attn.skip_up.weight"] = (
                    torch.from_numpy(np.array(attn["skip_up"]["kernel"])).t()
                )

            h_scale = np.array(attn["highway_scale"])
            # Flax (1, 1, K, 1) -> Torch (1, 1, K, 1)
            torch_state_dict[torch_prefix + "attn.highway_scale"] = torch.from_numpy(
                h_scale
            )

            torch_state_dict[torch_prefix + "attn.out_proj.weight"] = torch.from_numpy(
                np.array(attn["out_proj"]["kernel"])
            ).t()

            seqcond_idx += 1

    # 3. Output Projection
    if "output_projection" in flax_params:
        torch_state_dict["output_projection.weight"] = torch.from_numpy(
            np.array(flax_params["output_projection"]["kernel"])
        ).t()
    else:
        # Tied weights
        torch_state_dict["output_projection.weight"] = torch_state_dict[
            "embedding.weight"
        ]

    print(f"Saving Torch checkpoint to {torch_checkpoint_path}...")
    torch.save(
        {"state_dict": torch_state_dict, "config": config}, torch_checkpoint_path
    )
    print("Done!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python convert_weights.py <flax_pkl_path> <torch_pt_path>")
    else:
        convert_flax_to_torch(sys.argv[1], sys.argv[2])

"""
Convert JAX checkpoint to torch format.
Matches JAX parameter names exactly.
Can also export to HuggingFace format.
"""

import argparse
import pickle
import torch
import numpy as np
import os
from seqcond.torch.model import SeqCondModel


def get_config_value(config, key, default=None):
    """Get config value from either dict or object."""
    if isinstance(config, dict):
        return config.get(key, default)
    else:
        return getattr(config, key, default)


def convert_jax_to_torch(jax_path: str, torch_path: str):
    print(f"Loading JAX checkpoint from {jax_path}...")
    with open(jax_path, "rb") as f:
        data = pickle.load(f)

    jax_params = data["params"]

    # Handle both Config object and dict formats
    config_data = data["config"]
    if hasattr(config_data, "model"):
        # Config object from count_params.py
        config = config_data.model
    else:
        # Dict format from training checkpoints
        config = config_data["model"]

    print("Converting weights...")
    state_dict = {}

    # Embedding
    state_dict["embedding.weight"] = torch.from_numpy(
        np.array(jax_params["token_embedding"]["embedding"])
    )

    # Final norm is initialized with ones, no weights to copy

    # Blocks
    model_type = get_config_value(config, "model_type", "seqcond")
    seqcond_ratio = get_config_value(config, "seqcond_ratio", 3)
    if model_type == "transformer":
        seqcond_ratio = 0
    transformer_idx = 0
    seqcond_idx = 0

    for i in range(get_config_value(config, "num_layers")):
        torch_prefix = f"blocks.{i}."

        if model_type == "transformer" or (i + 1) % (seqcond_ratio + 1) == 0:
            # Transformer block
            if model_type == "transformer":
                jax_key = f"transformer_block_{i}"
            else:
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
    if model_type == "transformer":
        skip_low_rank = False
    else:
        first_seqcond = jax_params["seqcond_block_0"]["SeqCondAttention_0"]
        skip_low_rank = "skip_up" in first_seqcond

    # Create model config
    model_config = {
        "d_model": get_config_value(config, "d_model"),
        "d_ff": get_config_value(config, "d_ff"),
        "num_layers": get_config_value(config, "num_layers"),
        "vocab_size": get_config_value(config, "vocab_size"),
        "maxlen": get_config_value(config, "maxlen"),
        "num_heads": get_config_value(config, "num_heads"),
        "num_kv_heads": get_config_value(config, "num_kv_heads"),
        "qk_norm": get_config_value(config, "qk_norm", True),
        "qk_norm_eps": get_config_value(config, "qk_norm_eps", 1e-6),
        "seqcond_heads": get_config_value(config, "seqcond_heads", 32),
        "num_query_heads": get_config_value(config, "num_query_heads", 6),
        "num_thetas": get_config_value(config, "num_thetas", 4),
        "conv_kernel_size": get_config_value(config, "conv_kernel_size", 4),
        "expand_factor": get_config_value(config, "expand_factor", 1),
        "out_expand_factor": get_config_value(config, "out_expand_factor", 3),
        "seqcond_ratio": seqcond_ratio,
        "skip_low_rank": skip_low_rank,
        "num_anchor_heads": get_config_value(config, "num_anchor_heads", 0),
    }

    print(f"Saving to {torch_path}...")
    torch.save({"config": model_config, "state_dict": state_dict}, torch_path)
    print("Done!")
    return model_config, state_dict


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)


def prepare_hf_code(output_dir):
    """Copy necessary files for trust_remote_code=True."""
    print(f"Preparing autonomous HF code in {output_dir}...")

    # 1. configuration_seqcond.py
    config_content = read_file("seqcond/config.py")
    hf_wrapper_content = read_file("seqcond/torch/hf_wrapper.py")

    start_idx = hf_wrapper_content.find("class SeqCondHFConfig")
    end_idx = hf_wrapper_content.find("class SeqCondForCausalLM")
    seqcond_hf_config_code = hf_wrapper_content[start_idx:end_idx]

    config_header = """from transformers import PretrainedConfig
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
"""
    lines = config_content.splitlines()
    filtered_lines = [
        l
        for l in lines
        if not l.startswith("from dataclasses") and not l.startswith("from typing")
    ]
    config_body = "\n".join(filtered_lines)
    final_config_content = (
        config_header + "\n" + config_body + "\n\n" + seqcond_hf_config_code
    )
    write_file(
        os.path.join(output_dir, "configuration_seqcond.py"), final_config_content
    )

    # 2. modeling_seqcond.py
    model_content = read_file("seqcond/torch/model.py")
    seqcond_content = read_file("seqcond/torch/seqcond.py")
    rope_content = read_file("seqcond/torch/rope.py")
    norm_content = read_file("seqcond/torch/norm.py")

    start_idx = hf_wrapper_content.find("class SeqCondForCausalLM")
    seqcond_for_causal_lm_code = hf_wrapper_content[start_idx:]

    modeling_header = """import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_seqcond import SeqCondHFConfig
import math
"""

    final_modeling_content = (
        modeling_header
        + "\n\n"
        + norm_content
        + "\n\n"
        + rope_content
        + "\n\n"
        + seqcond_content
        + "\n\n"
        + model_content
        + "\n\n"
        + seqcond_for_causal_lm_code
    )
    write_file(os.path.join(output_dir, "modeling_seqcond.py"), final_modeling_content)

    # 3. generation_config.json
    generation_config = {"_from_model_config": True}
    import json

    with open(os.path.join(output_dir, "generation_config.json"), "w") as f:
        json.dump(generation_config, f, indent=2)


def export_to_hf(model_config, state_dict, hf_dir):
    """Export PyTorch checkpoint to HuggingFace format."""
    from seqcond.torch.hf_wrapper import SeqCondForCausalLM, SeqCondHFConfig

    os.makedirs(hf_dir, exist_ok=True)
    print(f"Initializing HF Model...")
    hf_config = SeqCondHFConfig(**model_config)
    hf_model = SeqCondForCausalLM(hf_config)

    # Prefix keys with 'model.' for HF wrapper
    hf_state_dict = {f"model.{k}": v for k, v in state_dict.items()}

    print("Loading state dict into HF model...")
    hf_model.load_state_dict(hf_state_dict, strict=True)

    print(f"Saving HF model files to {hf_dir}...")
    hf_model.save_pretrained(hf_dir, safe_serialization=False)

    # Prepare autonomous code for trust_remote_code=True
    prepare_hf_code(hf_dir)
    print(f"HF model saved to {hf_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert JAX checkpoint to PyTorch and optionally HuggingFace"
    )
    parser.add_argument("jax_path", help="Path to JAX checkpoint")
    parser.add_argument(
        "--torch_path", default=None, help="Path to save torch checkpoint"
    )
    parser.add_argument(
        "--hf_dir", default=None, help="Directory to save HuggingFace model"
    )
    args = parser.parse_args()

    if args.torch_path is None and args.hf_dir is None:
        args.torch_path = args.jax_path.replace(".pkl", "_torch.pt")

    # Convert to PyTorch
    if args.torch_path:
        model_config, state_dict = convert_jax_to_torch(args.jax_path, args.torch_path)
    else:
        # Load for HF export only
        print(f"Loading JAX checkpoint from {args.jax_path}...")
        with open(args.jax_path, "rb") as f:
            data = pickle.load(f)
        jax_params = data["params"]
        config = data["config"]["model"]
        # Run conversion without saving
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            model_config, state_dict = convert_jax_to_torch(args.jax_path, tmp.name)

    # Export to HuggingFace if requested
    if args.hf_dir:
        export_to_hf(model_config, state_dict, args.hf_dir)

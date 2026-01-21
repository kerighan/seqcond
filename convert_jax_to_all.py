import pickle
import torch
import numpy as np
import os
import argparse
import json
from typing import Dict, Any
from seqcond.torch.hf_wrapper import SeqCondForCausalLM, SeqCondHFConfig


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)


def prepare_hf_code(output_dir):
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
    seqcond_lm_code = hf_wrapper_content[end_idx:]

    modeling_header = """import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any, Union
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_seqcond import SeqCondHFConfig, ModelConfig
"""
    model_lines = model_content.splitlines()
    model_body = []
    for line in model_lines:
        if line.startswith("import ") or line.startswith("from typing"):
            continue
        model_body.append(line)
    model_body = "\n".join(model_body)
    final_modeling_content = (
        modeling_header + "\n" + model_body + "\n\n" + seqcond_lm_code
    )
    write_file(os.path.join(output_dir, "modeling_seqcond.py"), final_modeling_content)

    # 3. Update config.json for auto_map
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_json = json.load(f)
        config_json["auto_map"] = {
            "AutoConfig": "configuration_seqcond.SeqCondHFConfig",
            "AutoModelForCausalLM": "modeling_seqcond.SeqCondForCausalLM",
        }
        config_json["model_type"] = "seqcond"
        config_json["architectures"] = ["SeqCondForCausalLM"]
        with open(config_path, "w") as f:
            json.dump(config_json, f, indent=2)


def convert_flax_to_torch_state_dict(flax_params: Dict, config: Dict) -> Dict:
    torch_state_dict = {}

    # 1. Token Embedding
    torch_state_dict["embedding.weight"] = torch.from_numpy(
        np.array(flax_params["token_embedding"]["embedding"])
    )

    # 2. Blocks
    transformer_idx = 0
    seqcond_idx = 0
    num_layers = config["model"]["num_layers"]
    seqcond_ratio = config["model"]["seqcond_ratio"]

    for i in range(num_layers):
        torch_prefix = f"blocks.{i}."
        if (i + 1) % (seqcond_ratio + 1) == 0:
            # Transformer Block
            flax_key = f"transformer_block_{transformer_idx}"
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

            full_ff_in = np.array(f_block["ff_in"]["kernel"])
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
            f_block = flax_params[flax_key]

            torch_state_dict[torch_prefix + "norm.weight"] = torch.from_numpy(
                np.array(f_block["RMSNorm_0"]["scale"])
            )

            attn = f_block["SeqCondAttention_0"]
            torch_state_dict[torch_prefix + "attn.in_proj.weight"] = torch.from_numpy(
                np.array(attn["in_proj"]["kernel"])
            ).t()

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

            torch_state_dict[torch_prefix + "attn.highway_scale"] = torch.from_numpy(
                np.array(attn["highway_scale"])
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
        torch_state_dict["output_projection.weight"] = torch_state_dict[
            "embedding.weight"
        ]

    return torch_state_dict


def main():
    parser = argparse.ArgumentParser(
        description="Convert JAX/Flax checkpoint to PyTorch and Hugging Face"
    )
    parser.add_argument("jax_path", help="Path to JAX .pkl checkpoint")
    parser.add_argument(
        "--torch_path", help="Optional path to save standard PyTorch .pt checkpoint"
    )
    parser.add_argument(
        "--hf_dir", help="Optional directory to save Hugging Face model"
    )
    args = parser.parse_args()

    if not args.torch_path and not args.hf_dir:
        print("Error: Provide at least --torch_path or --hf_dir")
        return

    print(f"Loading JAX checkpoint from {args.jax_path}...")
    with open(args.jax_path, "rb") as f:
        data = pickle.load(f)

    flax_params = data["params"]
    config = data["config"]
    model_config_dict = config["model"] if "model" in config else config

    print("Converting weights...")
    torch_state_dict = convert_flax_to_torch_state_dict(flax_params, config)

    # 1. Save standard PyTorch checkpoint
    if args.torch_path:
        print(f"Saving Torch checkpoint to {args.torch_path}...")
        torch.save({"state_dict": torch_state_dict, "config": config}, args.torch_path)

    # 2. Save Hugging Face model
    if args.hf_dir:
        os.makedirs(args.hf_dir, exist_ok=True)
        print(f"Initializing HF Model...")
        hf_config = SeqCondHFConfig(**model_config_dict)
        hf_model = SeqCondForCausalLM(hf_config)

        # Prefix keys with 'model.' for HF wrapper
        hf_state_dict = {f"model.{k}": v for k, v in torch_state_dict.items()}

        print("Loading state dict into HF model...")
        hf_model.load_state_dict(hf_state_dict, strict=True)

        print(f"Saving HF model files to {args.hf_dir}...")
        hf_model.save_pretrained(args.hf_dir, safe_serialization=False)

        # Prepare autonomous code for trust_remote_code=True
        prepare_hf_code(args.hf_dir)

    print("Done!")


if __name__ == "__main__":
    main()

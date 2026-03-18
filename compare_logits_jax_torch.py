"""
Compare forward pass logits between JAX and PyTorch for the 762k checkpoint.
Identifies where the discrepancy happens (embedding, blocks, output).

Usage:
    JAX_PLATFORMS=cpu python compare_logits_jax_torch.py
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import torch
import pickle
import numpy as np
from seqcond.jax.model import SeqCondModel as JAXModel
from seqcond.dataset import Tokenizer

TORCH_CKPT = "checkpoints/seqcond_torch_762k.pt"
JAX_CKPT = "/tmp/seqcond_jax_762k.pkl"


def load_torch_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    from seqcond.torch.model import SeqCondModel as TorchModel

    model = TorchModel(
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        vocab_size=config["vocab_size"],
        maxlen=config["maxlen"],
        num_heads=config["num_heads"],
        num_kv_heads=config.get("num_kv_heads"),
        qk_norm=config.get("qk_norm", True),
        qk_norm_eps=config.get("qk_norm_eps", 1e-6),
        seqcond_heads=config.get("seqcond_heads", config["num_heads"]),
        num_query_heads=config.get("num_query_heads", 6),
        num_thetas=config.get("num_thetas", 4),
        conv_kernel_size=config.get("conv_kernel_size", 4),
        expand_factor=config.get("expand_factor", 1),
        out_expand_factor=config.get("out_expand_factor", 3),
        seqcond_ratio=config.get("seqcond_ratio", 3),
        num_anchor_heads=config.get("num_anchor_heads", 0),
    )
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    model.eval()
    return model, config


def load_jax_model(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    config = data["config"]
    params = data["params"]
    model = JAXModel(
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        vocab_size=config["vocab_size"],
        maxlen=config["maxlen"],
        seqcond_ratio=config.get("seqcond_ratio", 3),
        num_heads=config["num_heads"],
        num_kv_heads=config.get("num_kv_heads"),
        seqcond_heads=config.get("seqcond_heads"),
        num_query_heads=config.get("num_query_heads", 6),
        num_anchor_heads=config.get("num_anchor_heads", 0),
        num_thetas=config.get("num_thetas", 4),
        conv_kernel_size=config.get("conv_kernel_size", 4),
        expand_factor=config.get("expand_factor", 1),
        out_expand_factor=config.get("out_expand_factor", 3),
        qk_norm=config.get("qk_norm", True),
        qk_norm_eps=config.get("qk_norm_eps", 1e-6),
        remat=False,
        use_square_matrix=config.get("use_square_matrix", False),
    )
    return model, {"params": params}, config


def compare_logits(prompt_text):
    """Compare forward pass logits for the same prompt."""
    tokenizer = Tokenizer()
    token_ids = tokenizer.encode(prompt_text)

    print(f"Prompt: '{prompt_text}'")
    print(f"Token IDs ({len(token_ids)}): {token_ids[:20]}...")

    # === Load models ===
    print("\nLoading Torch model...")
    torch_model, torch_config = load_torch_model(TORCH_CKPT)

    print("Loading JAX model...")
    jax_model, jax_vars, jax_config = load_jax_model(JAX_CKPT)

    # === Forward pass ===
    input_ids_torch = torch.tensor([token_ids], dtype=torch.long)
    input_ids_jax = jnp.array([token_ids], dtype=jnp.int32)

    print("\nRunning Torch forward...")
    with torch.no_grad():
        torch_logits = torch_model(input_ids_torch)  # (1, L, V)
    torch_logits_np = torch_logits[0].numpy()  # (L, V)

    print("Running JAX forward...")
    jax_logits = jax_model.apply(jax_vars, input_ids_jax)  # (1, L, V)
    jax_logits_np = np.array(jax_logits[0])  # (L, V)

    # === Compare ===
    print("\n" + "=" * 60)
    print("FORWARD PASS LOGITS COMPARISON")
    print("=" * 60)

    for pos in range(len(token_ids)):
        jl = jax_logits_np[pos]
        tl = torch_logits_np[pos]
        diff = np.abs(jl - tl)
        jax_top5 = np.argsort(jl)[-5:][::-1]
        torch_top5 = np.argsort(tl)[-5:][::-1]
        top1_match = jax_top5[0] == torch_top5[0]

        if pos < 5 or pos == len(token_ids) - 1 or not top1_match:
            tok_text = (
                tokenizer.decode([token_ids[pos]]) if pos < len(token_ids) else "?"
            )
            print(f"\nPos {pos:3d} (tok={token_ids[pos]:6d} '{tok_text.strip()}')")
            print(f"  Max diff: {diff.max():.4f}, Mean diff: {diff.mean():.6f}")
            print(
                f"  JAX  top5: {jax_top5} = {[tokenizer.decode([t]) for t in jax_top5]}"
            )
            print(
                f"  Torch top5: {torch_top5} = {[tokenizer.decode([t]) for t in torch_top5]}"
            )
            print(f"  Top-1 match: {'YES' if top1_match else 'NO <<<'}")

    # Summary
    all_diff = np.abs(jax_logits_np - torch_logits_np)
    jax_top1 = np.argmax(jax_logits_np, axis=-1)
    torch_top1 = np.argmax(torch_logits_np, axis=-1)
    agree = (jax_top1 == torch_top1).mean()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Max logit diff (overall):  {all_diff.max():.4f}")
    print(f"Mean logit diff (overall): {all_diff.mean():.6f}")
    print(f"Top-1 agreement:           {agree*100:.1f}%")
    print(
        f"Positions with mismatch:   {(jax_top1 != torch_top1).sum()}/{len(token_ids)}"
    )

    # Show generated token comparison
    print("\n" + "=" * 60)
    print("FIRST GENERATED TOKEN (greedy from last position)")
    print("=" * 60)
    jax_next = int(jax_top1[-1])
    torch_next = int(torch_top1[-1])
    print(f"JAX:   {jax_next} = '{tokenizer.decode([jax_next])}'")
    print(f"Torch: {torch_next} = '{tokenizer.decode([torch_next])}'")
    print(f"Match: {jax_next == torch_next}")


if __name__ == "__main__":
    prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    compare_logits(prompt)

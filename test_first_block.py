"""Compare first block output between JAX and PyTorch."""

import numpy as np
import subprocess
import sys

CHECKPOINT_JAX = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step30000.pkl"
CHECKPOINT_TORCH = "checkpoints/seqcond_torch_30k_v2.pt"
TOKEN_ID = 792  # "The"


def run_jax():
    import jax
    import jax.numpy as jnp
    import pickle
    from seqcond.jax.model import SeqCondModel
    from seqcond.config import ModelConfig

    print("[JAX] Loading checkpoint...")
    with open(CHECKPOINT_JAX, "rb") as f:
        ckpt = pickle.load(f)

    params = ckpt["params"]
    # Skip config loading - not needed for this test

    # Get embedding
    emb_weight = params["token_embedding"]["embedding"]
    x = emb_weight[TOKEN_ID : TOKEN_ID + 1]  # (1, 960)
    print(f"[JAX] Input embedding: mean={x.mean():.6f}, std={x.std():.6f}")

    # Get first block params (seqcond_block_0)
    block_params = params["seqcond_block_0"]

    # RMSNorm
    norm_weight = block_params["RMSNorm_0"]["scale"]
    print(f"[JAX] Norm weight shape: {norm_weight.shape}")

    # Apply RMSNorm manually
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(x_f32**2, axis=-1, keepdims=True)
    x_normed = x_f32 * jax.lax.rsqrt(variance + 1e-5)
    x_normed = x_normed * norm_weight

    print(f"[JAX] After RMSNorm: mean={x_normed.mean():.6f}, std={x_normed.std():.6f}")
    print(f"[JAX] First 10 values: {x_normed[0, :10]}")

    # Now apply in_proj
    in_proj_kernel = block_params["SeqCondAttention_0"]["in_proj"]["kernel"]
    print(f"[JAX] in_proj kernel shape: {in_proj_kernel.shape}")

    z_all = jnp.dot(x_normed, in_proj_kernel)
    print(
        f"[JAX] After in_proj: shape={z_all.shape}, mean={z_all.mean():.6f}, std={z_all.std():.6f}"
    )
    print(f"[JAX] First 10 values: {z_all[0, :10]}")

    np.save("/tmp/jax_after_inproj.npy", np.array(z_all))


def run_torch():
    import torch
    from seqcond.torch.model import SeqCondModel, RMSNorm
    from seqcond.config import ModelConfig

    print("[PyTorch] Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT_TORCH, map_location="cpu")
    state_dict = ckpt["state_dict"]

    # Get embedding
    emb_weight = state_dict["embedding.weight"]
    x = emb_weight[TOKEN_ID : TOKEN_ID + 1]  # (1, 960)
    print(f"[PyTorch] Input embedding: mean={x.mean():.6f}, std={x.std():.6f}")

    # Get first block norm weight
    norm_weight = state_dict["blocks.0.norm.weight"]
    print(f"[PyTorch] Norm weight shape: {norm_weight.shape}")

    # Apply RMSNorm manually (matching the corrected implementation)
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + 1e-5)
    x_normed = x_normed * norm_weight.float()

    print(
        f"[PyTorch] After RMSNorm: mean={x_normed.mean():.6f}, std={x_normed.std():.6f}"
    )
    print(f"[PyTorch] First 10 values: {x_normed[0, :10]}")

    # Now apply in_proj
    in_proj_weight = state_dict["blocks.0.attn.in_proj.weight"]
    print(f"[PyTorch] in_proj weight shape: {in_proj_weight.shape}")

    z_all = torch.mm(x_normed, in_proj_weight.t())
    print(
        f"[PyTorch] After in_proj: shape={z_all.shape}, mean={z_all.mean():.6f}, std={z_all.std():.6f}"
    )
    print(f"[PyTorch] First 10 values: {z_all[0, :10]}")

    np.save("/tmp/torch_after_inproj.npy", z_all.numpy())


def compare():
    jax_out = np.load("/tmp/jax_after_inproj.npy")
    torch_out = np.load("/tmp/torch_after_inproj.npy")

    print("\n" + "=" * 60)
    print("AFTER IN_PROJ COMPARISON")
    print("=" * 60)

    diff = jax_out - torch_out
    print(f"Max absolute diff: {np.abs(diff).max():.10f}")
    print(f"Mean absolute diff: {np.abs(diff).mean():.10f}")

    if np.allclose(jax_out, torch_out, atol=1e-5):
        print("✅ After RMSNorm outputs are IDENTICAL")
    else:
        print("❌ After RMSNorm outputs DIFFER")
        # Find where they differ most
        idx = np.argmax(np.abs(diff).flatten())
        print(
            f"   Max diff at index {idx}: JAX={jax_out.flatten()[idx]:.6f}, PyTorch={torch_out.flatten()[idx]:.6f}"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "jax":
            run_jax()
        elif sys.argv[1] == "torch":
            run_torch()
        elif sys.argv[1] == "compare":
            compare()
    else:
        subprocess.run([sys.executable, __file__, "jax"], check=True)
        print()
        subprocess.run([sys.executable, __file__, "torch"], check=True)
        compare()

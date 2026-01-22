"""Compare embeddings between JAX and PyTorch to find divergence source."""

import numpy as np
import subprocess
import sys

CHECKPOINT_JAX = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step30000.pkl"
CHECKPOINT_TORCH = "checkpoints/seqcond_torch_30k_v2.pt"
TOKEN_ID = 792  # "The"


def run_jax():
    import jax.numpy as jnp
    import pickle

    print("[JAX] Loading checkpoint...")
    with open(CHECKPOINT_JAX, "rb") as f:
        ckpt = pickle.load(f)

    params = ckpt["params"]

    # Get embedding weight
    emb_weight = params["token_embedding"]["embedding"]
    print(f"[JAX] Embedding shape: {emb_weight.shape}")

    # Get embedding for token 792
    emb = emb_weight[TOKEN_ID]
    print(f"[JAX] Embedding for token {TOKEN_ID}: shape={emb.shape}")
    print(f"[JAX] First 10 values: {emb[:10]}")
    print(f"[JAX] Mean: {emb.mean():.6f}, Std: {emb.std():.6f}")

    np.save("/tmp/jax_embedding.npy", np.array(emb))


def run_torch():
    import torch

    print("[PyTorch] Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT_TORCH, map_location="cpu")

    # Get embedding weight
    emb_weight = ckpt["state_dict"]["embedding.weight"]
    print(f"[PyTorch] Embedding shape: {emb_weight.shape}")

    # Get embedding for token 792
    emb = emb_weight[TOKEN_ID].numpy()
    print(f"[PyTorch] Embedding for token {TOKEN_ID}: shape={emb.shape}")
    print(f"[PyTorch] First 10 values: {emb[:10]}")
    print(f"[PyTorch] Mean: {emb.mean():.6f}, Std: {emb.std():.6f}")

    np.save("/tmp/torch_embedding.npy", emb)


def compare():
    jax_emb = np.load("/tmp/jax_embedding.npy")
    torch_emb = np.load("/tmp/torch_embedding.npy")

    print("\n" + "=" * 60)
    print("EMBEDDING COMPARISON")
    print("=" * 60)

    diff = jax_emb - torch_emb
    print(f"Max absolute diff: {np.abs(diff).max():.10f}")
    print(f"Mean absolute diff: {np.abs(diff).mean():.10f}")

    if np.allclose(jax_emb, torch_emb, atol=1e-6):
        print("✅ Embeddings are IDENTICAL")
    else:
        print("❌ Embeddings DIFFER")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "jax":
            run_jax()
        elif sys.argv[1] == "torch":
            run_torch()
        elif sys.argv[1] == "compare":
            compare()
    else:
        print("=" * 60)
        print("Comparing Embeddings: JAX vs PyTorch")
        print("=" * 60)

        subprocess.run([sys.executable, __file__, "jax"], check=True)
        print()
        subprocess.run([sys.executable, __file__, "torch"], check=True)
        compare()

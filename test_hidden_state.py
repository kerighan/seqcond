"""Compare hidden state before output projection."""

import subprocess
import sys
import numpy as np

CHECKPOINT_JAX = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step50000.pkl"
CHECKPOINT_TORCH = "checkpoints/seqcond_torch_50k.pt"
TOKEN_ID = 792  # "The"


def run_jax():
    import jax.numpy as jnp
    import pickle

    print("[JAX] Loading checkpoint...")
    with open(CHECKPOINT_JAX, "rb") as f:
        ckpt = pickle.load(f)

    from seqcond.jax.generator import Generator as JAXGenerator, _make_step_fn

    jax_gen = JAXGenerator(CHECKPOINT_JAX)

    states = jax_gen.model.apply(
        {"params": jax_gen.params},
        batch_size=1,
        method=jax_gen.model.init_state,
    )

    # We need to manually run through the model to capture hidden state
    # Let's use a modified step that returns hidden state
    token_array = jnp.array([[TOKEN_ID]], dtype=jnp.int32)
    pos = jnp.array(0, dtype=jnp.int32)

    # Run step
    logits, new_states = _make_step_fn(jax_gen.model, jax_gen.params)(
        token_array, states, pos
    )

    # To get hidden state, we need to trace through manually
    # Let's compute it from logits and embedding
    emb = ckpt["params"]["token_embedding"]["embedding"]  # (V, D)
    logits_np = np.array(logits[0])  # (V,)

    # logits = hidden @ emb.T, so hidden = logits @ pinv(emb.T) = logits @ emb @ pinv(emb @ emb.T)
    # This is approximate. Better to modify the model to return hidden.

    # For now, just save logits
    np.save("/tmp/jax_logits_debug.npy", logits_np)
    print(f"[JAX] Logits saved, shape: {logits_np.shape}")
    print(f"[JAX] Logits[:5]: {logits_np[:5]}")
    print(f"[JAX] Logits mean: {logits_np.mean():.4f}, std: {logits_np.std():.4f}")
    print(f"[JAX] Argmax: {np.argmax(logits_np)}")


def run_torch():
    import torch
    from seqcond.torch.generator import TorchGenerator

    print("[PyTorch] Loading model...")
    torch_gen = TorchGenerator(CHECKPOINT_TORCH)
    model = torch_gen.model

    state = model.init_state(batch_size=1, device="cuda")

    # Hook to capture hidden state before output projection
    hidden_state = {}

    def hook(module, input, output):
        hidden_state["input"] = input[0].detach().cpu().numpy()

    handle = model.output_projection.register_forward_hook(hook)

    with torch.no_grad():
        token = torch.tensor([[TOKEN_ID]], device="cuda")
        logits, new_state = model.step(token, state)

    handle.remove()

    logits_np = logits[0].cpu().numpy()
    hidden_np = hidden_state["input"][0]

    np.save("/tmp/torch_logits_debug.npy", logits_np)
    np.save("/tmp/torch_hidden_debug.npy", hidden_np)

    print(f"[PyTorch] Logits saved, shape: {logits_np.shape}")
    print(f"[PyTorch] Logits[:5]: {logits_np[:5]}")
    print(f"[PyTorch] Logits mean: {logits_np.mean():.4f}, std: {logits_np.std():.4f}")
    print(f"[PyTorch] Argmax: {np.argmax(logits_np)}")
    print(f"[PyTorch] Hidden state shape: {hidden_np.shape}")
    print(f"[PyTorch] Hidden mean: {hidden_np.mean():.4f}, std: {hidden_np.std():.4f}")


def compare():
    jax_logits = np.load("/tmp/jax_logits_debug.npy")
    torch_logits = np.load("/tmp/torch_logits_debug.npy")
    torch_hidden = np.load("/tmp/torch_hidden_debug.npy")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    print(f"\nLogits comparison:")
    print(f"  Max diff: {np.abs(jax_logits - torch_logits).max():.4f}")
    print(f"  Mean diff: {np.abs(jax_logits - torch_logits).mean():.4f}")

    print(f"\nPyTorch hidden state stats:")
    print(f"  Shape: {torch_hidden.shape}")
    print(f"  Mean: {torch_hidden.mean():.4f}")
    print(f"  Std: {torch_hidden.std():.4f}")
    print(f"  Min: {torch_hidden.min():.4f}")
    print(f"  Max: {torch_hidden.max():.4f}")


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

"""Compare final hidden state before output projection."""

import subprocess
import sys
import numpy as np

CHECKPOINT_JAX = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step50000.pkl"
CHECKPOINT_TORCH = "checkpoints/seqcond_torch_50k.pt"
TOKEN_ID = 792  # "The"


def run_jax():
    import jax.numpy as jnp
    from seqcond.jax.generator import Generator as JAXGenerator, _make_step_fn

    print("[JAX] Loading model...")
    jax_gen = JAXGenerator(CHECKPOINT_JAX)

    # We need to capture the hidden state before output projection
    # Let's run the full forward and save intermediate
    states = jax_gen.model.apply(
        {"params": jax_gen.params},
        batch_size=1,
        method=jax_gen.model.init_state,
    )

    step_fn = _make_step_fn(jax_gen.model, jax_gen.params)
    token_array = jnp.array([[TOKEN_ID]], dtype=jnp.int32)
    pos = jnp.array(0, dtype=jnp.int32)
    logits, new_states = step_fn(token_array, states, pos)

    # Save logits
    np.save("/tmp/jax_logits_single.npy", np.array(logits[0]))
    print(f"[JAX] Logits shape: {logits.shape}")
    print(f"[JAX] Logits[0,:5]: {logits[0, :5]}")
    print(f"[JAX] Argmax: {np.argmax(logits[0])}")

    # Top 5
    top5_idx = np.argsort(logits[0])[-5:][::-1]
    print(f"[JAX] Top 5: {top5_idx} with values {logits[0, top5_idx]}")


def run_torch():
    import torch
    from seqcond.torch.generator import TorchGenerator

    print("[PyTorch] Loading model...")
    torch_gen = TorchGenerator(CHECKPOINT_TORCH)

    state = torch_gen.model.init_state(batch_size=1, device="cuda")

    with torch.no_grad():
        token = torch.tensor([[TOKEN_ID]], device="cuda")
        logits, new_state = torch_gen.model.step(token, state)

    logits_np = logits[0].cpu().numpy()
    np.save("/tmp/torch_logits_single.npy", logits_np)
    print(f"[PyTorch] Logits shape: {logits.shape}")
    print(f"[PyTorch] Logits[0,:5]: {logits_np[:5]}")
    print(f"[PyTorch] Argmax: {np.argmax(logits_np)}")

    # Top 5
    top5_idx = np.argsort(logits_np)[-5:][::-1]
    print(f"[PyTorch] Top 5: {top5_idx} with values {logits_np[top5_idx]}")


def compare():
    jax_logits = np.load("/tmp/jax_logits_single.npy")
    torch_logits = np.load("/tmp/torch_logits_single.npy")

    print("\n" + "=" * 60)
    print("SINGLE TOKEN LOGITS COMPARISON")
    print("=" * 60)

    print(f"Max diff: {np.abs(jax_logits - torch_logits).max():.6f}")
    print(f"Mean diff: {np.abs(jax_logits - torch_logits).mean():.6f}")

    # Find where the biggest differences are
    diff = np.abs(jax_logits - torch_logits)
    top_diff_idx = np.argsort(diff)[-10:][::-1]
    print(f"\nTop 10 differing tokens:")
    for idx in top_diff_idx:
        print(
            f"  Token {idx}: JAX={jax_logits[idx]:.4f}, Torch={torch_logits[idx]:.4f}, diff={diff[idx]:.4f}"
        )

    jax_argmax = np.argmax(jax_logits)
    torch_argmax = np.argmax(torch_logits)
    if jax_argmax == torch_argmax:
        print(f"\n✅ Argmax MATCH: {jax_argmax}")
    else:
        print(f"\n❌ Argmax DIFFER: JAX={jax_argmax}, PyTorch={torch_argmax}")


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

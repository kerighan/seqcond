"""Compare SeqCond step output between JAX and PyTorch."""

import subprocess
import sys
import numpy as np

CHECKPOINT_JAX = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step30000.pkl"
CHECKPOINT_TORCH = "checkpoints/seqcond_torch_30k_v2.pt"
TOKEN_ID = 792


def run_jax():
    import jax
    import jax.numpy as jnp
    import pickle
    from seqcond.jax.generator import Generator as JAXGenerator, _make_step_fn

    print("[JAX] Loading model...")
    jax_gen = JAXGenerator(CHECKPOINT_JAX)

    # Initialize state
    states = jax_gen.model.apply(
        {"params": jax_gen.params},
        batch_size=1,
        method=jax_gen.model.init_state,
    )

    # Create step function
    step_fn = _make_step_fn(jax_gen.model, jax_gen.params)

    # Run single step
    token_array = jnp.array([[TOKEN_ID]], dtype=jnp.int32)
    pos = jnp.array(0, dtype=jnp.int32)
    logits, new_states = step_fn(token_array, states, pos)

    logits_np = np.array(logits[0])
    print(f"[JAX] Logits shape: {logits_np.shape}")
    print(f"[JAX] Logits mean: {logits_np.mean():.6f}, std: {logits_np.std():.6f}")
    print(f"[JAX] Top-5 indices: {np.argsort(logits_np)[-5:][::-1]}")

    np.save("/tmp/jax_step_logits.npy", logits_np)


def run_torch():
    import torch
    from seqcond.torch.generator import TorchGenerator

    print("[PyTorch] Loading model...")
    torch_gen = TorchGenerator(CHECKPOINT_TORCH)

    # Initialize state
    state = torch_gen.model.init_state(batch_size=1, device="cuda")

    # Run single step
    with torch.no_grad():
        token = torch.tensor([[TOKEN_ID]], device="cuda")
        logits, new_state = torch_gen.model.step(token, state)

    logits_np = logits[0].cpu().numpy()
    print(f"[PyTorch] Logits shape: {logits_np.shape}")
    print(f"[PyTorch] Logits mean: {logits_np.mean():.6f}, std: {logits_np.std():.6f}")
    print(f"[PyTorch] Top-5 indices: {np.argsort(logits_np)[-5:][::-1]}")

    np.save("/tmp/torch_step_logits.npy", logits_np)


def compare():
    jax_logits = np.load("/tmp/jax_step_logits.npy")
    torch_logits = np.load("/tmp/torch_step_logits.npy")

    print("\n" + "=" * 60)
    print("STEP LOGITS COMPARISON")
    print("=" * 60)

    diff = jax_logits - torch_logits
    print(f"Max absolute diff: {np.abs(diff).max():.6f}")
    print(f"Mean absolute diff: {np.abs(diff).mean():.6f}")
    print(f"Correlation: {np.corrcoef(jax_logits, torch_logits)[0,1]:.6f}")

    jax_argmax = np.argmax(jax_logits)
    torch_argmax = np.argmax(torch_logits)
    print(f"\nJAX argmax: {jax_argmax}")
    print(f"PyTorch argmax: {torch_argmax}")

    if jax_argmax == torch_argmax:
        print("✅ Argmax tokens MATCH!")
    else:
        print("❌ Argmax tokens DIFFER!")


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

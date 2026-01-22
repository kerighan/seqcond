"""Compare logits between JAX and PyTorch models to find the source of divergence."""

import numpy as np
import torch
import sys

CHECKPOINT_JAX = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step50000.pkl"
CHECKPOINT_TORCH = "checkpoints/seqcond_torch_50k.pt"
PROMPT = "The quick brown fox"  # 4 tokens - divergence at token 5


def run_jax_logits():
    """Get logits from JAX model."""
    import jax
    import jax.numpy as jnp
    from seqcond.jax.generator import Generator as JAXGenerator, _make_step_fn
    from seqcond.dataset import Tokenizer

    print("[JAX] Loading model...")
    jax_gen = JAXGenerator(CHECKPOINT_JAX)
    tokenizer = Tokenizer()

    # Tokenize prompt
    tokens = tokenizer([PROMPT])[0]
    print(f"[JAX] Prompt tokens: {tokens}")

    # Get logits for next token using the generator's step function
    print("[JAX] Computing logits...")

    # Initialize state
    states = jax_gen.model.apply(
        {"params": jax_gen.params},
        batch_size=1,
        method=jax_gen.model.init_state,
    )

    # Create step function
    step_fn = _make_step_fn(jax_gen.model, jax_gen.params)

    # Process prompt tokens one by one (pos must be a scalar for JAX)
    for i, tok in enumerate(tokens):
        token_array = jnp.array([[tok]], dtype=jnp.int32)
        pos = jnp.array(i, dtype=jnp.int32)  # Scalar, not array
        logits, states = step_fn(token_array, states, pos)

    # Get final logits
    logits_np = np.array(logits[0])

    # Get top-5 predictions
    top5_idx = np.argsort(logits_np)[-5:][::-1]
    print(f"[JAX] Top-5 next tokens:")
    for idx in top5_idx:
        token_str = tokenizer.decode([idx])
        print(f"  {idx:5d} ({token_str!r:10s}): {logits_np[idx]:.4f}")

    # Save logits
    np.save("/tmp/jax_logits.npy", logits_np)
    print(f"[JAX] Logits saved. Shape: {logits_np.shape}")

    # Also get argmax
    argmax_token = int(np.argmax(logits_np))
    print(f"[JAX] Argmax token: {argmax_token} ({tokenizer.decode([argmax_token])!r})")


def run_torch_logits():
    """Get logits from PyTorch model."""
    from seqcond.torch.generator import TorchGenerator
    from seqcond.dataset import Tokenizer

    print("[PyTorch] Loading model...")
    torch_gen = TorchGenerator(CHECKPOINT_TORCH)
    tokenizer = Tokenizer()

    # Tokenize prompt
    tokens = tokenizer([PROMPT])[0]
    print(f"[PyTorch] Prompt tokens: {tokens}")

    # Get logits for next token
    print("[PyTorch] Computing logits...")

    # Initialize state
    state = torch_gen.model.init_state(batch_size=1, device="cuda")

    # Process prompt tokens one by one
    with torch.no_grad():
        for i, tok in enumerate(tokens):
            x = torch.tensor([[tok]], device="cuda")
            logits, state = torch_gen.model.step(x, state)

    # Get final logits
    logits_np = logits[0].cpu().numpy()

    # Get top-5 predictions
    top5_idx = np.argsort(logits_np)[-5:][::-1]
    print(f"[PyTorch] Top-5 next tokens:")
    for idx in top5_idx:
        token_str = tokenizer.decode([idx])
        print(f"  {idx:5d} ({token_str!r:10s}): {logits_np[idx]:.4f}")

    # Save logits
    np.save("/tmp/torch_logits.npy", logits_np)
    print(f"[PyTorch] Logits saved. Shape: {logits_np.shape}")

    # Also get argmax
    argmax_token = int(np.argmax(logits_np))
    print(
        f"[PyTorch] Argmax token: {argmax_token} ({tokenizer.decode([argmax_token])!r})"
    )


def compare_logits():
    """Compare saved logits."""
    jax_logits = np.load("/tmp/jax_logits.npy")
    torch_logits = np.load("/tmp/torch_logits.npy")

    print("\n" + "=" * 60)
    print("LOGITS COMPARISON")
    print("=" * 60)

    # Basic stats
    diff = jax_logits - torch_logits
    print(f"Max absolute diff: {np.abs(diff).max():.6f}")
    print(f"Mean absolute diff: {np.abs(diff).mean():.6f}")
    print(f"Correlation: {np.corrcoef(jax_logits, torch_logits)[0,1]:.6f}")

    # Argmax comparison
    jax_argmax = np.argmax(jax_logits)
    torch_argmax = np.argmax(torch_logits)
    print(f"\nJAX argmax: {jax_argmax}")
    print(f"PyTorch argmax: {torch_argmax}")

    if jax_argmax == torch_argmax:
        print("✅ Argmax tokens match!")
    else:
        print("❌ Argmax tokens differ!")
        from seqcond.dataset import Tokenizer

        tokenizer = Tokenizer()
        print(f"  JAX predicts: {tokenizer.decode([jax_argmax])!r}")
        print(f"  PyTorch predicts: {tokenizer.decode([torch_argmax])!r}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "jax":
            run_jax_logits()
        elif sys.argv[1] == "torch":
            run_torch_logits()
        elif sys.argv[1] == "compare":
            compare_logits()
    else:
        import subprocess

        print("=" * 60)
        print("Comparing Logits: JAX vs PyTorch")
        print("=" * 60)

        # Run JAX
        print("\n[1/2] Running JAX...")
        subprocess.run([sys.executable, __file__, "jax"], check=True)

        # Run PyTorch
        print("\n[2/2] Running PyTorch...")
        subprocess.run([sys.executable, __file__, "torch"], check=True)

        # Compare
        compare_logits()

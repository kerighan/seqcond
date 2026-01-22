"""Debug script to compare JAX and PyTorch models block by block.

This script loads both models, runs them step by step, and compares
the latent states at each block to identify where the divergence occurs.
"""

import numpy as np
import jax
import jax.numpy as jnp
import torch
from seqcond.jax.generator import Generator as JAXGenerator
from seqcond.torch.generator import TorchGenerator
from seqcond.dataset import Tokenizer

CHECKPOINT_JAX = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step60000.pkl"
CHECKPOINT_TORCH = "checkpoints/seqcond_torch_60k.pt"
PROMPT = "NAD+ is"
MAX_TOKENS = 10


def cosine_similarity(a, b):
    """Compute cosine similarity between two arrays."""
    a = a.flatten()
    b = b.flatten()
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if (norm_a > 0 and norm_b > 0) else 0


def compare_block_outputs():
    """Compare outputs block by block."""
    print("Loading models...")

    # Load JAX model
    jax_gen = JAXGenerator(CHECKPOINT_JAX)

    # Load PyTorch model
    torch_gen = TorchGenerator(CHECKPOINT_TORCH)

    # Tokenize prompt
    tokenizer = Tokenizer()
    prompt_tokens = tokenizer([PROMPT])[0]
    print(f"Prompt: {PROMPT}")
    print(f"Prompt tokens: {prompt_tokens}")

    # Initialize states
    print("\nInitializing states...")
    jax_state = jax_gen.model.apply(
        {"params": jax_gen.params},
        batch_size=1,
        method=jax_gen.model.init_state,
    )
    torch_state = torch_gen.model.init_state(
        1, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # We'll use the step function directly, which handles embedding internally
    print(f"Prompt has {len(prompt_tokens)} tokens")

    # Run through each token step by step
    print("\n" + "=" * 80)
    print("STEP-BY-STEP COMPARISON")
    print("=" * 80)

    all_jax_latents = []
    all_torch_latents = []

    for step in range(len(prompt_tokens)):
        print(f"\n--- Step {step} (Token: {prompt_tokens[step]}) ---")

        # JAX step
        token_array = jnp.array([[prompt_tokens[step]]], dtype=jnp.int32)
        pos_array = jnp.array(step, dtype=jnp.int32)
        jax_output, jax_state = jax_gen.model.apply(
            {"params": jax_gen.params},
            token_array,
            jax_state,
            pos_array,
            deterministic=True,
            method=jax_gen.model.step,
        )

        # PyTorch step
        torch_token = torch.tensor(
            [[prompt_tokens[step]]],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        with torch.no_grad():
            torch_output, torch_state = torch_gen.model.step(torch_token, torch_state)

        print(f"JAX output shape: {jax_output.shape}")
        print(f"Torch output shape: {torch_output.shape}")

        # Compare final outputs
        output_sim = cosine_similarity(jax_output, torch_output.cpu().numpy())
        print(f"Final output similarity: {output_sim:.6f}")

        # Get top predictions
        jax_logits_np = np.array(jax_output[0])
        torch_logits_np = torch_output[0].cpu().numpy()

        jax_top3 = np.argsort(jax_logits_np)[-3:][::-1]
        torch_top3 = np.argsort(torch_logits_np)[-3:][::-1]

        print(f"JAX top 3: {jax_top3} -> {[tokenizer.decode([t]) for t in jax_top3]}")
        print(
            f"Torch top 3: {torch_top3} -> {[tokenizer.decode([t]) for t in torch_top3]}"
        )

        # Check argmax divergence
        jax_argmax = np.argmax(jax_logits_np)
        torch_argmax = np.argmax(torch_logits_np)

        if jax_argmax != torch_argmax:
            print(f"⚠️  Argmax divergence at step {step}!")
            print(f"  JAX: {jax_argmax} ({tokenizer.decode([jax_argmax])!r})")
            print(f"  Torch: {torch_argmax} ({tokenizer.decode([torch_argmax])!r})")
        else:
            print(
                f"✓ Argmax matches: {jax_argmax} ({tokenizer.decode([jax_argmax])!r})"
            )

        # Early exit if we've processed enough
        if step >= MAX_TOKENS - 1:
            break

    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(
        "Step-by-step comparison completed. Check individual steps for divergence points."
    )

    return []  # No block latents since we're not doing block-by-block comparison


if __name__ == "__main__":
    try:
        compare_block_outputs()
        print(f"\nStep-by-step comparison completed successfully!")
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback

        traceback.print_exc()

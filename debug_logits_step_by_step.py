"""Debug script to compare logits step by step.

This script compares logits at each token position to find where divergence starts.
"""

import numpy as np
import torch
import jax
import jax.numpy as jnp
from seqcond.jax.generator import Generator as JAXGenerator, _make_step_fn
from seqcond.torch.generator import TorchGenerator
from seqcond.dataset import Tokenizer

CHECKPOINT_JAX = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step40000.pkl"
CHECKPOINT_TORCH = "checkpoints/seqcond_torch_40k.pt"
PROMPT = "The quick brown fox"


def compare_step_by_step_logits():
    """Compare logits at each step."""
    print("Loading models...")

    # Load JAX model
    jax_gen = JAXGenerator(CHECKPOINT_JAX)

    # Load PyTorch model
    torch_gen = TorchGenerator(CHECKPOINT_TORCH)

    # Tokenize prompt
    tokenizer = Tokenizer()
    tokens = tokenizer([PROMPT])[0]
    print(f"Prompt: {PROMPT}")
    print(f"Prompt tokens: {tokens}")
    print(f"Prompt token strings: {[tokenizer.decode([t]) for t in tokens]}")

    # Initialize JAX state
    jax_states = jax_gen.model.apply(
        {"params": jax_gen.params},
        batch_size=1,
        method=jax_gen.model.init_state,
    )

    # Initialize PyTorch state
    torch_states = torch_gen.model.init_state(batch_size=1, device="cuda")

    # Create JAX step function
    jax_step_fn = _make_step_fn(jax_gen.model, jax_gen.params)

    print("\n" + "=" * 80)
    print("STEP-BY-STEP LOGITS COMPARISON")
    print("=" * 80)

    all_correlations = []
    all_max_diffs = []

    for step, token_id in enumerate(tokens):
        print(
            f"\n--- Step {step}: Token {token_id} ('{tokenizer.decode([token_id])}') ---"
        )

        # JAX step
        token_array = jnp.array([[token_id]], dtype=jnp.int32)
        pos_array = jnp.array(step, dtype=jnp.int32)
        jax_logits, jax_states = jax_step_fn(token_array, jax_states, pos_array)

        # PyTorch step
        torch_token = torch.tensor([[token_id]], device="cuda")
        with torch.no_grad():
            torch_logits, torch_states = torch_gen.model.step(torch_token, torch_states)

        # Convert to numpy
        jax_logits_np = np.array(jax_logits[0])
        torch_logits_np = torch_logits[0].cpu().numpy()

        # Compute statistics
        diff = jax_logits_np - torch_logits_np
        correlation = np.corrcoef(jax_logits_np, torch_logits_np)[0, 1]
        max_diff = np.abs(diff).max()
        mean_diff = np.abs(diff).mean()

        all_correlations.append(correlation)
        all_max_diffs.append(max_diff)

        print(f"Correlation: {correlation:.6f}")
        print(f"Max absolute diff: {max_diff:.6f}")
        print(f"Mean absolute diff: {mean_diff:.6f}")

        # Get top predictions
        jax_top3 = np.argsort(jax_logits_np)[-3:][::-1]
        torch_top3 = np.argsort(torch_logits_np)[-3:][::-1]

        print(f"JAX top 3: {[tokenizer.decode([t]) for t in jax_top3]}")
        print(f"Torch top 3: {[tokenizer.decode([t]) for t in torch_top3]}")

        # Check if argmax differs
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

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Overall correlation range: {min(all_correlations):.6f} to {max(all_correlations):.6f}"
    )
    print(
        f"Overall max diff range: {min(all_max_diffs):.6f} to {max(all_max_diffs):.6f}"
    )

    # Find step with worst correlation
    worst_corr_step = np.argmin(all_correlations)
    print(
        f"Worst correlation at step {worst_corr_step}: {all_correlations[worst_corr_step]:.6f}"
    )

    # Find step with largest max diff
    worst_diff_step = np.argmax(all_max_diffs)
    print(
        f"Largest max diff at step {worst_diff_step}: {all_max_diffs[worst_diff_step]:.6f}"
    )


if __name__ == "__main__":
    try:
        compare_step_by_step_logits()
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback

        traceback.print_exc()

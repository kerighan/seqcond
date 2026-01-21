import torch
import time
import numpy as np
import jax
import jax.numpy as jnp
import pickle
from seqcond.torch.generator import TorchGenerator
from seqcond.jax.generator import Generator as JAXGenerator
from seqcond.dataset import Tokenizer

CHECKPOINT_JAX = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step20000.pkl"
CHECKPOINT_TORCH = "checkpoints/seqcond_torch.pt"


def test_torch_jax_equivalence():
    print("=" * 60)
    print("Testing Equivalence: PyTorch vs JAX (Argmax)")
    print("=" * 60)

    prompt = "The quick brown fox"
    max_new_tokens = 128

    # 1. JAX Generation
    print("\nRunning JAX generation (Stepwise)...")
    jax_gen = JAXGenerator(CHECKPOINT_JAX)
    # Warmup
    _ = jax_gen.generate(
        prompt=prompt, max_new_tokens=1, temperature=0.0, verbose=False
    )

    t0 = time.time()
    jax_text = jax_gen.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,  # Greedy
        verbose=False,
    )
    print(jax_text)
    jax_duration = time.time() - t0
    print(f"JAX Output: {jax_text}")
    print(
        f"JAX Duration: {jax_duration:.2f}s ({jax_duration/max_new_tokens*1000:.1f}ms/token)"
    )

    # FREE JAX MEMORY
    del jax_gen
    import gc
    import jax

    # JAX doesn't have a direct equivalent to empty_cache, but gc and giving back to system helps
    gc.collect()
    # On some systems, JAX pre-allocates VRAM. This is a best effort.

    # 2. PyTorch Generation
    print("\nRunning PyTorch generation (Stepwise)...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch_gen = TorchGenerator(CHECKPOINT_TORCH)
    # Warmup
    _ = torch_gen.generate(
        prompt=prompt, max_new_tokens=1, temperature=0.0, verbose=False
    )

    t0 = time.time()
    torch_text = torch_gen.generate(
        prompt=prompt, max_new_tokens=max_new_tokens, temperature=0.0, verbose=False
    )
    torch_duration = time.time() - t0
    print(f"Torch Output: {torch_text}")
    print(
        f"Torch Duration: {torch_duration:.2f}s ({torch_duration/max_new_tokens*1000:.1f}ms/token)"
    )

    # 3. Compare
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(
        f"JAX:   {jax_duration:.2f}s ({jax_duration/max_new_tokens*1000:.1f}ms/token)"
    )
    print(
        f"Torch: {torch_duration:.2f}s ({torch_duration/max_new_tokens*1000:.1f}ms/token)"
    )
    print("-" * 30)
    print(f"Speedup Torch vs JAX: {jax_duration / torch_duration:.2f}x")

    if jax_text == torch_text:
        print("\n" + "✓" * 60)
        print("SUCCESS: Both frameworks produced identical text!")
        print("✓" * 60)
        return True
    else:
        print("\n" + "✗" * 60)
        print("FAILURE: Outputs differ!")
        tokenizer = Tokenizer()
        tokens_jax = tokenizer([jax_text])[0]
        tokens_torch = tokenizer([torch_text])[0]

        print("\nToken comparison:")
        min_len = min(len(tokens_jax), len(tokens_torch))
        for i in range(min_len):
            char_jax = tokenizer.decode([tokens_jax[i]])
            char_torch = tokenizer.decode([tokens_torch[i]])
            status = "✓" if tokens_jax[i] == tokens_torch[i] else "✗"
            print(
                f"Pos {i:2d}: JAX={tokens_jax[i]:5d} ({char_jax!r}) | Torch={tokens_torch[i]:5d} ({char_torch!r}) | {status}"
            )
        return False


if __name__ == "__main__":
    test_torch_jax_equivalence()

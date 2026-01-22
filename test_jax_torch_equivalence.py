"""Test equivalence between JAX (ground truth) and PyTorch models at temperature 0.

Run JAX and PyTorch in separate subprocesses to avoid GPU memory conflicts.
"""

import subprocess
import sys

CHECKPOINT_JAX = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step60000.pkl"
CHECKPOINT_TORCH = "checkpoints/seqcond_torch_60k.pt"
PROMPT = "NAD+ is"
MAX_TOKENS = 100


def run_jax():
    """Run JAX generation and return output text."""
    import time
    from seqcond.jax.generator import Generator as JAXGenerator

    print("[JAX] Loading model...")
    jax_gen = JAXGenerator(CHECKPOINT_JAX)

    # Warmup
    _ = jax_gen.generate(
        prompt=PROMPT, max_new_tokens=1, temperature=0.0, verbose=False
    )

    print("[JAX] Generating...")
    t0 = time.time()
    jax_text = jax_gen.generate(
        prompt=PROMPT,
        max_new_tokens=MAX_TOKENS,
        temperature=0.0,
        verbose=False,
    )
    duration = time.time() - t0
    print(f"[JAX] Duration: {duration:.2f}s ({MAX_TOKENS/duration:.1f} tok/s)")
    print(f"[JAX] Output:\n{jax_text}")

    # Write output to file for comparison
    with open("/tmp/jax_output.txt", "w") as f:
        f.write(jax_text)

    return jax_text


def run_torch():
    """Run PyTorch generation and return output text."""
    import time
    import torch
    from seqcond.torch.generator import TorchGenerator

    print("[PyTorch] Loading model...")
    torch_gen = TorchGenerator(CHECKPOINT_TORCH)

    # Warmup
    _ = torch_gen.generate(
        prompt=PROMPT, max_new_tokens=1, temperature=0.0, verbose=False
    )

    print("[PyTorch] Generating...")
    t0 = time.time()
    torch_text = torch_gen.generate(
        prompt=PROMPT,
        max_new_tokens=MAX_TOKENS,
        temperature=0.0,
        verbose=False,
        use_cuda_graph=False,
    )
    duration = time.time() - t0
    print(f"[PyTorch] Duration: {duration:.2f}s ({MAX_TOKENS/duration:.1f} tok/s)")
    print(f"[PyTorch] Output:\n{torch_text}")

    # Write output to file for comparison
    with open("/tmp/torch_output.txt", "w") as f:
        f.write(torch_text)

    return torch_text


def compare_outputs():
    """Compare outputs from files."""
    from seqcond.dataset import Tokenizer

    with open("/tmp/jax_output.txt", "r") as f:
        jax_text = f.read()
    with open("/tmp/torch_output.txt", "r") as f:
        torch_text = f.read()

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"\nJAX Output:\n{jax_text}")
    print(f"\nPyTorch Output:\n{torch_text}")

    if jax_text == torch_text:
        print("\n✅ SUCCESS: JAX and PyTorch produced IDENTICAL outputs!")
        return True
    else:
        print("\n❌ FAILURE: Outputs differ!")
        tokenizer = Tokenizer()
        tokens_jax = tokenizer([jax_text])[0]
        tokens_torch = tokenizer([torch_text])[0]

        print(f"\nJAX tokens:    {len(tokens_jax)}")
        print(f"PyTorch tokens: {len(tokens_torch)}")

        # Find first difference
        min_len = min(len(tokens_jax), len(tokens_torch))
        first_diff = None
        for i in range(min_len):
            if tokens_jax[i] != tokens_torch[i]:
                first_diff = i
                break

        if first_diff is not None:
            print(f"\nFirst difference at position {first_diff}:")
            start = max(0, first_diff - 2)
            end = min(min_len, first_diff + 5)
            for i in range(start, end):
                char_jax = tokenizer.decode([tokens_jax[i]])
                char_torch = tokenizer.decode([tokens_torch[i]])
                status = "✓" if tokens_jax[i] == tokens_torch[i] else "← DIFF"
                print(
                    f"  Pos {i:3d}: JAX={tokens_jax[i]:5d} ({char_jax!r:10s}) | Torch={tokens_torch[i]:5d} ({char_torch!r:10s}) {status}"
                )

        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "jax":
            run_jax()
        elif sys.argv[1] == "torch":
            run_torch()
        elif sys.argv[1] == "compare":
            compare_outputs()
    else:
        print("=" * 60)
        print("Testing Equivalence: JAX (Ground Truth) vs PyTorch")
        print("Temperature: 0 (Greedy/Argmax)")
        print("=" * 60)

        # Run JAX in subprocess
        print("\n[1/2] Running JAX generation in subprocess...")
        result = subprocess.run([sys.executable, __file__, "jax"], check=True)

        # Run PyTorch in subprocess
        print("\n[2/2] Running PyTorch generation in subprocess...")
        result = subprocess.run([sys.executable, __file__, "torch"], check=True)

        # Compare
        compare_outputs()

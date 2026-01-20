"""
Test equivalence between stepwise decoding and full forward pass decoding.
Checks that both methods produce the exact same tokens when using argmax.
"""

import pickle
import time
import jax
import jax.numpy as jnp
import numpy as np
from seqcond.jax.generator import Generator, generate_text_stepwise
from seqcond.jax.model import SeqCondModel
from seqcond.dataset import Tokenizer
from seqcond.config import ModelConfig
from seqcond.jax.callback import generate_text as generate_text_full

CHECKPOINT_PATH = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step1.pkl"


def test_equivalence():
    print("=" * 60)
    print("Testing Equivalence: Stepwise vs Full Forward (Argmax)")
    print("=" * 60)

    prompt = "The quick brown fox jumps over"
    max_new_tokens = 20

    # 1. Generate with Full Forward (the old way)
    print(f"\nPrompt: {prompt}")
    print("\nRunning Full Forward generation (recomputing sequence at each token)...")

    # Load checkpoint for full forward
    with open(CHECKPOINT_PATH, "rb") as f:
        data = pickle.load(f)
    params = data["params"]
    config_dict = data["config"]
    model_config = ModelConfig(**config_dict["model"])

    tokenizer = Tokenizer()
    model = SeqCondModel(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        num_kv_heads=model_config.num_kv_heads,
        d_ff=model_config.d_ff,
        maxlen=model_config.maxlen,
        dropout=0.0,
        tie_weights=model_config.tie_weights,
        qk_norm=model_config.qk_norm,
        seqcond_heads=model_config.seqcond_heads,
        num_query_heads=model_config.num_query_heads,
        num_thetas=model_config.num_thetas,
        num_anchor_heads=model_config.num_anchor_heads,
        conv_kernel_size=model_config.conv_kernel_size,
        expand_factor=model_config.expand_factor,
        out_expand_factor=model_config.out_expand_factor,
        seqcond_ratio=model_config.seqcond_ratio,
        use_square_matrix=model_config.use_square_matrix,
        remat=False,
    )

    # Use temperature=0.0 or very low for deterministic argmax-like behavior
    # callback.generate_text doesn't have a direct argmax mode, but we can wrap it or use temp 0
    t0_full = time.time()
    text_full = generate_text_full(
        model=model,
        params=params,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=1e-6,  # effectively argmax
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.0,
        maxlen=model_config.maxlen,
        verbose=False,
    )
    duration_full = time.time() - t0_full
    print(f"Full Forward Output: {text_full}")
    print(
        f"Full Forward Duration: {duration_full:.2f}s ({duration_full/max_new_tokens*1000:.1f}ms/token)"
    )

    # 2. Generate with Stepwise (the new way)
    print("\nRunning Stepwise generation (O(1) decoding)...")
    gen = Generator(CHECKPOINT_PATH)

    # Warmup for stepwise JIT
    print("Warmup stepwise JIT...")
    _ = gen.generate(prompt="Warmup", max_new_tokens=5, verbose=False)

    t0_step = time.time()
    text_step = gen.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=1e-6,  # effectively argmax
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.0,
        verbose=False,
    )
    duration_step = time.time() - t0_step
    print(f"Stepwise Output: {text_step}")
    print(
        f"Stepwise Duration: {duration_step:.2f}s ({duration_step/max_new_tokens*1000:.1f}ms/token)"
    )

    # 3. Generate with Stepwise + lax.scan (the optimized way)
    print("\nRunning Stepwise generation with lax.scan (Optimized)...")
    # Warmup for scan JIT with SAME number of tokens to avoid recompilation
    print(f"Warmup scan JIT (with {max_new_tokens} tokens)...")
    _ = gen.generate(
        prompt="Warmup", max_new_tokens=max_new_tokens, use_scan=True, verbose=False
    )

    t0_scan = time.time()
    text_scan = gen.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        use_scan=True,
        verbose=False,
    )
    duration_scan = time.time() - t0_scan
    print(f"Scan Output: {text_scan}")
    print(
        f"Scan Duration: {duration_scan:.2f}s ({duration_scan/max_new_tokens*1000:.1f}ms/token)"
    )

    # 4. Compare everything
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(
        f"Full Forward: {duration_full:.2f}s ({duration_full/max_new_tokens*1000:.1f}ms/token)"
    )
    print(
        f"Stepwise:     {duration_step:.2f}s ({duration_step/max_new_tokens*1000:.1f}ms/token)"
    )
    print(
        f"Scan (Opt):   {duration_scan:.2f}s ({duration_scan/max_new_tokens*1000:.1f}ms/token)"
    )
    print("-" * 30)
    print(f"Stepwise Speedup vs Full: {duration_full / duration_step:.2f}x")
    print(f"Scan Speedup vs Stepwise: {duration_step / duration_scan:.2f}x")
    print(f"Total Speedup vs Full:    {duration_full / duration_scan:.2f}x")
    print("-" * 30)

    all_match = text_full == text_step == text_scan
    if all_match:
        print("\n" + "✓" * 60)
        print("SUCCESS: All methods produced identical text!")
        print("✓" * 60)
        return True
    else:
        print("\n" + "✗" * 60)
        print("FAILURE: Outputs differ!")
        print(f"Full == Step: {text_full == text_step}")
        print(f"Step == Scan: {text_step == text_scan}")
        print("✗" * 60)
        return False


if __name__ == "__main__":
    test_equivalence()

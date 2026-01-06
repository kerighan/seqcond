"""
Optimized speed benchmark using jax.jit and scan to avoid recompilations.
"""

import time
import jax
import jax.numpy as jnp
from functools import partial

from seqcond.jax.seqcond_summary import SeqCondAttention
from seqcond.jax.rope import (
    TransformerDecoderBlock,
    precompute_freqs,
    get_rope_embeddings,
)


def benchmark_seqcond_optimized():
    """Benchmark SeqCond with JIT-compiled step function."""
    print("\n" + "=" * 70)
    print("SeqCond O(1) Benchmark (Optimized with JIT)")
    print("=" * 70)

    batch_size = 1
    d_model = 256
    num_heads = 8
    maxlen = 1024
    seq_lengths = [64, 128, 256, 512, 1024]

    model = SeqCondAttention(
        num_heads=num_heads,
        num_thetas=1,
        expand_factor=1.0,
        out_expand_factor=2,
        maxlen=maxlen,
        use_square_matrix=False,
        compute_dtype=jnp.float32,
        param_dtype=jnp.float32,
    )

    # Initialize
    key = jax.random.PRNGKey(42)
    x_init = jax.random.normal(key, (batch_size, 64, d_model))
    variables = model.init(jax.random.PRNGKey(0), x_init, deterministic=True)
    params = variables["params"]

    # JIT-compile step function
    @jax.jit
    def step_fn(x_t, state):
        return model.apply(
            {"params": params}, x_t, state, deterministic=True, method=model.step
        )

    # JIT-compile scan over steps
    @jax.jit
    def scan_steps(x_seq, init_state):
        def scan_fn(state, x_t):
            out, new_state = step_fn(x_t, state)
            return new_state, out

        final_state, outputs = jax.lax.scan(scan_fn, init_state, x_seq)
        return outputs, final_state

    print(f"{'Seq Len':<10} {'Full (ms)':<12} {'Step (ms)':<12} {'Speedup':<10}")
    print("-" * 60)

    for seq_len in seq_lengths:
        x = jax.random.normal(key, (batch_size, seq_len, d_model))

        # Full forward
        full_fn = jax.jit(
            lambda x: model.apply({"params": params}, x, deterministic=True)
        )
        _ = full_fn(x)  # Warmup

        start = time.time()
        for _ in range(10):
            _ = full_fn(x)
        jax.block_until_ready(_)
        time_full = (time.time() - start) / 10 * 1000

        # Step-by-step with scan
        H = max(1, int(d_model * 1.0) // (num_heads * 1))
        conv_kernel_size = 4
        dim_memory = num_heads * H

        init_state = (
            jnp.zeros((batch_size, num_heads), dtype=jnp.float32),
            jnp.zeros((batch_size, num_heads, H, 1), dtype=jnp.float32),
            jnp.zeros((batch_size, num_heads, H, 1), dtype=jnp.float32),
            jnp.zeros((batch_size,), dtype=jnp.int32),
            jnp.zeros(
                (batch_size, conv_kernel_size - 1, dim_memory + num_heads),
                dtype=jnp.float32,
            ),
        )

        # Transpose to (seq_len, batch, d_model) for scan
        x_transposed = jnp.transpose(x, (1, 0, 2))

        _, _ = scan_steps(x_transposed, init_state)  # Warmup

        start = time.time()
        for _ in range(10):
            _, _ = scan_steps(x_transposed, init_state)
        jax.block_until_ready(_)
        time_step = (time.time() - start) / 10 * 1000

        speedup = time_full / time_step if time_step > 0 else float("inf")
        print(f"{seq_len:<10} {time_full:<12.2f} {time_step:<12.2f} {speedup:<10.2f}x")


def benchmark_transformer_optimized():
    """Benchmark Transformer with JIT-compiled step function."""
    print("\n" + "=" * 70)
    print("Transformer O(L) Benchmark (Optimized with JIT)")
    print("=" * 70)

    batch_size = 1
    d_model = 256
    num_heads = 8
    num_kv_heads = 4
    d_ff = 512
    maxlen = 1024
    seq_lengths = [64, 128, 256, 512, 1024]

    model = TransformerDecoderBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_kv_heads=num_kv_heads,
        dropout=0.0,
        qk_norm=False,
    )

    cos_emb, sin_emb = precompute_freqs(maxlen, d_model // num_heads)

    # Initialize
    key = jax.random.PRNGKey(42)
    x_init = jax.random.normal(key, (batch_size, 64, d_model))
    cos_init, sin_init = get_rope_embeddings(
        64, cos_emb, sin_emb, batch_size, num_heads
    )
    variables = model.init(
        jax.random.PRNGKey(0), x_init, cos=cos_init, sin=sin_init, deterministic=True
    )
    params = variables["params"]

    print(f"{'Seq Len':<10} {'Full (ms)':<12} {'Step (ms)':<12} {'Speedup':<10}")
    print("-" * 60)

    for seq_len in seq_lengths:
        x = jax.random.normal(key, (batch_size, seq_len, d_model))
        cos, sin = get_rope_embeddings(seq_len, cos_emb, sin_emb, batch_size, num_heads)

        # Full forward
        full_fn = jax.jit(
            lambda x, cos, sin: model.apply(
                {"params": params}, x, cos=cos, sin=sin, deterministic=True
            )
        )
        _ = full_fn(x, cos, sin)  # Warmup

        start = time.time()
        for _ in range(10):
            _ = full_fn(x, cos, sin)
        jax.block_until_ready(_)
        time_full = (time.time() - start) / 10 * 1000

        # Step-by-step with manual loop (KV cache grows, can't use scan easily)
        head_dim = d_model // num_heads

        @jax.jit
        def step_fn(x_t, kv_cache, t, cos_t, sin_t):
            return model.apply(
                {"params": params},
                x_t,
                kv_cache,
                t,
                cos_t,
                sin_t,
                deterministic=True,
                method=model.step,
            )

        # Warmup
        kv_cache = (
            jnp.zeros((batch_size, 0, num_kv_heads, head_dim), dtype=x.dtype),
            jnp.zeros((batch_size, 0, num_kv_heads, head_dim), dtype=x.dtype),
        )
        for t in range(min(10, seq_len)):
            x_t = x[:, t, :]
            cos_t = cos[:, t : t + 1, :, :]
            sin_t = sin[:, t : t + 1, :, :]
            _, kv_cache = step_fn(x_t, kv_cache, t, cos_t, sin_t)

        # Actual timing
        kv_cache = (
            jnp.zeros((batch_size, 0, num_kv_heads, head_dim), dtype=x.dtype),
            jnp.zeros((batch_size, 0, num_kv_heads, head_dim), dtype=x.dtype),
        )

        start = time.time()
        for _ in range(10):
            kv_tmp = kv_cache
            for t in range(seq_len):
                x_t = x[:, t, :]
                cos_t = cos[:, t : t + 1, :, :]
                sin_t = sin[:, t : t + 1, :, :]
                _, kv_tmp = step_fn(x_t, kv_tmp, t, cos_t, sin_t)
        jax.block_until_ready(kv_tmp)
        time_step = (time.time() - start) / 10 * 1000

        speedup = time_full / time_step if time_step > 0 else float("inf")
        print(f"{seq_len:<10} {time_full:<12.2f} {time_step:<12.2f} {speedup:<10.2f}x")


if __name__ == "__main__":
    print("=" * 70)
    print("Optimized Speed Benchmarks (with JIT compilation)")
    print("=" * 70)

    benchmark_seqcond_optimized()
    benchmark_transformer_optimized()

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("=" * 70)
    print("- SeqCond O(1): Constant time per step, ideal for autoregressive generation")
    print("- Transformer O(L): Linear growth with sequence length")
    print("- JIT compilation is critical for performance")

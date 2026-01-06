"""
Quick benchmark: Verify O(1) SeqCond is faster than O(L) Transformer.
Focus on per-token generation time.
"""

import time
import jax
import jax.numpy as jnp

from seqcond.jax.seqcond_summary import SeqCondAttention
from seqcond.jax.rope import (
    TransformerDecoderBlock,
    precompute_freqs,
    get_rope_embeddings,
)


print("=" * 70)
print("Quick Benchmark: O(1) vs O(L) Per-Token Generation")
print("=" * 70)

batch_size = 1
d_model = 256
num_heads = 8
num_kv_heads = 4
d_ff = 512
maxlen = 1024
key = jax.random.PRNGKey(42)

# ============================================================
# SeqCond O(1)
# ============================================================
print("\n### SeqCond O(1) ###")
seqcond = SeqCondAttention(
    num_heads=num_heads,
    num_thetas=1,
    expand_factor=1.0,
    out_expand_factor=2,
    maxlen=maxlen,
    use_square_matrix=False,
    compute_dtype=jnp.float32,
    param_dtype=jnp.float32,
)

x_init = jax.random.normal(key, (batch_size, 64, d_model))
variables = seqcond.init(jax.random.PRNGKey(0), x_init, deterministic=True)
params_sc = variables["params"]

H = max(1, int(d_model * 1.0) // (num_heads * 1))
conv_kernel_size = 4
dim_memory = num_heads * H

init_state_sc = (
    jnp.zeros((batch_size, num_heads), dtype=jnp.float32),
    jnp.zeros((batch_size, num_heads, H, 1), dtype=jnp.float32),
    jnp.zeros((batch_size, num_heads, H, 1), dtype=jnp.float32),
    jnp.zeros((batch_size,), dtype=jnp.int32),
    jnp.zeros(
        (batch_size, conv_kernel_size - 1, dim_memory + num_heads), dtype=jnp.float32
    ),
)


@jax.jit
def seqcond_step(x_t, state):
    return seqcond.apply(
        {"params": params_sc}, x_t, state, deterministic=True, method=seqcond.step
    )


# Warmup
x_t = jax.random.normal(key, (batch_size, d_model))
_, state_sc = seqcond_step(x_t, init_state_sc)
print("✓ Compiled")

# Measure at different sequence positions
print("\nPer-token time at different positions:")
print(f"{'Position':<12} {'Time (ms)':<12}")
print("-" * 30)

for pos in [10, 50, 100, 200, 500]:
    # Generate to position
    state_sc = init_state_sc
    for _ in range(pos):
        x_t = jax.random.normal(key, (batch_size, d_model))
        _, state_sc = seqcond_step(x_t, state_sc)

    # Time next token
    x_t = jax.random.normal(key, (batch_size, d_model))
    start = time.time()
    for _ in range(50):
        _, state_sc = seqcond_step(x_t, state_sc)
    jax.block_until_ready(state_sc)
    elapsed = (time.time() - start) / 50 * 1000
    print(f"{pos:<12} {elapsed:<12.4f}")

# ============================================================
# Transformer O(L)
# ============================================================
print("\n### Transformer O(L) ###")
transformer = TransformerDecoderBlock(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_kv_heads=num_kv_heads,
    dropout=0.0,
    qk_norm=False,
)

cos_emb, sin_emb = precompute_freqs(maxlen, d_model // num_heads)
cos_init, sin_init = get_rope_embeddings(64, cos_emb, sin_emb, batch_size, num_heads)
variables = transformer.init(
    jax.random.PRNGKey(0), x_init, cos=cos_init, sin=sin_init, deterministic=True
)
params_tf = variables["params"]

head_dim = d_model // num_heads
init_kv = (
    jnp.zeros((batch_size, 0, num_kv_heads, head_dim), dtype=jnp.float32),
    jnp.zeros((batch_size, 0, num_kv_heads, head_dim), dtype=jnp.float32),
)

print("✓ Initialized")

# Measure at different sequence positions
print("\nPer-token time at different positions:")
print(f"{'Position':<12} {'Time (ms)':<12}")
print("-" * 30)

cos_full, sin_full = get_rope_embeddings(1024, cos_emb, sin_emb, batch_size, num_heads)

for pos in [10, 50, 100, 200, 500]:
    # Build KV cache up to position
    kv_cache = init_kv
    for t in range(pos):
        x_t = jax.random.normal(key, (batch_size, d_model))
        cos_t = cos_full[:, t : t + 1, :, :]
        sin_t = sin_full[:, t : t + 1, :, :]

        # Create JIT function for this specific cache size
        @jax.jit
        def tf_step(x_t, kv, t, cos_t, sin_t):
            return transformer.apply(
                {"params": params_tf},
                x_t,
                kv,
                t,
                cos_t,
                sin_t,
                deterministic=True,
                method=transformer.step,
            )

        _, kv_cache = tf_step(x_t, kv_cache, t, cos_t, sin_t)

    # Time next token (will recompile for new cache size, but that's the reality)
    x_t = jax.random.normal(key, (batch_size, d_model))
    cos_t = cos_full[:, pos : pos + 1, :, :]
    sin_t = sin_full[:, pos : pos + 1, :, :]

    @jax.jit
    def tf_step_timed(x_t, kv, t, cos_t, sin_t):
        return transformer.apply(
            {"params": params_tf},
            x_t,
            kv,
            t,
            cos_t,
            sin_t,
            deterministic=True,
            method=transformer.step,
        )

    # Warmup for this size
    _, kv_tmp = tf_step_timed(x_t, kv_cache, pos, cos_t, sin_t)

    start = time.time()
    for _ in range(50):
        _, kv_tmp = tf_step_timed(x_t, kv_cache, pos, cos_t, sin_t)
    jax.block_until_ready(kv_tmp)
    elapsed = (time.time() - start) / 50 * 1000
    print(f"{pos:<12} {elapsed:<12.4f}")

print("\n" + "=" * 70)
print("Analysis:")
print("=" * 70)
print("- SeqCond O(1): Time should be CONSTANT regardless of position")
print("- Transformer O(L): Time should GROW LINEARLY with position")
print("- At long sequences, SeqCond should be much faster per token")

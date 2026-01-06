"""
Simple benchmark - compile once, run many times.
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
print("Simple Speed Test - Compile Once, Run Many Times")
print("=" * 70)

# Small model
batch_size = 1
d_model = 256
num_heads = 8
maxlen = 1024

# SeqCond
print("\n### SeqCond Test ###")
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

key = jax.random.PRNGKey(42)
x_init = jax.random.normal(key, (batch_size, 64, d_model))
variables = model.init(jax.random.PRNGKey(0), x_init, deterministic=True)
params = variables["params"]

# Compile step function ONCE
print("Compiling step function...")
H = max(1, int(d_model * 1.0) // (num_heads * 1))
conv_kernel_size = 4
dim_memory = num_heads * H

init_state = (
    jnp.zeros((batch_size, num_heads), dtype=jnp.float32),
    jnp.zeros((batch_size, num_heads, H, 1), dtype=jnp.float32),
    jnp.zeros((batch_size, num_heads, H, 1), dtype=jnp.float32),
    jnp.zeros((batch_size,), dtype=jnp.int32),
    jnp.zeros(
        (batch_size, conv_kernel_size - 1, dim_memory + num_heads), dtype=jnp.float32
    ),
)


@jax.jit
def step_fn(x_t, state):
    return model.apply(
        {"params": params}, x_t, state, deterministic=True, method=model.step
    )


# Warmup
x_t = jax.random.normal(key, (batch_size, d_model))
_, state = step_fn(x_t, init_state)
print("✓ Compiled")

# Time single step
print("\nTiming single step (100 iterations)...")
start = time.time()
state = init_state
for _ in range(100):
    _, state = step_fn(x_t, state)
jax.block_until_ready(state)
elapsed = (time.time() - start) * 1000
print(f"Total: {elapsed:.2f} ms")
print(f"Per step: {elapsed/100:.3f} ms")

# Time full sequence
print("\nTiming 512-token generation...")
state = init_state
start = time.time()
for t in range(512):
    x_t = jax.random.normal(key, (batch_size, d_model))
    _, state = step_fn(x_t, state)
jax.block_until_ready(state)
elapsed = (time.time() - start) * 1000
print(f"Total: {elapsed:.2f} ms")
print(f"Per token: {elapsed/512:.3f} ms")

# Compare with full forward
print("\nComparing with full forward pass...")
x_full = jax.random.normal(key, (batch_size, 512, d_model))


@jax.jit
def full_fn(x):
    return model.apply({"params": params}, x, deterministic=True)


_ = full_fn(x_full)  # Warmup

start = time.time()
for _ in range(10):
    _ = full_fn(x_full)
jax.block_until_ready(_)
time_full = (time.time() - start) / 10 * 1000
print(f"Full forward (512 tokens): {time_full:.2f} ms")
print(f"Step-by-step (512 tokens): {elapsed:.2f} ms")
print(f"Ratio: {elapsed/time_full:.2f}x")

# Transformer
print("\n### Transformer Test ###")
num_kv_heads = 4
d_ff = 512

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
params_t = variables["params"]

print("Compiling step function...")
head_dim = d_model // num_heads
init_kv = (
    jnp.zeros((batch_size, 0, num_kv_heads, head_dim), dtype=jnp.float32),
    jnp.zeros((batch_size, 0, num_kv_heads, head_dim), dtype=jnp.float32),
)

# Note: Transformer step changes shape (KV cache grows), so it WILL recompile
# But we can still measure
cos_full, sin_full = get_rope_embeddings(512, cos_emb, sin_emb, batch_size, num_heads)

print("\nTiming 512-token generation (with recompilation)...")
kv_cache = init_kv
start = time.time()
for t in range(512):
    x_t = jax.random.normal(key, (batch_size, d_model))
    cos_t = cos_full[:, t : t + 1, :, :]
    sin_t = sin_full[:, t : t + 1, :, :]

    @jax.jit
    def step_t_fn(x_t, kv, t, cos_t, sin_t):
        return transformer.apply(
            {"params": params_t},
            x_t,
            kv,
            t,
            cos_t,
            sin_t,
            deterministic=True,
            method=transformer.step,
        )

    _, kv_cache = step_t_fn(x_t, kv_cache, t, cos_t, sin_t)
jax.block_until_ready(kv_cache)
elapsed_tf = (time.time() - start) * 1000
print(f"Total: {elapsed_tf:.2f} ms (includes recompilation overhead)")
print(f"Per token: {elapsed_tf/512:.3f} ms")

# Full forward
x_full = jax.random.normal(key, (batch_size, 512, d_model))


@jax.jit
def full_tf_fn(x, cos, sin):
    return transformer.apply(
        {"params": params_t}, x, cos=cos, sin=sin, deterministic=True
    )


_ = full_tf_fn(x_full, cos_full, sin_full)

start = time.time()
for _ in range(10):
    _ = full_tf_fn(x_full, cos_full, sin_full)
jax.block_until_ready(_)
time_full_tf = (time.time() - start) / 10 * 1000
print(f"Full forward (512 tokens): {time_full_tf:.2f} ms")
print(f"Ratio: {elapsed_tf/time_full_tf:.2f}x (note: includes recompilation)")

print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)
print("- SeqCond O(1): No recompilation, should be fast")
print("- Transformer O(L): Recompiles each step (KV cache size changes)")
print("- For real generation, use pre-allocated KV cache with max size")

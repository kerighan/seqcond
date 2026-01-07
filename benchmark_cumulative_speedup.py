"""
Cumulative speedup benchmark: SeqCond vs Transformer during generation.
Shows speedup ratio at each token position (cumulative time ratio).
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


def next_power_of_2(n):
    """Get next power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


print("=" * 70)
print("CUMULATIVE SPEEDUP: SeqCond vs Transformer")
print("Measuring speedup ratio as generation progresses")
print("=" * 70)

batch_size = 1
d_model = 256
num_heads = 8
num_kv_heads = 4
d_ff = 512
maxlen = 2048 * 2
max_tokens = 2048 * 2
key = jax.random.PRNGKey(42)

# Sample points for measurement
sample_points = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# ============================================================
# Initialize SeqCond
# ============================================================
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
_, _ = seqcond_step(x_t, init_state_sc)

# ============================================================
# Initialize Transformer
# ============================================================
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

cache_size = next_power_of_2(max_tokens)
kv_cache_init = (
    jnp.zeros((batch_size, cache_size, num_kv_heads, head_dim), dtype=jnp.float32),
    jnp.zeros((batch_size, cache_size, num_kv_heads, head_dim), dtype=jnp.float32),
)

cos_cache, sin_cache = get_rope_embeddings(
    cache_size, cos_emb, sin_emb, batch_size, num_heads
)


@jax.jit
def transformer_step(x_t, kv_cache, pos, cos_t, sin_t):
    return transformer.apply(
        {"params": params_tf},
        x_t,
        kv_cache,
        pos,
        cos_t,
        sin_t,
        deterministic=True,
        method=transformer.step,
    )


# Warmup
x_t = jax.random.normal(key, (batch_size, d_model))
cos_t = cos_cache[:, 0:1, :, :]
sin_t = sin_cache[:, 0:1, :, :]
_, _ = transformer_step(x_t, kv_cache_init, 0, cos_t, sin_t)

# ============================================================
# Generate and measure cumulative times
# ============================================================
print("\nGenerating tokens and measuring cumulative times...")

# SeqCond generation
print("\n[1/2] SeqCond generation...")
state_sc = init_state_sc
seqcond_times = {}

start_total = time.time()
for t in range(max_tokens):
    x_t = jax.random.normal(key, (batch_size, d_model))

    start = time.time()
    _, state_sc = seqcond_step(x_t, state_sc)
    jax.block_until_ready(state_sc)
    elapsed = time.time() - start

    # Record cumulative time at sample points
    if (t + 1) in sample_points:
        cumulative_time = time.time() - start_total
        seqcond_times[t + 1] = cumulative_time * 1000  # ms
        print(f"  Token {t+1:4d}: {cumulative_time*1000:8.2f} ms cumulative")

# Transformer generation
print("\n[2/2] Transformer generation...")
kv_cache = kv_cache_init
transformer_times = {}

start_total = time.time()
for t in range(max_tokens):
    x_t = jax.random.normal(key, (batch_size, d_model))
    cos_t = cos_cache[:, t : t + 1, :, :]
    sin_t = sin_cache[:, t : t + 1, :, :]

    start = time.time()
    _, kv_cache = transformer_step(x_t, kv_cache, t, cos_t, sin_t)
    jax.block_until_ready(kv_cache)
    elapsed = time.time() - start

    # Record cumulative time at sample points
    if (t + 1) in sample_points:
        cumulative_time = time.time() - start_total
        transformer_times[t + 1] = cumulative_time * 1000  # ms
        print(f"  Token {t+1:4d}: {cumulative_time*1000:8.2f} ms cumulative")

# ============================================================
# Calculate and display speedup ratios
# ============================================================
print("\n" + "=" * 70)
print("CUMULATIVE SPEEDUP RATIO (Transformer time / SeqCond time)")
print("=" * 70)
print(f"\n{'Tokens':<10} {'SeqCond (ms)':<15} {'Transformer (ms)':<18} {'Speedup':<12}")
print("-" * 70)

speedup_data = []
for n_tokens in sample_points:
    sc_time = seqcond_times[n_tokens]
    tf_time = transformer_times[n_tokens]
    speedup = tf_time / sc_time
    speedup_data.append((n_tokens, sc_time, tf_time, speedup))
    print(f"{n_tokens:<10} {sc_time:<15.2f} {tf_time:<18.2f} {speedup:<12.2f}x")

# ============================================================
# Save results to file for plotting
# ============================================================
output_file = "cumulative_speedup_data.txt"
with open(output_file, "w") as f:
    f.write("# Cumulative Speedup: SeqCond vs Transformer\n")
    f.write("# Columns: Tokens, SeqCond_ms, Transformer_ms, Speedup\n")
    for n_tokens, sc_time, tf_time, speedup in speedup_data:
        f.write(f"{n_tokens}\t{sc_time:.2f}\t{tf_time:.2f}\t{speedup:.2f}\n")

print(f"\n✓ Data saved to {output_file}")

# ============================================================
# Analysis
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

# Find where SeqCond becomes faster
for i, (n_tokens, sc_time, tf_time, speedup) in enumerate(speedup_data):
    if speedup > 1.0:
        print(f"\n✓ SeqCond becomes faster than Transformer at {n_tokens} tokens")
        print(f"  SeqCond: {sc_time:.2f} ms")
        print(f"  Transformer: {tf_time:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        break

# Final speedup
final_tokens, final_sc, final_tf, final_speedup = speedup_data[-1]
print(f"\n✓ At {final_tokens} tokens:")
print(f"  SeqCond: {final_sc:.2f} ms ({final_sc/final_tokens:.3f} ms/token)")
print(f"  Transformer: {final_tf:.2f} ms ({final_tf/final_tokens:.3f} ms/token)")
print(f"  Final speedup: {final_speedup:.2f}x")

print("\n" + "=" * 70)
print("To plot this data:")
print("  import matplotlib.pyplot as plt")
print("  import numpy as np")
print("  data = np.loadtxt('cumulative_speedup_data.txt')")
print("  plt.plot(data[:, 0], data[:, 3])")
print("  plt.xlabel('Number of tokens')")
print("  plt.ylabel('Speedup (Transformer / SeqCond)')")
print("  plt.title('Cumulative Speedup: SeqCond vs Transformer')")
print("  plt.grid(True)")
print("  plt.show()")
print("=" * 70)

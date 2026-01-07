"""
Debug script to identify the exact issue with step method.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np

from seqcond.jax.seqcond_light import SeqCondAttention


def main():
    print("Debugging step method...")

    # Simple setup
    B = 1
    L = 3
    D = 64
    num_heads = 4
    num_thetas = 1
    conv_kernel_size = 4

    model = SeqCondAttention(
        num_heads=num_heads,
        num_thetas=num_thetas,
        conv_kernel_size=conv_kernel_size,
        expand_factor=1.0,
        compute_dtype=jnp.float32,
        param_dtype=jnp.float32,
    )

    # Initialize
    key = jax.random.PRNGKey(42)
    x_full = jax.random.normal(key, (B, L, D))

    print(f"\nInput shape: {x_full.shape}")
    print(f"First token: {x_full[0, 0, :5]}")

    # Initialize model
    key, init_key = jax.random.split(key)
    variables = model.init(init_key, x_full, deterministic=True)

    # Print conv kernel shape
    print(f"\nConv kernel shape: {variables['params']['conv_mem']['kernel'].shape}")
    print(
        f"Conv kernel first values: {variables['params']['conv_mem']['kernel'][0, 0, :5]}"
    )
    print(f"in_proj kernel shape: {variables['params']['in_proj']['kernel'].shape}")
    print(f"\nRMSNorm params:")
    print(f"k_norm scale shape: {variables['params']['k_norm']['scale'].shape}")
    print(f"out_proj kernel shape: {variables['params']['out_proj']['kernel'].shape}")

    # Dimensions
    d_inner = int(D * model.expand_factor)
    K = model.num_heads
    M = model.num_thetas
    H = max(1, d_inner // (K * M))

    print(f"\nDimensions: K={K}, M={M}, H={H}")

    # Test 1: Forward pass on single token
    print("\n" + "=" * 60)
    print("Test 1: Forward pass on first token only")
    print("=" * 60)
    x_1 = x_full[:, :1, :]
    out_1 = model.apply(variables, x_1, deterministic=True)
    print(f"Output shape: {out_1.shape}")
    print(f"Output: {out_1[0, 0, :5]}")

    # Test 2: Step on first token
    print("\n" + "=" * 60)
    print("Test 2: Step on first token")
    print("=" * 60)

    # Initialize state
    den_acc = jnp.zeros((B, K), dtype=jnp.float32)
    re_acc = jnp.zeros((B, K, H, M), dtype=jnp.float32)
    im_acc = jnp.zeros((B, K, H, M), dtype=jnp.float32)
    pos = jnp.zeros((B,), dtype=jnp.int32)

    dim_memory = K * H
    conv_buffer = jnp.zeros(
        (B, conv_kernel_size - 1, dim_memory + K), dtype=jnp.float32
    )

    print(f"Conv buffer shape: {conv_buffer.shape}")
    print(f"Conv buffer (should be all zeros): {jnp.max(jnp.abs(conv_buffer))}")

    state = (den_acc, re_acc, im_acc, pos, conv_buffer)

    x_t = x_full[:, 0, :]
    print(f"Input token shape: {x_t.shape}")
    out_t, state = model.apply(
        variables, x_t, state, deterministic=True, method=model.step
    )
    print(f"Output shape: {out_t.shape}")
    print(f"Output: {out_t[0, :5]}")

    # Compare
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    diff = jnp.abs(out_1[0, 0] - out_t[0])
    print(f"Max diff: {jnp.max(diff):.2e}")
    print(f"Mean diff: {jnp.mean(diff):.2e}")

    if jnp.max(diff) < 1e-4:
        print("✅ First token matches!")
    else:
        print("❌ First token differs!")
        print(f"\nForward: {out_1[0, 0, :10]}")
        print(f"Step:    {out_t[0, :10]}")

    # Test 3: Second token
    print("\n" + "=" * 60)
    print("Test 3: Second token")
    print("=" * 60)

    # Forward on 2 tokens
    x_2 = x_full[:, :2, :]
    out_2 = model.apply(variables, x_2, deterministic=True)
    print(f"Forward output (token 2): {out_2[0, 1, :5]}")

    # Step on second token
    x_t2 = x_full[:, 1, :]
    out_t2, state = model.apply(
        variables, x_t2, state, deterministic=True, method=model.step
    )
    print(f"Step output (token 2): {out_t2[0, :5]}")

    diff2 = jnp.abs(out_2[0, 1] - out_t2[0])
    print(f"\nMax diff: {jnp.max(diff2):.2e}")
    print(f"Mean diff: {jnp.mean(diff2):.2e}")

    if jnp.max(diff2) < 1e-4:
        print("✅ Second token matches!")
    else:
        print("❌ Second token differs!")


if __name__ == "__main__":
    main()

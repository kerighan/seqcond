"""
Test to verify the exact dtype behavior in the convolution operation.
"""

import jax.numpy as jnp

# Simulate the convolution operation
B, K, C = 2, 4, 100

# float32 buffer (as initialized)
conv_buffer_f32 = jnp.zeros((B, K - 1, C), dtype=jnp.float32)

# bfloat16 new input (as computed)
z_conv_bf16 = jnp.ones((B, C), dtype=jnp.bfloat16)
z_conv_expanded = z_conv_bf16[:, None, :]

# Test 1: Original (no casting)
conv_input_1 = jnp.concatenate([conv_buffer_f32, z_conv_expanded], axis=1)
print(f"Test 1 - No casting:")
print(f"  conv_input dtype: {conv_input_1.dtype}")

# Test 2: Cast buffer before concat
conv_input_2 = jnp.concatenate(
    [conv_buffer_f32.astype(jnp.bfloat16), z_conv_expanded], axis=1
)
print(f"\nTest 2 - Cast buffer before concat:")
print(f"  conv_input dtype: {conv_input_2.dtype}")

# Test 3: Einsum with float32 kernel
kernel_f32 = jnp.ones((K, C), dtype=jnp.float32)

result_1 = jnp.einsum("bkc,kc->bc", conv_input_1, kernel_f32)
print(f"\nEinsum with float32 kernel:")
print(f"  Input 1 (float32) -> result dtype: {result_1.dtype}")

result_2 = jnp.einsum("bkc,kc->bc", conv_input_2, kernel_f32)
print(f"  Input 2 (bfloat16) -> result dtype: {result_2.dtype}")

# Test 4: What about buffer update?
conv_buffer_new_1 = jnp.concatenate(
    [conv_buffer_f32[:, 1:, :], z_conv_expanded], axis=1
)
print(f"\nBuffer update without casting:")
print(f"  conv_buffer_new dtype: {conv_buffer_new_1.dtype}")

conv_buffer_new_2 = jnp.concatenate(
    [conv_buffer_f32[:, 1:, :].astype(jnp.bfloat16), z_conv_expanded], axis=1
)
print(f"\nBuffer update with casting:")
print(f"  conv_buffer_new dtype: {conv_buffer_new_2.dtype}")

# Test 5: Check if the issue is that we need to cast the result back
print(f"\n=== Key insight ===")
print(f"When conv_input is bfloat16 and kernel is float32:")
print(f"  einsum promotes to float32: {result_2.dtype}")
print(f"This means z_conv_out will be float32, not bfloat16!")
print(f"We need to cast the kernel or the result.")

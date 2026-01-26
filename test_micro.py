#!/usr/bin/env python3
"""Micro test to debug seqcond model initialization and forward pass."""

import jax
import jax.numpy as jnp
from flax import linen as nn

# Force CPU for local testing
jax.config.update("jax_platform_name", "cpu")

print(f"JAX devices: {jax.devices()}")

# Import the model
from seqcond.jax.seqcond_fast import SeqCondAttention, SeqCondBlock

# Tiny config for testing
B, L, D = 2, 32, 64
K = 4
K_q = 2
H = 8
M = 2

print(
    f"\nTesting SeqCondAttention with B={B}, L={L}, D={D}, K={K}, K_q={K_q}, H={H}, M={M}"
)

# Create model - test with skip_low_rank=True (default)
model = SeqCondAttention(
    num_heads=K,
    num_query_heads=K_q,
    num_thetas=M,
    out_expand_factor=2,
    conv_kernel_size=4,
    dropout=0.0,
    compute_dtype=jnp.float32,
    skip_low_rank=True,  # Default config
)

# Initialize
key = jax.random.PRNGKey(42)
x = jax.random.normal(key, (B, L, D))

print(f"Input shape: {x.shape}")

try:
    print("\nInitializing model...")
    variables = model.init(key, x, deterministic=True)
    print("✓ Model initialized successfully")

    # Count params
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(variables["params"]))
    print(f"  Parameters: {param_count:,}")

    # Print param shapes
    print("\n  Param shapes:")

    def print_params(params, prefix=""):
        for k, v in params.items():
            if isinstance(v, dict):
                print_params(v, prefix + k + "/")
            else:
                print(f"    {prefix}{k}: {v.shape}")

    print_params(variables["params"])

except Exception as e:
    print(f"✗ Model init failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

try:
    print("\nRunning forward pass...")
    out = model.apply(variables, x, deterministic=True)
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {out.shape}")
    print(f"  Output min/max: {out.min():.4f} / {out.max():.4f}")
    print(f"  Output has NaN: {jnp.any(jnp.isnan(out))}")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Test a training step
print("\nTesting gradient computation...")
try:

    def loss_fn(params):
        out = model.apply({"params": params}, x, deterministic=True)
        return jnp.mean(out**2)

    grads = jax.grad(loss_fn)(variables["params"])
    print("✓ Gradient computation successful")

    # Check for NaN gradients
    grad_leaves = jax.tree_util.tree_leaves(grads)
    has_nan = any(jnp.any(jnp.isnan(g)) for g in grad_leaves)
    print(f"  Gradients have NaN: {has_nan}")

except Exception as e:
    print(f"✗ Gradient computation failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("\n✓ All tests passed!")

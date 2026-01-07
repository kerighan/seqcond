"""
Test that autoregressive generation using step() method produces the same output
as progressively feeding longer sequences to the forward pass.

This verifies that the O(1) step implementation is equivalent to the O(L) forward pass.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np

from seqcond.jax.seqcond_light import SeqCondAttention
from seqcond.jax.seqcond_summary import SeqCondAttention as SeqCondAttentionSummary
from seqcond.jax.seqcond_fast import SeqCondAttention as SeqCondAttentionFast


def init_state_light(model, B, K, H, M, conv_kernel_size):
    """Initialize state for seqcond_light/summary step method."""
    den_acc = jnp.zeros((B, K), dtype=jnp.float32)
    re_acc = jnp.zeros((B, K, H, M), dtype=jnp.float32)
    im_acc = jnp.zeros((B, K, H, M), dtype=jnp.float32)
    pos = jnp.zeros((B,), dtype=jnp.int32)

    # Conv buffer size is conv_kernel_size - 1
    dim_memory = K * H
    conv_buffer = jnp.zeros(
        (B, conv_kernel_size - 1, dim_memory + K), dtype=jnp.float32
    )

    return (den_acc, re_acc, im_acc, pos, conv_buffer)


def init_state_fast(model, B, K, K_q, H, M, conv_kernel_size):
    """Initialize state for seqcond_fast step method."""
    den_acc = jnp.zeros((B, K), dtype=jnp.float32)
    re_acc = jnp.zeros((B, K, H, M), dtype=jnp.float32)
    im_acc = jnp.zeros((B, K, H, M), dtype=jnp.float32)
    pos = jnp.zeros((B,), dtype=jnp.int32)

    # Two conv buffers for fast version
    dim_memory = K * H
    dim_query_head = H * M * 2
    dim_query_total = K_q * dim_query_head

    conv_buffer_mem = jnp.zeros(
        (B, conv_kernel_size - 1, dim_memory + K), dtype=jnp.float32
    )
    conv_buffer_query = jnp.zeros(
        (B, conv_kernel_size - 1, dim_query_total), dtype=jnp.float32
    )

    return (den_acc, re_acc, im_acc, pos, conv_buffer_mem, conv_buffer_query)


def test_model(model_class, model_name, num_query_heads=None):
    """Test that step and forward pass produce equivalent outputs."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")

    # Setup
    B = 2
    L = 10
    D = 128
    num_heads = 8
    num_thetas = 2
    conv_kernel_size = 4

    # Create model
    if num_query_heads is not None:
        model = model_class(
            num_heads=num_heads,
            num_query_heads=num_query_heads,
            num_thetas=num_thetas,
            conv_kernel_size=conv_kernel_size,
            expand_factor=1.0,
            compute_dtype=jnp.float32,
            param_dtype=jnp.float32,
        )
    else:
        model = model_class(
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

    # Initialize model parameters
    key, init_key = jax.random.split(key)
    variables = model.init(init_key, x_full, deterministic=True)

    # Compute dimensions
    d_inner = int(D * model.expand_factor)
    K = model.num_heads
    M = model.num_thetas
    H = max(1, d_inner // (K * M))

    print(f"Config: B={B}, L={L}, D={D}, K={K}, M={M}, H={H}")

    # Method 1: Progressive forward passes
    print("\nMethod 1: Progressive forward passes...")
    outputs_forward = []
    for i in range(1, L + 1):
        x_partial = x_full[:, :i, :]
        out = model.apply(variables, x_partial, deterministic=True)
        # Take only the last token output
        outputs_forward.append(out[:, -1, :])

    outputs_forward = jnp.stack(outputs_forward, axis=1)  # (B, L, D)
    print(f"Forward outputs shape: {outputs_forward.shape}")

    # Method 2: Step-by-step generation
    print("\nMethod 2: Step-by-step generation...")

    # Initialize state
    if num_query_heads is not None:
        K_q = num_query_heads
        state = init_state_fast(model, B, K, K_q, H, M, conv_kernel_size)
    else:
        state = init_state_light(model, B, K, H, M, conv_kernel_size)

    outputs_step = []
    for i in range(L):
        x_t = x_full[:, i, :]  # (B, D)
        out_t, state = model.apply(
            variables, x_t, state, deterministic=True, method=model.step
        )
        outputs_step.append(out_t)

    outputs_step = jnp.stack(outputs_step, axis=1)  # (B, L, D)
    print(f"Step outputs shape: {outputs_step.shape}")

    # Compare
    print("\nComparison:")
    max_diff = jnp.max(jnp.abs(outputs_forward - outputs_step))
    mean_diff = jnp.mean(jnp.abs(outputs_forward - outputs_step))
    rel_diff = mean_diff / (jnp.mean(jnp.abs(outputs_forward)) + 1e-8)

    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print(f"Relative difference: {rel_diff:.2e}")

    # Check if they match (within numerical precision)
    tolerance = 1e-4
    if max_diff < tolerance:
        print(f"✅ PASS: Outputs match within tolerance ({tolerance})")
        return True
    else:
        print(f"❌ FAIL: Outputs differ by more than tolerance ({tolerance})")

        # Show some sample values for debugging
        print("\nSample outputs (first batch, first 3 positions, first 5 dims):")
        print("Forward:", outputs_forward[0, :3, :5])
        print("Step:   ", outputs_step[0, :3, :5])
        return False


def main():
    print("=" * 60)
    print("Autoregressive Generation Equivalence Test")
    print("=" * 60)
    print("\nThis test verifies that the step() method produces the same")
    print("outputs as progressively feeding longer sequences to __call__().")

    results = {}

    # Test seqcond_light
    try:
        results["light"] = test_model(SeqCondAttention, "SeqCondAttention (light)")
    except Exception as e:
        print(f"\n❌ SeqCondAttention (light) failed with error: {e}")
        import traceback

        traceback.print_exc()
        results["light"] = False

    # Test seqcond_summary
    try:
        results["summary"] = test_model(
            SeqCondAttentionSummary, "SeqCondAttention (summary)"
        )
    except Exception as e:
        print(f"\n❌ SeqCondAttention (summary) failed with error: {e}")
        import traceback

        traceback.print_exc()
        results["summary"] = False

    # Test seqcond_fast
    try:
        results["fast"] = test_model(
            SeqCondAttentionFast, "SeqCondAttention (fast)", num_query_heads=4
        )
    except Exception as e:
        print(f"\n❌ SeqCondAttention (fast) failed with error: {e}")
        import traceback

        traceback.print_exc()
        results["fast"] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed.")
        return 1


if __name__ == "__main__":
    exit(main())

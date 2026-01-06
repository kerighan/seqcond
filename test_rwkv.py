"""
Test script for RWKV model.
Tests both inference and a dummy training loop.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import jax
import jax.numpy as jnp
import optax

from seqcond.config import Config, ModelConfig, TrainingConfig
from seqcond.jax.train import create_model_from_config
from seqcond.jax.model import (
    init_model,
    count_parameters,
    sparse_categorical_crossentropy_loss,
)


def test_inference():
    """Test RWKV inference."""
    print("\n" + "=" * 60)
    print("Testing RWKV Inference")
    print("=" * 60)

    # Create small RWKV config
    config = ModelConfig.small(model_type="rwkv")
    print(
        f"Model config: {config.model_type}, layers={config.num_layers}, d_model={config.d_model}"
    )

    # Create model
    model = create_model_from_config(config)

    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    batch_size = 2
    seq_len = 64
    input_shape = (batch_size, seq_len)

    print(f"Initializing model with input shape: {input_shape}")
    variables = init_model(model, rng, input_shape=input_shape)
    params = variables["params"]

    # Count parameters
    total_params = count_parameters(params)
    print(f"Total parameters: {total_params:,}")

    # Create dummy input
    dummy_input = jax.random.randint(
        jax.random.PRNGKey(123),
        shape=input_shape,
        minval=0,
        maxval=config.vocab_size,
    )

    # Run inference
    print("Running forward pass...")
    logits = model.apply(variables, dummy_input, deterministic=True)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")

    # Verify output shape
    assert logits.shape == (
        batch_size,
        seq_len,
        config.vocab_size,
    ), f"Output shape mismatch: {logits.shape} vs expected ({batch_size}, {seq_len}, {config.vocab_size})"

    # Check for NaNs
    assert not jnp.any(jnp.isnan(logits)), "Output contains NaN values!"

    print("✓ Inference test passed!")
    return model, variables, config


def test_training():
    """Test RWKV training loop."""
    print("\n" + "=" * 60)
    print("Testing RWKV Training")
    print("=" * 60)

    # Create small RWKV config
    config = ModelConfig.small(model_type="rwkv")

    # Create model
    model = create_model_from_config(config)

    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    batch_size = 2
    seq_len = 64
    input_shape = (batch_size, seq_len)

    variables = init_model(model, rng, input_shape=input_shape)
    params = variables["params"]

    # Create optimizer
    learning_rate = 1e-4
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    print(f"Optimizer: Adam with lr={learning_rate}")

    # Define loss function
    def loss_fn(params, inputs, targets):
        logits = model.apply({"params": params}, inputs, deterministic=False)
        loss = sparse_categorical_crossentropy_loss(logits, targets)
        return loss

    # Define training step
    @jax.jit
    def train_step(params, opt_state, inputs, targets):
        loss, grads = jax.value_and_grad(loss_fn)(params, inputs, targets)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Run a few training steps
    num_steps = 5
    print(f"\nRunning {num_steps} training steps...")

    for step in range(num_steps):
        # Create dummy batch
        rng, input_rng, target_rng = jax.random.split(rng, 3)

        inputs = jax.random.randint(
            input_rng,
            shape=(batch_size, seq_len),
            minval=0,
            maxval=config.vocab_size,
        )

        # Targets are inputs shifted by 1 (language modeling)
        targets = jnp.roll(inputs, -1, axis=1)

        # Training step
        params, opt_state, loss = train_step(params, opt_state, inputs, targets)

        print(f"Step {step + 1}/{num_steps}: loss = {loss:.4f}")

        # Check for NaNs
        assert not jnp.isnan(loss), f"Loss is NaN at step {step + 1}!"

    print("✓ Training test passed!")
    return params, opt_state


def test_gradient_flow():
    """Test that gradients flow through the model."""
    print("\n" + "=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)

    # Create small RWKV config
    config = ModelConfig.small(model_type="rwkv")
    model = create_model_from_config(config)

    # Initialize
    rng = jax.random.PRNGKey(42)
    variables = init_model(model, rng, input_shape=(1, 32))
    params = variables["params"]

    # Create dummy data
    inputs = jax.random.randint(
        jax.random.PRNGKey(123),
        shape=(1, 32),
        minval=0,
        maxval=config.vocab_size,
    )
    targets = jnp.roll(inputs, -1, axis=1)

    # Compute gradients
    def loss_fn(params):
        logits = model.apply({"params": params}, inputs, deterministic=True)
        return sparse_categorical_crossentropy_loss(logits, targets)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    # Check gradient statistics
    grad_leaves = jax.tree_util.tree_leaves(grads)
    # Filter out non-array leaves (e.g., dicts, strings)
    grad_arrays = [g for g in grad_leaves if isinstance(g, jnp.ndarray)]
    grad_norms = [jnp.linalg.norm(g.ravel()) for g in grad_arrays]

    print(f"Loss: {loss:.4f}")
    print(
        f"Number of gradient tensors: {len(grad_arrays)} (out of {len(grad_leaves)} leaves)"
    )
    print(f"Max gradient norm: {float(max(grad_norms)):.6f}")
    print(f"Min gradient norm: {float(min(grad_norms)):.6f}")
    print(f"Mean gradient norm: {float(jnp.mean(jnp.array(grad_norms))):.6f}")

    # Check that gradients are not all zero
    total_grad_norm = jnp.sqrt(sum([jnp.sum(g**2) for g in grad_arrays]))
    print(f"Total gradient norm: {float(total_grad_norm):.6f}")

    assert total_grad_norm > 0, "All gradients are zero!"
    assert not jnp.isnan(total_grad_norm), "Gradient norm is NaN!"

    print("✓ Gradient flow test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RWKV Model Test Suite")
    print("=" * 60)

    try:
        # Test 1: Inference
        model, variables, config = test_inference()

        # Test 2: Training
        params, opt_state = test_training()

        # Test 3: Gradient flow
        test_gradient_flow()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

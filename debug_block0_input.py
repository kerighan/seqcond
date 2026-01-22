"""
Debug script to check if Block 0 receives identical inputs in both scenarios.
"""

import jax
import jax.numpy as jnp
import pickle
from seqcond.jax.model import SeqCondModel

CHECKPOINT_PATH = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step50000.pkl"

with open(CHECKPOINT_PATH, "rb") as f:
    checkpoint = pickle.load(f)

config = checkpoint["config"]["model"]
params = checkpoint["params"]

print(
    f"Config: L={config['num_layers']}, D={config['d_model']}, seqcond_ratio={config['seqcond_ratio']}"
)
print(f"maxlen={config['maxlen']}")

# Create model (remove keys not in SeqCondModel.__init__)
model_config = {
    k: v for k, v in config.items() if k not in ["model_type", "state_size"]
}
model = SeqCondModel(**model_config)

# Test input
B = 2
L = 1
rng = jax.random.PRNGKey(42)
inputs = jax.random.randint(rng, (B, L), 0, config["vocab_size"])

# Initialize states using apply
states = model.apply(
    {"params": params},
    B,
    method=lambda module, batch_size: module.init_state(batch_size),
)

print("\n=== Checking Block 0 Input ===")


# Define a debug function to run inside apply
def debug_block0(module, inputs_arg, states_arg):
    # Get embedding output
    x_emb = module.embedding(inputs_arg)

    if module.use_positional_embedding:
        positions = jnp.arange(L, dtype=jnp.int32)[None, :]
        x_emb = x_emb + module.position_embedding(positions)

    print(f"Embedding output shape: {x_emb.shape}")
    print(
        f"Embedding output (call mode): mean={jnp.mean(x_emb):.6e}, std={jnp.std(x_emb):.6e}"
    )

    # For step mode
    x_emb_step = x_emb[:, 0, :]  # (B, D)
    print(
        f"Embedding output (step mode): mean={jnp.mean(x_emb_step):.6e}, std={jnp.std(x_emb_step):.6e}"
    )

    # Check state initialization
    state_0 = states_arg[0]
    den_acc, re_acc, im_acc, pos, conv_buffer = state_0

    print(f"\nState 0 shapes:")
    print(f"  den_acc: {den_acc.shape}, dtype={den_acc.dtype}")
    print(f"  re_acc: {re_acc.shape}, dtype={re_acc.dtype}")
    print(f"  im_acc: {im_acc.shape}, dtype={im_acc.dtype}")
    print(f"  pos: {pos.shape}, dtype={pos.dtype}")
    print(f"  conv_buffer: {conv_buffer.shape}, dtype={conv_buffer.dtype}")

    print(f"\nState 0 values (should be all zeros):")
    print(f"  den_acc sum: {jnp.sum(jnp.abs(den_acc)):.6e}")
    print(f"  re_acc sum: {jnp.sum(jnp.abs(re_acc)):.6e}")
    print(f"  im_acc sum: {jnp.sum(jnp.abs(im_acc)):.6e}")
    print(f"  pos: {pos}")
    print(f"  conv_buffer sum: {jnp.sum(jnp.abs(conv_buffer)):.6e}")

    # Get Block 0
    block_type, block = module.blocks[0]
    print(f"\nBlock 0 type: {block_type}")

    # Call mode
    mask = inputs_arg != 0
    out_call = block(x_emb, mask=mask, deterministic=True)

    print(f"\nBlock 0 output (call mode): shape={out_call.shape}")
    print(f"  mean={jnp.mean(out_call):.6e}, std={jnp.std(out_call):.6e}")
    print(
        f"  First token: mean={jnp.mean(out_call[:, 0, :]):.6e}, std={jnp.std(out_call[:, 0, :]):.6e}"
    )

    # Step mode
    out_step, new_state = block.step(x_emb_step, state_0, deterministic=True)

    print(f"\nBlock 0 output (step mode): shape={out_step.shape}")
    print(f"  mean={jnp.mean(out_step):.6e}, std={jnp.std(out_step):.6e}")

    # Compare
    diff = jnp.abs(out_call[:, 0, :] - out_step)
    max_diff = jnp.max(diff)
    mean_diff = jnp.mean(diff)

    print(f"\n=== Block 0 Consistency ===")
    print(f"Max diff: {max_diff:.6e}")
    print(f"Mean diff: {mean_diff:.6e}")

    if max_diff < 1e-5:
        print("✓ Block 0 is consistent!")
    else:
        print("✗ Block 0 has divergence!")

        # Find where the largest differences are
        max_idx = jnp.unravel_index(jnp.argmax(diff), diff.shape)
        print(f"  Largest diff at index {max_idx}")
        print(f"  Call value: {out_call[:, 0, :][max_idx]:.6e}")
        print(f"  Step value: {out_step[max_idx]:.6e}")

    return max_diff


# Run the debug function
max_diff = model.apply({"params": params}, inputs, states, method=debug_block0)

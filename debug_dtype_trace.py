"""
Trace dtype conversions through the SeqCondAttention step to find mismatches.
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

model_config = {
    k: v for k, v in config.items() if k not in ["model_type", "state_size"]
}
model = SeqCondModel(**model_config)

B = 2
L = 1
rng = jax.random.PRNGKey(42)
inputs = jax.random.randint(rng, (B, L), 0, config["vocab_size"])

states = model.apply(
    {"params": params},
    B,
    method=lambda module, batch_size: module.init_state(batch_size),
)

print("=== Dtype Trace Through SeqCondAttention ===\n")


def trace_dtypes(module, inputs_arg, states_arg):
    # Get embedding
    x_emb = module.embedding(inputs_arg)
    if module.use_positional_embedding:
        positions = jnp.arange(L, dtype=jnp.int32)[None, :]
        x_emb = x_emb + module.position_embedding(positions)

    x_emb_call = x_emb  # (B, 1, D)
    x_emb_step = x_emb[:, 0, :]  # (B, D)

    block_type, block = module.blocks[0]
    state_0 = states_arg[0]
    den_acc, re_acc, im_acc, pos, conv_buffer = state_0

    print(f"Input dtypes:")
    print(f"  x_emb_call: {x_emb_call.dtype}")
    print(f"  x_emb_step: {x_emb_step.dtype}")
    print(f"  conv_buffer: {conv_buffer.dtype}")
    print(f"  re_acc: {re_acc.dtype}")
    print(f"  im_acc: {im_acc.dtype}")
    print(f"  den_acc: {den_acc.dtype}")

    # Get parameters to trace through manually
    in_proj_kernel = module.params["seqcond_block_0"]["attn"]["in_proj"]["kernel"]
    conv_kernel = module.params["seqcond_block_0"]["attn"]["conv"]["kernel"]

    print(f"\nParameter dtypes:")
    print(f"  in_proj_kernel: {in_proj_kernel.dtype}")
    print(f"  conv_kernel: {conv_kernel.dtype}")

    # Trace __call__ path
    print(f"\n=== __call__ path ===")
    z_all_call = jnp.einsum("bld,de->ble", x_emb_call, in_proj_kernel)
    print(f"z_all (before cast): {z_all_call.dtype}")
    z_all_call = z_all_call.astype(jnp.bfloat16)
    print(f"z_all (after cast): {z_all_call.dtype}")

    # Trace step path
    print(f"\n=== step path ===")
    z_all_step = jnp.einsum("bd,de->be", x_emb_step, in_proj_kernel)
    print(f"z_all (before cast): {z_all_step.dtype}")
    z_all_step = z_all_step.astype(jnp.bfloat16)
    print(f"z_all (after cast): {z_all_step.dtype}")

    # Check conv operation
    dim_conv_total = 4350  # from earlier
    z_conv_call = z_all_call[..., :dim_conv_total]
    z_conv_step = z_all_step[..., :dim_conv_total]

    print(f"\n=== Convolution ===")
    print(f"z_conv_call: {z_conv_call.dtype}")
    print(f"z_conv_step: {z_conv_step.dtype}")

    # Step mode conv
    z_conv_expanded = z_conv_step[:, None, :]
    print(f"z_conv_expanded: {z_conv_expanded.dtype}")

    conv_buffer_cast = conv_buffer.astype(jnp.bfloat16)
    print(f"conv_buffer_cast: {conv_buffer_cast.dtype}")

    conv_input = jnp.concatenate([conv_buffer_cast, z_conv_expanded], axis=1)
    print(f"conv_input: {conv_input.dtype}")

    z_conv_out_step = jnp.einsum("bkc,kc->bc", conv_input, conv_kernel[:, 0, :])
    print(f"z_conv_out_step: {z_conv_out_step.dtype}")

    z_conv_act_step = jax.nn.silu(z_conv_out_step)
    print(f"z_conv_act_step: {z_conv_act_step.dtype}")

    # Call mode conv (manual simulation)
    K = 4
    z_conv_padded = jnp.pad(z_conv_call, ((0, 0), (K - 1, 0), (0, 0)), mode="constant")
    window = z_conv_padded[:, 0:K, :]
    z_conv_out_call = jnp.einsum("bkc,kc->bc", window, conv_kernel[:, 0, :])
    print(f"z_conv_out_call: {z_conv_out_call.dtype}")

    z_conv_act_call = jax.nn.silu(z_conv_out_call)
    print(f"z_conv_act_call: {z_conv_act_call.dtype}")

    print(f"\n=== Conv activation diff ===")
    diff = jnp.max(jnp.abs(z_conv_act_call - z_conv_act_step))
    print(f"Max diff: {diff:.6e}")

    # Now check if the issue is in the buffer update
    print(f"\n=== Buffer update ===")
    conv_buffer_new_wrong = jnp.concatenate(
        [conv_buffer[:, 1:, :], z_conv_expanded], axis=1
    )
    print(f"conv_buffer_new (wrong - mixed dtypes): {conv_buffer_new_wrong.dtype}")

    conv_buffer_new_right = jnp.concatenate(
        [conv_buffer_cast[:, 1:, :], z_conv_expanded], axis=1
    )
    print(f"conv_buffer_new (right - consistent dtype): {conv_buffer_new_right.dtype}")

    return diff


max_diff = model.apply({"params": params}, inputs, states, method=trace_dtypes)

print(f"\n=== Summary ===")
print(f"Max diff in conv activation: {max_diff:.6e}")

"""
Debug script to trace the convolution operation in detail for Block 0.
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

# Create model
model_config = {
    k: v for k, v in config.items() if k not in ["model_type", "state_size"]
}
model = SeqCondModel(**model_config)

# Test input
B = 2
L = 1
rng = jax.random.PRNGKey(42)
inputs = jax.random.randint(rng, (B, L), 0, config["vocab_size"])

# Initialize states
states = model.apply(
    {"params": params},
    B,
    method=lambda module, batch_size: module.init_state(batch_size),
)

print("=== Detailed Convolution Debug ===\n")


def debug_conv_operation(module, inputs_arg, states_arg):
    # Get embedding
    x_emb = module.embedding(inputs_arg)
    if module.use_positional_embedding:
        positions = jnp.arange(L, dtype=jnp.int32)[None, :]
        x_emb = x_emb + module.position_embedding(positions)

    x_emb_call = x_emb  # (B, 1, D)
    x_emb_step = x_emb[:, 0, :]  # (B, D)

    block_type, block = module.blocks[0]
    state_0 = states_arg[0]

    # Access the SeqCondAttention layer
    attn = block.attn

    # Get in_proj parameters
    in_proj_kernel = module.params["seqcond_block_0"]["attn"]["in_proj"]["kernel"]
    print(f"in_proj kernel shape: {in_proj_kernel.shape}")
    print(f"in_proj kernel dtype: {in_proj_kernel.dtype}")

    # Compute z_all for both modes
    # Call mode
    z_all_call = jnp.einsum("bld,de->ble", x_emb_call, in_proj_kernel)
    z_all_call = z_all_call.astype(jnp.bfloat16)  # compute_dtype

    # Step mode
    z_all_step = jnp.einsum("bd,de->be", x_emb_step, in_proj_kernel)
    z_all_step = z_all_step.astype(jnp.bfloat16)

    print(
        f"\nz_all (call): mean={jnp.mean(z_all_call):.6e}, std={jnp.std(z_all_call):.6e}"
    )
    print(
        f"z_all (step): mean={jnp.mean(z_all_step):.6e}, std={jnp.std(z_all_step):.6e}"
    )
    print(f"z_all diff: {jnp.max(jnp.abs(z_all_call[:, 0, :] - z_all_step)):.6e}")

    # Split z_all
    num_heads = attn.num_heads
    num_query_heads = attn.num_query_heads
    num_thetas = attn.num_thetas
    d_inner = int(960 * 3)  # expand_factor=3
    H = max(1, d_inner // (num_heads * num_thetas))

    dim_memory = num_heads * H
    dim_query_head = H * num_thetas * 2
    dim_query_total = num_query_heads * dim_query_head
    dim_mem_total = dim_memory + num_heads
    dim_conv_total = dim_mem_total + dim_query_total

    print(f"\nDimensions: H={H}, dim_conv_total={dim_conv_total}")

    # Split for call mode
    z_conv_call = z_all_call[..., :dim_conv_total]
    c_skip_call = z_all_call[..., dim_conv_total : dim_conv_total + 960]

    # Split for step mode
    z_conv_step = z_all_step[..., :dim_conv_total]
    c_skip_step = z_all_step[..., dim_conv_total : dim_conv_total + 960]

    print(
        f"\nz_conv (call): mean={jnp.mean(z_conv_call):.6e}, std={jnp.std(z_conv_call):.6e}"
    )
    print(
        f"z_conv (step): mean={jnp.mean(z_conv_step):.6e}, std={jnp.std(z_conv_step):.6e}"
    )
    print(f"z_conv diff: {jnp.max(jnp.abs(z_conv_call[:, 0, :] - z_conv_step)):.6e}")

    # Get conv kernel
    conv_kernel = module.params["seqcond_block_0"]["attn"]["conv"]["kernel"]
    print(f"\nconv kernel shape: {conv_kernel.shape}")
    print(f"conv kernel dtype: {conv_kernel.dtype}")

    # Call mode: use nn.Conv (padding is applied internally)
    # We need to manually replicate what nn.Conv does
    # padding=((K-1, 0),) means pad K-1 zeros on the left
    K = 4  # conv_kernel_size
    z_conv_padded = jnp.pad(z_conv_call, ((0, 0), (K - 1, 0), (0, 0)), mode="constant")
    print(f"\nz_conv_padded shape: {z_conv_padded.shape}")

    # Manual depthwise conv for call mode
    z_conv_out_call_manual = jnp.zeros((B, L, dim_conv_total), dtype=jnp.bfloat16)
    for i in range(L):
        window = z_conv_padded[:, i : i + K, :]  # (B, K, C)
        z_conv_out_call_manual = z_conv_out_call_manual.at[:, i, :].set(
            jnp.einsum("bkc,kc->bc", window, conv_kernel[:, 0, :])
        )

    z_conv_out_call_manual = z_conv_out_call_manual.astype(jnp.bfloat16)

    # Step mode: manual einsum with conv_buffer
    den_acc, re_acc, im_acc, pos, conv_buffer = state_0
    print(f"\nconv_buffer dtype: {conv_buffer.dtype}")
    print(f"conv_buffer shape: {conv_buffer.shape}")

    z_conv_expanded = z_conv_step[:, None, :]  # (B, 1, C)
    conv_input = jnp.concatenate([conv_buffer, z_conv_expanded], axis=1)  # (B, K, C)
    print(f"conv_input dtype: {conv_input.dtype}")
    print(f"conv_input shape: {conv_input.shape}")

    # Cast conv_input to bfloat16 to match compute_dtype
    conv_input_bf16 = conv_input.astype(jnp.bfloat16)
    z_conv_out_step = jnp.einsum("bkc,kc->bc", conv_input_bf16, conv_kernel[:, 0, :])
    z_conv_out_step = z_conv_out_step.astype(jnp.bfloat16)

    print(
        f"\nz_conv_out (call manual): mean={jnp.mean(z_conv_out_call_manual):.6e}, std={jnp.std(z_conv_out_call_manual):.6e}"
    )
    print(
        f"z_conv_out (step): mean={jnp.mean(z_conv_out_step):.6e}, std={jnp.std(z_conv_out_step):.6e}"
    )
    print(
        f"z_conv_out diff: {jnp.max(jnp.abs(z_conv_out_call_manual[:, 0, :] - z_conv_out_step)):.6e}"
    )

    # Apply activation
    z_conv_act_call = jax.nn.silu(z_conv_out_call_manual[:, 0, :])
    z_conv_act_step = jax.nn.silu(z_conv_out_step)

    print(
        f"\nz_conv_act (call): mean={jnp.mean(z_conv_act_call):.6e}, std={jnp.std(z_conv_act_call):.6e}"
    )
    print(
        f"z_conv_act (step): mean={jnp.mean(z_conv_act_step):.6e}, std={jnp.std(z_conv_act_step):.6e}"
    )
    print(f"z_conv_act diff: {jnp.max(jnp.abs(z_conv_act_call - z_conv_act_step)):.6e}")

    return jnp.max(jnp.abs(z_conv_act_call - z_conv_act_step))


max_diff = model.apply({"params": params}, inputs, states, method=debug_conv_operation)

print(f"\n=== Summary ===")
print(f"Max diff in z_conv_act: {max_diff:.6e}")

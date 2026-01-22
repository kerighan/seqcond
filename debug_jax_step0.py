import jax
import jax.numpy as jnp
import numpy as np
import pickle
from seqcond.jax.model import SeqCondModel
from seqcond.config import ModelConfig
from seqcond.jax.seqcond_fast import SeqCondBlock

CHECKPOINT_PATH = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step50000.pkl"


def load_first_block():
    print(f"Loading checkpoint...")
    with open(CHECKPOINT_PATH, "rb") as f:
        data = pickle.load(f)

    full_params = data["params"]
    config = data["config"]
    model_config = ModelConfig(**config["model"])

    # Extract params for block 0
    # Structure: params['seqcond_block_0'] ...
    block_params = full_params["seqcond_block_0"]

    # Create Block
    block = SeqCondBlock(
        num_heads=model_config.seqcond_heads,
        num_query_heads=model_config.num_query_heads,
        expand_factor=model_config.expand_factor,
        out_expand_factor=model_config.out_expand_factor,
        num_thetas=model_config.num_thetas,
        num_anchor_heads=model_config.num_anchor_heads,
        conv_kernel_size=model_config.conv_kernel_size,
        skip_low_rank=False,  # Assuming default/config
        dropout=0.0,
        maxlen=model_config.maxlen,
        chunk_size=0,
        use_square_matrix=False,
        compute_dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    )

    return block, block_params, model_config


def debug_step0():
    block, params, config = load_first_block()

    B = 1
    D = config.d_model
    rng = np.random.default_rng(42)

    # Random input embedding x0
    x0 = rng.normal(0, 1, (B, D)).astype(np.float32)
    x0_jax = jnp.array(x0)

    print("--- Running __call__ on L=1 ---")
    # Input shape (B, L, D) = (1, 1, D)
    x_seq = x0_jax[:, None, :]

    # We need to wrap it in a function to print intermediates if needed,
    # but let's first check output.
    out_call = block.apply(
        {"params": params},
        x_seq,
        deterministic=True,
    )  # (1, 1, D)

    out_call_0 = out_call[:, 0, :]

    print("--- Running step on t=0 ---")
    # Init state
    # We need to manually init state structure as block.step expects it.
    # SeqCond state: (den_acc, re_acc, im_acc, pos, conv_buffer)

    # Helper to get dimensions
    def init_state_fn(module):
        # Dummy call to init state shapes?
        # Easier to replicate logic from model.py init_state
        num_heads = config.seqcond_heads
        num_query_heads = config.num_query_heads
        num_thetas = config.num_thetas
        d_inner = int(D * config.expand_factor)
        H = max(1, d_inner // (num_heads * num_thetas))
        conv_kernel_size = config.conv_kernel_size

        dim_memory = num_heads * H
        dim_query_head = H * num_thetas * 2
        dim_query_total = num_query_heads * dim_query_head
        dim_mem_total = dim_memory + num_heads
        dim_conv_total = dim_mem_total + dim_query_total

        den_acc = jnp.zeros((B, num_heads), dtype=jnp.float32)
        re_acc = jnp.zeros((B, num_heads, H, num_thetas), dtype=jnp.float32)
        im_acc = jnp.zeros((B, num_heads, H, num_thetas), dtype=jnp.float32)
        pos = jnp.zeros((B,), dtype=jnp.int32)
        conv_buffer = jnp.zeros(
            (B, conv_kernel_size - 1, dim_conv_total), dtype=jnp.float32
        )

        return (den_acc, re_acc, im_acc, pos, conv_buffer)

    state_0 = init_state_fn(block)

    out_step, new_state = block.apply(
        {"params": params},
        x0_jax,  # (B, D)
        state_0,
        deterministic=True,
        method=block.step,
    )

    print("\n--- Comparison ---")
    print(f"Call output shape: {out_call_0.shape}")
    print(f"Step output shape: {out_step.shape}")

    diff = jnp.abs(out_call_0 - out_step)
    max_diff = jnp.max(diff)
    mean_diff = jnp.mean(diff)

    print(f"Max Diff: {max_diff:.6e}")
    print(f"Mean Diff: {mean_diff:.6e}")

    print("Call[:5]:", out_call_0[0, :5])
    print("Step[:5]:", out_step[0, :5])


if __name__ == "__main__":
    debug_step0()

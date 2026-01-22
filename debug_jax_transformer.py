import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from seqcond.jax.rope import (
    TransformerDecoderBlock,
    precompute_freqs,
    get_rope_embeddings,
)


def debug_transformer_consistency():
    print("Initializing Transformer Block...")

    # Config
    D = 64
    H = 4
    L_max = 32
    batch_size = 1

    # Initialize block
    block = TransformerDecoderBlock(
        d_model=D,
        num_heads=H,
        d_ff=D * 4,
        num_kv_heads=H,
        dropout=0.0,
        qk_norm=True,
    )

    # Init variables
    rng = jax.random.PRNGKey(0)
    dummy_x = jnp.ones((batch_size, 1, D))
    dummy_cos = jnp.ones((batch_size, 1, H, D // H // 2))
    dummy_sin = jnp.ones((batch_size, 1, H, D // H // 2))

    variables = block.init(rng, dummy_x, dummy_cos, dummy_sin)
    params = variables["params"]

    # Data
    L_seq = 5
    x_seq = jax.random.normal(rng, (batch_size, L_seq, D))

    # RoPE
    cos_emb, sin_emb = precompute_freqs(L_max, D // H)
    cos, sin = get_rope_embeddings(L_seq, cos_emb, sin_emb, batch_size, H)

    print(f"Running __call__ on sequence length {L_seq}...")
    out_call = block.apply({"params": params}, x_seq, cos, sin, deterministic=True)

    print("Running step loop...")
    # Init Cache
    k_cache = jnp.zeros((batch_size, L_max, H, D // H))
    v_cache = jnp.zeros((batch_size, L_max, H, D // H))
    kv_cache = (k_cache, v_cache)

    out_step_list = []

    for t in range(L_seq):
        x_t = x_seq[:, t : t + 1, :]  # (B, 1, D) -- step expects (B, D) usually?
        # model.py step receives (B, 1, D) from embedding but does x[:, 0, :]
        # TransformerDecoderBlock.step expects x_t: (B, D) based on usage in model.py
        # "x, new_state = block.step(x, ...)" where x is (B, D).

        x_t_flat = x_t[:, 0, :]

        # RoPE slice
        head_dim_half = D // H // 2
        cos_t = jax.lax.dynamic_slice(cos_emb, (t, 0), (1, head_dim_half))
        sin_t = jax.lax.dynamic_slice(sin_emb, (t, 0), (1, head_dim_half))
        # Broadcast to (B, 1, H, D//2)
        cos_t = jnp.broadcast_to(
            cos_t[None, :, None, :], (batch_size, 1, H, head_dim_half)
        )
        sin_t = jnp.broadcast_to(
            sin_t[None, :, None, :], (batch_size, 1, H, head_dim_half)
        )

        out_t, kv_cache = block.apply(
            {"params": params},
            x_t_flat,
            kv_cache,
            jnp.array(t, dtype=jnp.int32),
            cos_t,
            sin_t,
            deterministic=True,
            method=block.step,
        )
        out_step_list.append(out_t)

    out_step = jnp.stack(out_step_list, axis=1)  # (B, L, D)

    print("\n--- Comparison ---")
    print(f"Call output shape: {out_call.shape}")
    print(f"Step output shape: {out_step.shape}")

    diff = jnp.abs(out_call - out_step)
    max_diff = jnp.max(diff)
    mean_diff = jnp.mean(diff)

    print(f"Max Diff: {max_diff:.6e}")
    print(f"Mean Diff: {mean_diff:.6e}")

    for t in range(L_seq):
        d_t = jnp.max(jnp.abs(out_call[:, t] - out_step[:, t]))
        print(f"Step {t} max diff: {d_t:.6e}")
        if d_t > 1e-4:
            print(f"  Call: {out_call[0, t, :5]}")
            print(f"  Step: {out_step[0, t, :5]}")


if __name__ == "__main__":
    debug_transformer_consistency()

import jax
import jax.numpy as jnp
import numpy as np
import pickle
from seqcond.jax.model import SeqCondModel
from seqcond.config import ModelConfig
from seqcond.jax.rope import get_rope_embeddings

CHECKPOINT_PATH = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step50000.pkl"


class DebugSeqCondModel(SeqCondModel):
    def debug_consistency(self, inputs, states):
        """
        Run both __call__ (L=1) and step (t=0) layer by layer and compare.
        inputs: (B, 1)
        states: initial states
        """
        print("--- Debugging Consistency Layer by Layer ---")

        b, l = inputs.shape  # (B, 1)
        mask = inputs != 0

        # 1. Embeddings
        x_call = self.embedding(inputs)  # (B, 1, D)
        x_step = self.embedding(inputs)[:, 0, :]  # (B, D)

        if self.use_positional_embedding:
            positions = jnp.arange(l, dtype=jnp.int32)[None, :]
            x_call = x_call + self.position_embedding(positions)

            pos_emb = self.position_embedding(jnp.array([[0]]))[:, 0, :]
            x_step = x_step + pos_emb

        print(f"Embedding Diff: {jnp.max(jnp.abs(x_call[:, 0, :] - x_step)):.6e}")

        # 2. RoPE
        # Call mode
        cos_call, sin_call = get_rope_embeddings(
            l, self.cos_emb, self.sin_emb, b, self.num_heads
        )

        # Step mode (t=0)
        pos = 0
        head_dim_half = self.cos_emb.shape[1]
        cos_t = jax.lax.dynamic_slice(self.cos_emb, (pos, 0), (1, head_dim_half))
        sin_t = jax.lax.dynamic_slice(self.sin_emb, (pos, 0), (1, head_dim_half))
        cos_t = cos_t[None, :, None, :]  # (1, 1, 1, head_dim//2)
        sin_t = sin_t[None, :, None, :]
        cos_step = jnp.broadcast_to(cos_t, (b, 1, self.num_heads, head_dim_half))
        sin_step = jnp.broadcast_to(sin_t, (b, 1, self.num_heads, head_dim_half))

        print(f"RoPE Cos Diff: {jnp.max(jnp.abs(cos_call - cos_step)):.6e}")

        # 3. Layers
        current_x_call = x_call
        current_x_step = x_step

        new_states = []

        for i, (block_type, block) in enumerate(self.blocks):
            # Capture inputs before block
            in_call = current_x_call
            in_step = current_x_step

            # --- CALL ---
            if block_type == "transformer":
                out_call = block(
                    current_x_call,
                    cos=cos_call,
                    sin=sin_call,
                    mask=mask,
                    deterministic=True,
                )
            else:
                out_call = block(current_x_call, mask=mask, deterministic=True)

            # --- STEP ---
            if block_type == "transformer":
                # Transformer step expects x_t: (B, D)
                out_step, new_state = block.step(
                    current_x_step,
                    states[i],
                    jnp.array(pos, dtype=jnp.int32),
                    cos_step,
                    sin_step,
                    deterministic=True,
                )
            else:
                # SeqCond step expects x_t: (B, D)
                out_step, new_state = block.step(
                    current_x_step, states[i], deterministic=True
                )
            new_states.append(new_state)

            # Compare outputs
            out_call_flat = out_call[:, 0, :]
            diff = jnp.abs(out_call_flat - out_step)
            max_diff = jnp.max(diff)

            print(f"Block {i:2d} ({block_type}): Max Diff = {max_diff:.6e}")

            if max_diff > 1e-4:
                print(f"❌ DIVERGENCE FOUND at Block {i}")
                print(
                    f"  Input Diff: {jnp.max(jnp.abs(in_call[:, 0, :] - in_step)):.6e}"
                )
                print(f"  Call Output[:5]: {out_call_flat[0, :5]}")
                print(f"  Step Output[:5]: {out_step[0, :5]}")
                return  # Stop at first divergence

            current_x_call = out_call
            current_x_step = out_step

        print("✅ All blocks passed consistency check.")


def load_and_debug():
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    with open(CHECKPOINT_PATH, "rb") as f:
        data = pickle.load(f)

    params = data["params"]
    config_dict = data["config"]
    model_config = ModelConfig(**config_dict["model"])

    # Instantiate Debug Model
    model = DebugSeqCondModel(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        num_kv_heads=model_config.num_kv_heads,
        d_ff=model_config.d_ff,
        maxlen=model_config.maxlen,
        dropout=0.0,
        tie_weights=model_config.tie_weights,
        qk_norm=model_config.qk_norm,
        qk_norm_eps=model_config.qk_norm_eps,
        seqcond_heads=model_config.seqcond_heads,
        num_query_heads=model_config.num_query_heads,
        num_thetas=model_config.num_thetas,
        derivative_order=model_config.derivative_order,
        num_anchor_heads=model_config.num_anchor_heads,
        conv_kernel_size=model_config.conv_kernel_size,
        expand_factor=model_config.expand_factor,
        out_expand_factor=model_config.out_expand_factor,
        seqcond_ratio=model_config.seqcond_ratio,
        use_square_matrix=model_config.use_square_matrix,
        remat=False,
    )

    # Init inputs
    B = 1
    rng = np.random.default_rng(42)
    inputs = rng.integers(0, model_config.vocab_size, size=(B, 1)).astype(np.int32)
    inputs_jax = jnp.array(inputs)

    # Init states using the model method (it will use the base class method)
    states = model.apply({"params": params}, batch_size=B, method=model.init_state)

    # Run debug method
    model.apply({"params": params}, inputs_jax, states, method=model.debug_consistency)


if __name__ == "__main__":
    load_and_debug()

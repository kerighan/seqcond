import jax
import jax.numpy as jnp
import numpy as np
import pickle
import sys
from seqcond.jax.model import SeqCondModel
from seqcond.config import ModelConfig

CHECKPOINT_PATH = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step50000.pkl"


def load_model():
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    with open(CHECKPOINT_PATH, "rb") as f:
        data = pickle.load(f)

    params = data["params"]
    config_dict = data["config"]
    model_config_dict = config_dict["model"]
    model_config = ModelConfig(**model_config_dict)

    model = SeqCondModel(
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
    return model, params, model_config


def run_test():
    model, params, config = load_model()

    print(f"Model config maxlen: {config.maxlen}")

    # Create random input sequence
    # Use a small sequence length for debugging
    B = 1
    L = 16
    rng = np.random.default_rng(42)
    inputs = rng.integers(0, config.vocab_size, size=(B, L)).astype(np.int32)
    inputs_jax = jnp.array(inputs)

    print(f"Running __call__ (parallel) on sequence length {L}...")
    # Run __call__ (Parallel / Prefill mode)
    logits_parallel = model.apply(
        {"params": params},
        inputs_jax,
        deterministic=True,
    )

    print("Running step (recurrent) loop...")
    # Run step (Recurrent / Generation mode)
    # 1. Init state
    states = model.apply({"params": params}, batch_size=B, method=model.init_state)

    logits_recurrent_list = []

    # 2. Step loop
    # Note: step() outputs logits for the *next* token.
    # __call__ outputs logits for *next* token at each position.
    # So for inputs[t], step outputs prediction for inputs[t+1].
    # __call__ output at index t corresponds to prediction for inputs[t+1].

    for t in range(L):
        token_id = inputs_jax[:, t : t + 1]  # (B, 1)
        pos = jnp.array(t, dtype=jnp.int32)

        # We need to use model.step
        # The signature in model.py is: step(self, token_id, states, pos, deterministic=True)
        # But we call it via apply
        logits_step, states = model.apply(
            {"params": params},
            token_id,
            states,
            pos,
            deterministic=True,
            method=model.step,
        )
        logits_recurrent_list.append(logits_step)

    logits_recurrent = jnp.stack(logits_recurrent_list, axis=1)  # (B, L, V)

    # Compare
    print("\nComparing Logits...")

    # Check shapes
    print(f"Parallel shape: {logits_parallel.shape}")
    print(f"Recurrent shape: {logits_recurrent.shape}")

    diff = jnp.abs(logits_parallel - logits_recurrent)
    max_diff = jnp.max(diff)
    mean_diff = jnp.mean(diff)

    print(f"Max Diff: {max_diff:.6e}")
    print(f"Mean Diff: {mean_diff:.6e}")

    if max_diff < 1e-4:
        print("✅ JAX Consistency Test PASSED")
    else:
        print("❌ JAX Consistency Test FAILED")

        # Diagnose where it starts diverging
        for t in range(L):
            d_t = jnp.max(jnp.abs(logits_parallel[:, t] - logits_recurrent[:, t]))
            if d_t > 1e-4:
                print(f"Divergence starts at step {t}: max_diff={d_t:.6e}")

                # Check logits at this step
                l_par = logits_parallel[0, t]
                l_rec = logits_recurrent[0, t]
                print(f"  Parallel[0, {t}, :5]: {l_par[:5]}")
                print(f"  Recurrent[0, {t}, :5]: {l_rec[:5]}")
                break


if __name__ == "__main__":
    run_test()

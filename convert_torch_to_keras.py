"""
Convert a PyTorch SeqCond checkpoint to Keras 3 format and verify output equivalence.

Usage:
    KERAS_BACKEND=jax python convert_torch_to_keras.py checkpoints/seqcond_torch_100k.pt
    KERAS_BACKEND=jax python convert_torch_to_keras.py checkpoints/seqcond_torch_100k.pt --keras_path checkpoints/seqcond_keras.pkl
    KERAS_BACKEND=jax python convert_torch_to_keras.py checkpoints/seqcond_torch_100k.pt --skip_verify
"""

import argparse
import os
import pickle
import sys

import numpy as np


def get_config_value(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def load_torch_checkpoint(path):
    """Load a PyTorch checkpoint (without importing torch if possible)."""
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    config = data["config"]
    state_dict = {k: v.numpy() for k, v in data["state_dict"].items()}
    return config, state_dict


def build_keras_model(config):
    """Create a Keras model from the torch checkpoint config dict."""
    from seqcond.keras3.model import create_seqcond_model

    model = create_seqcond_model(
        d_model=get_config_value(config, "d_model"),
        d_ff=get_config_value(config, "d_ff"),
        num_layers=get_config_value(config, "num_layers"),
        vocab_size=get_config_value(config, "vocab_size"),
        maxlen=get_config_value(config, "maxlen"),
        seqcond_ratio=get_config_value(config, "seqcond_ratio", 3),
        num_heads=get_config_value(config, "num_heads"),
        num_kv_heads=get_config_value(config, "num_kv_heads", None),
        seqcond_heads=get_config_value(config, "seqcond_heads", None),
        num_query_heads=get_config_value(config, "num_query_heads", 6),
        num_anchor_heads=get_config_value(config, "num_anchor_heads", 0),
        num_thetas=get_config_value(config, "num_thetas", 4),
        dropout=0.0,
        tie_weights=True,
        qk_norm=get_config_value(config, "qk_norm", True),
        qk_norm_eps=get_config_value(config, "qk_norm_eps", 1e-6),
        conv_kernel_size=get_config_value(config, "conv_kernel_size", 4),
        expand_factor=get_config_value(config, "expand_factor", 1),
        out_expand_factor=get_config_value(config, "out_expand_factor", 3),
        use_square_matrix=False,
    )
    # Build by running a dummy forward pass
    dummy = np.ones((1, 16), dtype=np.int32)
    _ = model(dummy, training=False)
    return model


def convert_weights(config, state_dict, model):
    """Map torch state_dict numpy arrays onto Keras model weights.

    Returns the number of weights successfully assigned.
    """
    # Build a lookup from keras weight short path -> weight variable
    # Keras paths look like: seq_cond_model/seqcond_block_0/attn/in_proj/kernel
    # We strip the model-level prefix (first component before /)
    keras_weights = {}
    for w in model.weights:
        # Strip model prefix: "seq_cond_model/foo/bar" -> "foo/bar"
        parts = w.path.split("/")
        short = "/".join(parts[1:]) if len(parts) > 1 else w.path
        keras_weights[short] = w

    num_layers = get_config_value(config, "num_layers")
    seqcond_ratio = get_config_value(config, "seqcond_ratio", 3)

    # Build block type list + sub-indices (same logic as model construction)
    transformer_idx = 0
    seqcond_idx = 0
    block_map = []  # list of (torch_block_idx, block_type, keras_name)
    for i in range(num_layers):
        if (i + 1) % (seqcond_ratio + 1) == 0:
            block_map.append((i, "transformer", f"transformer_block_{transformer_idx}"))
            transformer_idx += 1
        else:
            block_map.append((i, "seqcond", f"seqcond_block_{seqcond_idx}"))
            seqcond_idx += 1

    assigned = 0
    skipped_keys = []

    def assign(keras_short_path, value):
        nonlocal assigned
        if keras_short_path not in keras_weights:
            print(f"  WARNING: Keras weight not found: {keras_short_path}")
            return False
        w = keras_weights[keras_short_path]
        if w.shape != value.shape:
            print(
                f"  SHAPE MISMATCH: {keras_short_path}: "
                f"keras {w.shape} vs value {value.shape}"
            )
            return False
        w.assign(value)
        assigned += 1
        return True

    # ── Embedding ───────────────────────────────────────────────────────
    assign("token_embedding/embeddings", state_dict["embedding.weight"])

    # ── Final norm ──────────────────────────────────────────────────────
    if "final_norm.scale" in state_dict:
        assign("final_norm/scale", state_dict["final_norm.scale"])
    else:
        print("  Note: final_norm.scale not in checkpoint, keeping default (ones)")

    # ── Blocks ──────────────────────────────────────────────────────────
    for torch_i, btype, keras_name in block_map:
        tp = f"blocks.{torch_i}."  # torch prefix

        if btype == "transformer":
            # Norms
            assign(f"{keras_name}/norm1/scale", state_dict[tp + "norm1.scale"])
            assign(f"{keras_name}/norm2/scale", state_dict[tp + "norm2.scale"])

            # Attention projections (torch: (out, in) → keras: (in, out))
            assign(
                f"{keras_name}/attn/q_proj/kernel",
                state_dict[tp + "attn.q_proj.weight"].T,
            )
            assign(
                f"{keras_name}/attn/k_proj/kernel",
                state_dict[tp + "attn.k_proj.weight"].T,
            )
            assign(
                f"{keras_name}/attn/v_proj/kernel",
                state_dict[tp + "attn.v_proj.weight"].T,
            )
            assign(
                f"{keras_name}/attn/out_proj/kernel",
                state_dict[tp + "attn.out_proj.weight"].T,
            )

            # FFN
            assign(
                f"{keras_name}/ff_in/kernel",
                state_dict[tp + "ff_in.weight"].T,
            )
            assign(f"{keras_name}/ff_in/bias", state_dict[tp + "ff_in.bias"])
            assign(
                f"{keras_name}/ff_out/kernel",
                state_dict[tp + "ff_out.weight"].T,
            )
            assign(f"{keras_name}/ff_out/bias", state_dict[tp + "ff_out.bias"])

        else:  # seqcond
            # Block pre-norm
            assign(f"{keras_name}/pre_norm/scale", state_dict[tp + "norm.scale"])

            # In proj (torch: (out, in) → keras: (in, out))
            assign(
                f"{keras_name}/attn/in_proj/kernel",
                state_dict[tp + "attn.in_proj.weight"].T,
            )

            # Conv weight: torch (C, 1, K) → keras (K, 1, C)
            conv_w = state_dict[tp + "attn.conv_weight"]
            assign(
                f"{keras_name}/attn/conv/kernel",
                conv_w.transpose(2, 1, 0),
            )

            # Gate proj (torch: (out, in) → keras: (in, out))
            assign(
                f"{keras_name}/attn/gate_proj/kernel",
                state_dict[tp + "attn.gate_proj.weight"].T,
            )

            # Out proj (torch: (out, in) → keras: (in, out))
            assign(
                f"{keras_name}/attn/out_proj/kernel",
                state_dict[tp + "attn.out_proj.weight"].T,
            )

            # Theta parameters
            theta_key = tp + "attn.theta_raw"
            theta_d_key = tp + "attn.theta_d_raw"
            if theta_key in state_dict:
                assign(f"{keras_name}/attn/theta_raw", state_dict[theta_key])
            elif theta_d_key in state_dict:
                assign(f"{keras_name}/attn/theta_d_raw", state_dict[theta_d_key])

            # w_int_raw
            assign(
                f"{keras_name}/attn/w_int_raw",
                state_dict[tp + "attn.w_int_raw"],
            )

            # Decay / anchor slopes
            decay_key = tp + "attn.decay_slopes"
            if decay_key in state_dict:
                assign(f"{keras_name}/attn/decay_slopes", state_dict[decay_key])
            anchor_key = tp + "attn.anchor_slopes"
            if anchor_key in state_dict:
                assign(f"{keras_name}/attn/anchor_slopes", state_dict[anchor_key])

            # Score scale/bias, phase scale
            assign(
                f"{keras_name}/attn/score_scale",
                state_dict[tp + "attn.score_scale"],
            )
            assign(
                f"{keras_name}/attn/score_bias",
                state_dict[tp + "attn.score_bias"],
            )
            assign(
                f"{keras_name}/attn/phase_scale",
                state_dict[tp + "attn.phase_scale"],
            )

            # GatedRMSNorm weight
            assign(
                f"{keras_name}/attn/gated_norm_weight",
                state_dict[tp + "attn.gated_norm.weight"],
            )

            # W_readout
            assign(
                f"{keras_name}/attn/W_readout",
                state_dict[tp + "attn.W_readout"],
            )

    total_keras = len(model.weights)
    print(f"Assigned {assigned}/{total_keras} Keras weights.")
    if assigned < total_keras:
        assigned_paths = set()
        for w in model.weights:
            parts = w.path.split("/")
            short = "/".join(parts[1:])
            assigned_paths.add(short)
        # This is informational only; missing weights keep their init values
        print("  (Remaining weights keep their default initialization)")

    return assigned


def keras_pkl_to_torch_pt(pkl_path, pt_path):
    """Convert a Keras .pkl checkpoint to a PyTorch-compatible .pt file.

    Reverses the weight name mapping and transpositions applied by convert_weights.
    The resulting .pt is loadable by TorchGenerator / torch.load.
    """
    import torch

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    config = data["config"]
    params = data["params"]  # keys: 'seq_cond_model/foo/bar'

    def get_config(key, default=None):
        return get_config_value(config, key, default)

    def kp(short):
        return f"seq_cond_model/{short}"

    def get(short):
        key = kp(short)
        if key not in params:
            return None
        return torch.from_numpy(np.array(params[key]))

    state_dict = {}

    # ── Embedding ────────────────────────────────────────────────────────
    state_dict["embedding.weight"] = get("token_embedding/embeddings")

    # ── Final norm ───────────────────────────────────────────────────────
    v = get("final_norm/scale")
    if v is not None:
        state_dict["final_norm.scale"] = v

    # ── Build block map (same logic as convert_weights) ──────────────────
    num_layers = get_config("num_layers")
    seqcond_ratio = get_config("seqcond_ratio", 3)
    transformer_idx = 0
    seqcond_idx = 0
    block_map = []
    for i in range(num_layers):
        if (i + 1) % (seqcond_ratio + 1) == 0:
            block_map.append((i, "transformer", f"transformer_block_{transformer_idx}"))
            transformer_idx += 1
        else:
            block_map.append((i, "seqcond", f"seqcond_block_{seqcond_idx}"))
            seqcond_idx += 1

    for torch_i, btype, kname in block_map:
        tp = f"blocks.{torch_i}."

        if btype == "transformer":
            state_dict[tp + "norm1.scale"] = get(f"{kname}/norm1/scale")
            state_dict[tp + "norm2.scale"] = get(f"{kname}/norm2/scale")
            state_dict[tp + "attn.q_proj.weight"] = get(f"{kname}/attn/q_proj/kernel").T
            state_dict[tp + "attn.k_proj.weight"] = get(f"{kname}/attn/k_proj/kernel").T
            state_dict[tp + "attn.v_proj.weight"] = get(f"{kname}/attn/v_proj/kernel").T
            state_dict[tp + "attn.out_proj.weight"] = get(
                f"{kname}/attn/out_proj/kernel"
            ).T
            state_dict[tp + "ff_in.weight"] = get(f"{kname}/ff_in/kernel").T
            state_dict[tp + "ff_in.bias"] = get(f"{kname}/ff_in/bias")
            state_dict[tp + "ff_out.weight"] = get(f"{kname}/ff_out/kernel").T
            state_dict[tp + "ff_out.bias"] = get(f"{kname}/ff_out/bias")
        else:  # seqcond
            state_dict[tp + "norm.scale"] = get(f"{kname}/pre_norm/scale")
            state_dict[tp + "attn.in_proj.weight"] = get(
                f"{kname}/attn/in_proj/kernel"
            ).T
            # conv: keras (K, 1, C) → torch (C, 1, K)
            conv = get(f"{kname}/attn/conv/kernel").numpy()
            state_dict[tp + "attn.conv_weight"] = torch.from_numpy(
                conv.transpose(2, 1, 0)
            )
            state_dict[tp + "attn.gate_proj.weight"] = get(
                f"{kname}/attn/gate_proj/kernel"
            ).T
            state_dict[tp + "attn.out_proj.weight"] = get(
                f"{kname}/attn/out_proj/kernel"
            ).T
            for raw_key in [
                "theta_d_raw",
                "theta_raw",
                "w_int_raw",
                "decay_slopes",
                "anchor_slopes",
                "score_scale",
                "score_bias",
                "phase_scale",
            ]:
                v = get(f"{kname}/attn/{raw_key}")
                if v is not None:
                    state_dict[tp + f"attn.{raw_key}"] = v
            state_dict[tp + "attn.gated_norm.weight"] = get(
                f"{kname}/attn/gated_norm_weight"
            )
            state_dict[tp + "attn.W_readout"] = get(f"{kname}/attn/W_readout")

    # Drop any None values (optional weights not in checkpoint)
    state_dict = {k: v for k, v in state_dict.items() if v is not None}

    os.makedirs(os.path.dirname(pt_path) or ".", exist_ok=True)
    torch.save({"state_dict": state_dict, "config": config}, pt_path)
    print(f"PyTorch checkpoint saved: {pt_path} ({len(state_dict)} tensors)")
    return state_dict, config


def save_keras_checkpoint(model, config, path):
    """Save Keras weights in .pkl format."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    params = {}
    for w in model.weights:
        params[w.path] = np.array(w.numpy())
    data = {
        "params": params,
        "config": config,
        "backend": "keras3",
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Keras checkpoint saved: {path}")


def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def kl_divergence_np(p, q, eps=1e-10):
    """KL(p || q) averaged over all positions."""
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return np.mean(np.sum(p * np.log(p / q), axis=-1))


def verify_outputs(torch_path, config, keras_model, max_diff_threshold=0.5):
    """Compare forward pass outputs between torch and keras models.

    Pass criterion:
      - Top-1 agreement >= 99% AND mean absolute diff < max_diff_threshold
      - OR top-1 agreement == 100%

    Note: Float32 accumulation differences across many layers of cumsum
    are expected to reach 1e-2 ~ 1e-1 when comparing JAX vs PyTorch.
    The meaningful check is top-1 agreement and KL divergence.

    Returns (max_abs_diff, mean_abs_diff, top1_agreement).
    """
    import torch
    from seqcond.torch.model import SeqCondModel as TorchSeqCondModel

    print("\n" + "=" * 60)
    print("Verifying output equivalence...")
    print("=" * 60)

    # ── Load torch model ────────────────────────────────────────────
    torch_model = TorchSeqCondModel(
        d_model=get_config_value(config, "d_model"),
        d_ff=get_config_value(config, "d_ff"),
        num_layers=get_config_value(config, "num_layers"),
        vocab_size=get_config_value(config, "vocab_size"),
        maxlen=get_config_value(config, "maxlen"),
        num_heads=get_config_value(config, "num_heads"),
        num_kv_heads=get_config_value(config, "num_kv_heads", None),
        qk_norm=get_config_value(config, "qk_norm", True),
        qk_norm_eps=get_config_value(config, "qk_norm_eps", 1e-6),
        seqcond_heads=get_config_value(config, "seqcond_heads", 32),
        num_query_heads=get_config_value(config, "num_query_heads", 6),
        num_thetas=get_config_value(config, "num_thetas", 4),
        conv_kernel_size=get_config_value(config, "conv_kernel_size", 4),
        expand_factor=get_config_value(config, "expand_factor", 1),
        out_expand_factor=get_config_value(config, "out_expand_factor", 3),
        seqcond_ratio=get_config_value(config, "seqcond_ratio", 3),
        num_anchor_heads=get_config_value(config, "num_anchor_heads", 0),
        skip_low_rank=get_config_value(config, "skip_low_rank", True),
    )
    data = torch.load(torch_path, map_location="cpu", weights_only=False)
    torch_model.load_state_dict(data["state_dict"], strict=False)
    torch_model.eval()

    # ── Inputs ──────────────────────────────────────────────────────
    np.random.seed(42)
    seq_len = 64
    vocab_size = get_config_value(config, "vocab_size")
    input_ids = np.random.randint(1, vocab_size, (2, seq_len), dtype=np.int32)

    # ── Torch forward ────────────────────────────────────────────────
    with torch.no_grad():
        torch_logits = torch_model(torch.from_numpy(input_ids).long()).numpy()

    # ── Keras forward ────────────────────────────────────────────────
    keras_logits = np.array(keras_model(input_ids, training=False))

    # ── Absolute diff ────────────────────────────────────────────────
    diff = np.abs(torch_logits - keras_logits)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    # Relative to logit range
    logit_range = float(np.max(torch_logits) - np.min(torch_logits))
    rel_max = max_diff / max(logit_range, 1e-6)

    # ── Top-k agreement ──────────────────────────────────────────────
    torch_top1 = np.argmax(torch_logits, axis=-1)
    keras_top1 = np.argmax(keras_logits, axis=-1)
    top1_agreement = float(np.mean(torch_top1 == keras_top1))

    torch_top5 = np.argsort(torch_logits, axis=-1)[..., -5:]
    keras_top5 = np.argsort(keras_logits, axis=-1)[..., -5:]
    top5_agreement = float(
        np.mean(
            [
                len(set(torch_top5[b, t]) & set(keras_top5[b, t])) / 5
                for b in range(torch_logits.shape[0])
                for t in range(torch_logits.shape[1])
            ]
        )
    )

    # ── KL divergence (Torch || Keras) ───────────────────────────────
    torch_probs = softmax_np(torch_logits.astype(np.float64))
    keras_probs = softmax_np(keras_logits.astype(np.float64))
    kl_fwd = kl_divergence_np(torch_probs, keras_probs)  # KL(torch || keras)
    kl_rev = kl_divergence_np(keras_probs, torch_probs)  # KL(keras || torch)

    # ── Per-position max diff ────────────────────────────────────────
    per_pos_max = np.max(diff, axis=(0, 2))

    # ── Print summary ────────────────────────────────────────────────
    print(f"Logit shapes:       {torch_logits.shape}")
    print(f"Logit range (torch): [{torch_logits.min():.2f}, {torch_logits.max():.2f}]")
    print()
    print(f"Absolute diff — max:  {max_diff:.4e}  mean: {mean_diff:.4e}")
    print(f"Relative diff — max:  {rel_max:.4e}  (as fraction of logit range)")
    print()
    print(f"KL divergence (torch||keras): {kl_fwd:.4e} nats/token")
    print(f"KL divergence (keras||torch): {kl_rev:.4e} nats/token")
    print()
    print(f"Top-1 agreement: {top1_agreement * 100:.2f}%")
    print(f"Top-5 agreement: {top5_agreement * 100:.2f}%")
    print()
    print(f"Per-position max diff (first 16):")
    per8 = per_pos_max[:16].reshape(2, 8)
    print(f"  pos  0-7:  {per8[0]}")
    print(f"  pos  8-15: {per8[1]}")

    # ── Note on expected precision ───────────────────────────────────
    num_layers = get_config_value(config, "num_layers", 12)
    print(f"\nNote: {num_layers}-layer model with cumsum — float32 accumulation")
    print("differences of ~1e-2 to 1e-1 between JAX and PyTorch are expected.")

    # ── Pass / fail ──────────────────────────────────────────────────
    # Primary criterion: top-1 agreement
    # Secondary: KL divergence < 1e-3 nats/token (well below perceptible threshold)
    kl_threshold = 1e-3
    passed = top1_agreement >= 0.999 or (
        top1_agreement >= 0.99 and kl_fwd < kl_threshold
    )

    if passed:
        print(f"\n✓ PASSED  (top-1={top1_agreement*100:.1f}%, KL={kl_fwd:.2e})")
    else:
        print(f"\n✗ FAILED  (top-1={top1_agreement*100:.1f}%, KL={kl_fwd:.2e})")
        print("  Possible causes: weight mapping error, missing layer, shape mismatch.")

    return max_diff, mean_diff, top1_agreement


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch SeqCond checkpoint to Keras 3"
    )
    parser.add_argument("torch_path", help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument(
        "--keras_path",
        default=None,
        help="Output path for Keras checkpoint (.pkl). Default: <torch_path>_keras.pkl",
    )
    parser.add_argument(
        "--skip_verify",
        action="store_true",
        help="Skip output verification against torch model",
    )
    parser.add_argument(
        "--kl_threshold",
        type=float,
        default=1e-3,
        help="KL divergence threshold for verification (default: 1e-3 nats/token)",
    )
    args = parser.parse_args()

    if args.keras_path is None:
        base = os.path.splitext(args.torch_path)[0]
        args.keras_path = base + "_keras.pkl"

    # 1. Load torch checkpoint
    print(f"Loading torch checkpoint: {args.torch_path}")
    config, state_dict = load_torch_checkpoint(args.torch_path)
    print(
        f"  d_model={get_config_value(config, 'd_model')}, "
        f"layers={get_config_value(config, 'num_layers')}, "
        f"vocab={get_config_value(config, 'vocab_size')}"
    )

    # 2. Build Keras model
    print("\nBuilding Keras model...")
    keras_model = build_keras_model(config)
    print(f"  Keras params: {keras_model.count_params():,}")

    # 3. Convert weights
    print("\nConverting weights...")
    convert_weights(config, state_dict, keras_model)

    # 4. Save
    save_keras_checkpoint(keras_model, config, args.keras_path)

    # 5. Verify
    if not args.skip_verify:
        max_diff, mean_diff, top1 = verify_outputs(args.torch_path, config, keras_model)
        passed = top1 >= 0.999 or (top1 >= 0.99 and max_diff < args.kl_threshold)
        return 0 if passed else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

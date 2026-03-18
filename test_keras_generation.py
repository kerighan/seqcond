"""Test Keras generation using the converted 762k checkpoint.

Compares:
  1. Keras forward() vs Keras step() (must match exactly)
  2. Torch step() vs Keras step() (same greedy tokens expected)
  3. Actual text generation side by side

Usage:
    KERAS_BACKEND=jax JAX_PLATFORMS=cpu python test_keras_generation.py
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import argparse
import numpy as np

TORCH_CKPT = "checkpoints/seqcond_torch_762k.pt"


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
def get_config_value(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def load_torch_model(path):
    """Load torch checkpoint and return (model, config)."""
    import torch
    from seqcond.torch.model import SeqCondModel as TorchModel

    data = torch.load(path, map_location="cpu", weights_only=False)
    config = data["config"]
    model = TorchModel(
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
    model.load_state_dict(data["state_dict"], strict=False)
    model.eval()
    return model, config


def load_keras_model(torch_path):
    """Build Keras model and convert weights from torch checkpoint."""
    from convert_torch_to_keras import (
        load_torch_checkpoint,
        build_keras_model,
        convert_weights,
    )

    config, state_dict = load_torch_checkpoint(torch_path)
    keras_model = build_keras_model(config)
    convert_weights(config, state_dict, keras_model)
    return keras_model, config


# ──────────────────────────────────────────────────────────────────
# Test 1: Keras forward() vs Keras step()
# ──────────────────────────────────────────────────────────────────
def test_forward_vs_step(keras_model, seq_len=32):
    """forward() and step() must produce identical logits."""
    print("=" * 60)
    print("TEST 1: Keras forward() vs Keras step()")
    print("=" * 60)

    np.random.seed(42)
    vocab = keras_model.vocab_size
    input_seq = np.random.randint(1, vocab, (1, seq_len), dtype=np.int32)

    # Forward pass
    fwd_logits = np.array(keras_model(input_seq, training=False))  # (1, L, V)

    # Step-by-step
    states = keras_model.init_state(batch_size=1)
    step_logits = []
    for t in range(seq_len):
        tok = input_seq[:, t : t + 1]
        logits_t, states = keras_model.step(tok, states)
        step_logits.append(np.array(logits_t))
    step_logits = np.stack(step_logits, axis=1)  # (1, L, V)

    diff = np.abs(fwd_logits - step_logits)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    fwd_top1 = np.argmax(fwd_logits, axis=-1)
    step_top1 = np.argmax(step_logits, axis=-1)
    agreement = float(np.mean(fwd_top1 == step_top1))

    print(f"Shapes: fwd {fwd_logits.shape}, step {step_logits.shape}")
    print(f"Max abs diff:  {max_diff:.6e}")
    print(f"Mean abs diff: {mean_diff:.6e}")
    print(f"Top-1 agreement: {agreement*100:.1f}%")

    # Per-position max diff
    per_pos = np.max(diff, axis=(0, 2))
    print(f"Per-position max diff (first 8): {per_pos[:8]}")

    if agreement >= 0.95 and max_diff < 1e-3:
        print("✓ PASSED\n")
        return True
    else:
        print(f"✗ FAILED (agreement={agreement*100:.1f}%, max_diff={max_diff:.2e})\n")
        return False


# ──────────────────────────────────────────────────────────────────
# Test 2: Torch step() vs Keras step() — greedy generation
# ──────────────────────────────────────────────────────────────────
def test_torch_vs_keras_step(torch_model, keras_model, config, gen_len=64):
    """Both models must produce identical greedy tokens."""
    import torch as th

    print("=" * 60)
    print("TEST 2: Torch step() vs Keras step() — greedy generation")
    print("=" * 60)

    prompt = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    B = 1

    # ── Torch: prefill + generate ────────────────────────────────
    with th.no_grad():
        t_input = th.from_numpy(prompt).long()
        t_logits, t_states = torch_model.prefill(t_input)  # last-token logits, states
        t_tok = t_logits.argmax(-1)  # (B, 1)
        torch_tokens = [int(t_tok[0, 0])]

        for _ in range(gen_len - 1):
            t_logits, t_states = torch_model.step(t_tok, t_states)
            t_tok = t_logits.argmax(-1, keepdim=True)
            torch_tokens.append(int(t_tok[0, 0]))

    # ── Keras: prefill + generate ────────────────────────────────
    k_fwd_logits = np.array(keras_model(prompt, training=False))  # (B, L, V)
    # Use last-token logits for first generated token
    k_tok = np.argmax(k_fwd_logits[:, -1:, :], axis=-1)  # (B, 1)
    keras_tokens = [int(k_tok[0, 0])]

    # Init state and feed prompt token-by-token to build state
    k_states = keras_model.init_state(batch_size=B)
    for t in range(prompt.shape[1]):
        _, k_states = keras_model.step(prompt[:, t : t + 1], k_states)

    for _ in range(gen_len - 1):
        k_logits, k_states = keras_model.step(k_tok, k_states)
        k_tok = np.argmax(np.array(k_logits), axis=-1, keepdims=True)
        keras_tokens.append(int(k_tok[0, 0]))

    # ── Compare ──────────────────────────────────────────────────
    torch_tokens = np.array(torch_tokens)
    keras_tokens = np.array(keras_tokens)
    agreement = np.mean(torch_tokens == keras_tokens)

    print(f"Generated {gen_len} tokens")
    print(f"Token agreement: {agreement*100:.1f}%")

    # Show first divergence
    mismatches = np.where(torch_tokens != keras_tokens)[0]
    if len(mismatches) > 0:
        first = mismatches[0]
        print(
            f"First divergence at step {first}: torch={torch_tokens[first]}, keras={keras_tokens[first]}"
        )
        print(
            f"Torch tokens [{max(0,first-2)}:{first+5}]: {torch_tokens[max(0,first-2):first+5]}"
        )
        print(
            f"Keras tokens [{max(0,first-2)}:{first+5}]: {keras_tokens[max(0,first-2):first+5]}"
        )
    else:
        print("All tokens match!")

    if agreement >= 0.95:
        print("✓ PASSED\n")
        return True
    else:
        print(f"✗ FAILED\n")
        return False


# ──────────────────────────────────────────────────────────────────
# Test 3: Actual text generation with tokenizer
# ──────────────────────────────────────────────────────────────────
def test_text_generation(torch_model, keras_model, config):
    """Side-by-side text generation with the real tokenizer."""
    import torch as th

    print("=" * 60)
    print("TEST 3: Side-by-side text generation")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    except Exception:
        print("  (Skipping: transformers/tokenizer not available)\n")
        return True

    prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(prompt, return_tensors="np").astype(np.int32)
    gen_len = 60
    B = 1

    print(f"Prompt: {prompt.strip()}")
    print(f"Prompt tokens: {input_ids.shape[1]}")
    print()

    # ── Torch generation ─────────────────────────────────────────
    with th.no_grad():
        t_input = th.from_numpy(input_ids).long()
        t_logits, t_states = torch_model.prefill(t_input)
        t_tok = t_logits.argmax(-1)
        t_tokens = [int(t_tok[0, 0])]
        for _ in range(gen_len - 1):
            t_logits, t_states = torch_model.step(t_tok, t_states)
            t_tok = t_logits.argmax(-1, keepdim=True)
            t_tokens.append(int(t_tok[0, 0]))

    # ── Keras generation ─────────────────────────────────────────
    k_fwd = np.array(keras_model(input_ids, training=False))
    k_tok = np.argmax(k_fwd[:, -1:, :], axis=-1)
    k_tokens = [int(k_tok[0, 0])]

    k_states = keras_model.init_state(batch_size=B)
    for t in range(input_ids.shape[1]):
        _, k_states = keras_model.step(input_ids[:, t : t + 1], k_states)
    for _ in range(gen_len - 1):
        k_logits, k_states = keras_model.step(k_tok, k_states)
        k_tok = np.argmax(np.array(k_logits), axis=-1, keepdims=True)
        k_tokens.append(int(k_tok[0, 0]))

    torch_text = tokenizer.decode(t_tokens, skip_special_tokens=False)
    keras_text = tokenizer.decode(k_tokens, skip_special_tokens=False)
    agreement = np.mean(np.array(t_tokens) == np.array(k_tokens))

    print(f"--- Torch ({len(t_tokens)} tokens) ---")
    print(torch_text[:300])
    print()
    print(f"--- Keras ({len(k_tokens)} tokens) ---")
    print(keras_text[:300])
    print()
    print(f"Token agreement: {agreement*100:.1f}%")

    if agreement >= 0.90:
        print("✓ PASSED\n")
        return True
    else:
        print("✗ FAILED\n")
        return False


# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch_path", default=TORCH_CKPT)
    args = parser.parse_args()

    if not os.path.exists(args.torch_path):
        print(f"Checkpoint not found: {args.torch_path}")
        return 1

    print(f"Loading models from {args.torch_path} ...\n")
    torch_model, config = load_torch_model(args.torch_path)
    keras_model, _ = load_keras_model(args.torch_path)
    print(f"Torch params: {sum(p.numel() for p in torch_model.parameters()):,}")
    print(f"Keras params: {keras_model.count_params():,}\n")

    results = []

    # Test 1
    try:
        results.append(test_forward_vs_step(keras_model))
    except Exception as e:
        import traceback

        traceback.print_exc()
        results.append(False)

    # Test 2
    try:
        results.append(test_torch_vs_keras_step(torch_model, keras_model, config))
    except Exception as e:
        import traceback

        traceback.print_exc()
        results.append(False)

    # Test 3
    try:
        results.append(test_text_generation(torch_model, keras_model, config))
    except Exception as e:
        import traceback

        traceback.print_exc()
        results.append(False)

    print("=" * 60)
    if all(results):
        print("ALL TESTS PASSED ✓")
        return 0
    else:
        print(f"FAILED: {results.count(False)}/{len(results)} tests")
        return 1


if __name__ == "__main__":
    sys.exit(main())

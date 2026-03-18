"""Quick JAX generation test — uses forward pass (no step needed).

Tests if the JAX model itself rambles or if the issue is Torch-specific.

Usage:
    JAX_PLATFORMS=cpu python test_jax_generation.py
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
from seqcond.jax.model import SeqCondModel
from seqcond.dataset import Tokenizer

JAX_CKPT = "/tmp/seqcond_jax_762k.pkl"


def load_jax_model(path):
    """Load JAX model from pickle checkpoint."""
    print(f"Loading JAX checkpoint: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)

    config = data["config"]
    params = data["params"]

    print(f"  d_model={config['d_model']}, num_layers={config['num_layers']}")

    model = SeqCondModel(
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        vocab_size=config["vocab_size"],
        maxlen=config["maxlen"],
        seqcond_ratio=config.get("seqcond_ratio", 3),
        num_heads=config["num_heads"],
        num_kv_heads=config.get("num_kv_heads", None),
        seqcond_heads=config.get("seqcond_heads", None),
        num_query_heads=config.get("num_query_heads", 6),
        num_anchor_heads=config.get("num_anchor_heads", 0),
        num_thetas=config.get("num_thetas", 4),
        conv_kernel_size=config.get("conv_kernel_size", 4),
        expand_factor=config.get("expand_factor", 1),
        out_expand_factor=config.get("out_expand_factor", 3),
        qk_norm=config.get("qk_norm", True),
        qk_norm_eps=config.get("qk_norm_eps", 1e-6),
        remat=False,
        use_square_matrix=config.get("use_square_matrix", False),
    )

    variables = {"params": params}
    return model, variables, config


def generate_jax_forward(model, variables, tokenizer, prompt_text, max_tokens=50, temperature=0.0):
    """Generate text using repeated forward passes (O(L^2) but no step() needed)."""
    token_ids = tokenizer.encode(prompt_text)
    eos_id = tokenizer.encode("<|im_end|>")[0]

    # Start with prompt
    all_ids = list(token_ids)
    generated = []

    for i in range(max_tokens):
        input_ids = jnp.array([all_ids], dtype=jnp.int32)
        logits = model.apply(variables, input_ids)  # (1, L, V)
        last_logits = np.array(logits[0, -1, :])

        if temperature <= 0:
            next_tok = int(np.argmax(last_logits))
        else:
            logits_scaled = last_logits / temperature
            logits_scaled -= np.max(logits_scaled)
            probs = np.exp(logits_scaled)
            probs = probs / probs.sum()
            next_tok = int(np.random.choice(len(probs), p=probs))

        generated.append(next_tok)
        all_ids.append(next_tok)

        # Print token as it's generated
        tok_text = tokenizer.decode([next_tok])
        print(tok_text, end="", flush=True)

        if next_tok == eos_id:
            break

    print()
    return generated


def main():
    model, variables, config = load_jax_model(JAX_CKPT)
    tokenizer = Tokenizer()

    prompts = [
        "What is 2+2?",
        "What is the meaning of life?",
    ]

    for question in prompts:
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        prompt_toks = tokenizer.encode(prompt)
        print("=" * 60)
        print(f"Q: {question}")
        print(f"Prompt: {len(prompt_toks)} tokens")
        print(f"Generating (greedy, max {50} tokens)...")
        print()

        t0 = time.time()
        generated = generate_jax_forward(
            model, variables, tokenizer, prompt,
            max_tokens=50, temperature=0.0,
        )
        dt = time.time() - t0

        print(f"\n[{len(generated)} tokens in {dt:.1f}s, {len(generated)/dt:.2f} tok/s]")
        print()


if __name__ == "__main__":
    main()

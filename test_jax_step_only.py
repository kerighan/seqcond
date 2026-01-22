"""
Test JAX generation using only step() method for comparison with PyTorch.
"""

import jax
import jax.numpy as jnp
import pickle
from seqcond.jax.model import SeqCondModel
from seqcond.dataset import Tokenizer

checkpoint_path = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step50000.pkl"

print(f"Loading checkpoint from {checkpoint_path}...")
with open(checkpoint_path, "rb") as f:
    data = pickle.load(f)

params = data["params"]
config_dict = data["config"]["model"]

model = SeqCondModel(
    vocab_size=config_dict["vocab_size"],
    d_model=config_dict["d_model"],
    d_ff=config_dict["d_ff"],
    num_layers=config_dict["num_layers"],
    num_heads=config_dict["num_heads"],
    num_kv_heads=config_dict.get("num_kv_heads"),
    maxlen=config_dict["maxlen"],
    dropout=0.0,
    tie_weights=config_dict.get("tie_weights", True),
    qk_norm=config_dict.get("qk_norm", False),
    qk_norm_eps=config_dict.get("qk_norm_eps", 1e-6),
    seqcond_heads=config_dict.get("seqcond_heads"),
    num_query_heads=config_dict.get("num_query_heads"),
    num_thetas=config_dict.get("num_thetas", 4),
    num_anchor_heads=config_dict.get("num_anchor_heads", 0),
    conv_kernel_size=config_dict.get("conv_kernel_size", 4),
    expand_factor=config_dict.get("expand_factor", 2),
    out_expand_factor=config_dict.get("out_expand_factor", 2),
    use_positional_embedding=config_dict.get("use_positional_embedding", False),
    seqcond_ratio=config_dict.get("seqcond_ratio", 3),
    use_square_matrix=config_dict.get("use_square_matrix", False),
    remat=False,
)

tokenizer = Tokenizer()
prompt = "The quick brown fox"
tokens = tokenizer.encode(prompt)

print(f"Prompt: '{prompt}'")
print(f"Tokens: {tokens}")

# Initialize states
states = model.apply(
    {"params": params}, 1, method=lambda m, batch_size: m.init_state(batch_size)
)

# Process prompt tokens one by one using step()
print("\nProcessing prompt with step()...")
for i, token_id in enumerate(tokens):
    token_array = jnp.array([[token_id]], dtype=jnp.int32)
    logits, states = model.apply(
        {"params": params},
        token_array,
        states,
        i,
        deterministic=True,
        method=model.step,
    )
    print(f"  Token {i}: {token_id}, logits shape: {logits.shape}")

# Generate new tokens
print("\nGenerating new tokens...")
generated = []
for step in range(50):
    # Sample next token (greedy)
    next_token = int(jnp.argmax(logits[0]))
    generated.append(next_token)

    if step < 5:
        print(f"  Step {step}: token={next_token}, logits[:5]={logits[0, :5]}")

    # Step with new token
    token_array = jnp.array([[next_token]], dtype=jnp.int32)
    logits, states = model.apply(
        {"params": params},
        token_array,
        states,
        len(tokens) + step,
        deterministic=True,
        method=model.step,
    )

# Decode
full_tokens = tokens + generated
text = tokenizer.decode(full_tokens)

print(f"\nGenerated text:\n{text}")

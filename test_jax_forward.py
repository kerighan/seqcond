"""
Test JAX generation using forward (apply) only.
"""

import jax
import jax.numpy as jnp
import pickle
from seqcond.jax.model import SeqCondModel
from seqcond.dataset import Tokenizer

checkpoint_path = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step60000.pkl"

print("Loading JAX checkpoint...")
with open(checkpoint_path, "rb") as f:
    data = pickle.load(f)

params = data["params"]
config_dict = data["config"]["model"]

model = SeqCondModel(
    **{k: v for k, v in config_dict.items() if k not in ["model_type", "state_size"]}
)

tokenizer = Tokenizer()
prompt = "NAD+ is"
tokens = tokenizer([prompt])[0]

print(f"Prompt: '{prompt}'")
print(f"Tokens: {tokens}")

# Forward pass
input_ids = jnp.array([tokens])
logits = model.apply({"params": params}, input_ids, deterministic=True)

# Get prediction for last token
last_logits = logits[0, -1, :]
predicted_token = int(jnp.argmax(last_logits))

print(f"\nPredicted next token: {predicted_token}")
print(f"Token text: '{tokenizer.decode([predicted_token])}'")
print(f"Top 5 logits: {jnp.sort(last_logits)[-5:]}")
print(f"Top 5 tokens: {jnp.argsort(last_logits)[-5:]}")

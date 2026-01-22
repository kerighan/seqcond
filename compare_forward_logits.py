"""
Compare logits from JAX and PyTorch forward passes.
"""

import jax
import jax.numpy as jnp
import torch
import pickle
import numpy as np
from seqcond.jax.model import SeqCondModel as JAXModel
from seqcond.torch.generator import TorchGenerator
from seqcond.dataset import Tokenizer

# Load JAX model
jax_checkpoint = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step60000.pkl"
print(f"Loading JAX checkpoint...")
with open(jax_checkpoint, "rb") as f:
    data = pickle.load(f)

jax_params = data["params"]
config_dict = data["config"]["model"]

jax_model = JAXModel(
    **{k: v for k, v in config_dict.items() if k not in ["model_type", "state_size"]}
)

# Load PyTorch model
torch_checkpoint = "checkpoints/seqcond_torch_60k.pt"
print(f"Loading PyTorch checkpoint...")
torch_gen = TorchGenerator(torch_checkpoint)

# Test prompt
tokenizer = Tokenizer()
prompt = "NAD+ is"
tokens = tokenizer([prompt])[0]
print(f"\nPrompt: '{prompt}'")
print(f"Tokens: {tokens}")

# JAX forward
jax_input = jnp.array([tokens])
jax_logits = jax_model.apply({"params": jax_params}, jax_input, deterministic=True)
jax_logits_np = np.array(jax_logits[0, -1, :])

# PyTorch forward
torch_input = torch.tensor([tokens], dtype=torch.long).to("cuda")
with torch.no_grad():
    torch_logits, _ = torch_gen.model(torch_input)
torch_logits_np = torch_logits[0, -1, :].detach().cpu().numpy()

# Compare
diff = np.abs(jax_logits_np - torch_logits_np)
print(f"\n=== Forward Pass Comparison ===")
print(f"Max diff: {np.max(diff):.6e}")
print(f"Mean diff: {np.mean(diff):.6e}")
print(
    f"Cosine similarity: {np.dot(jax_logits_np, torch_logits_np) / (np.linalg.norm(jax_logits_np) * np.linalg.norm(torch_logits_np)):.6f}"
)

jax_top = int(np.argmax(jax_logits_np))
torch_top = int(np.argmax(torch_logits_np))

print(f"\nJAX top token: {jax_top} ('{tokenizer.decode([jax_top])}')")
print(f"PyTorch top token: {torch_top} ('{tokenizer.decode([torch_top])}')")
print(f"Match: {jax_top == torch_top}")

print(f"\nJAX logits[:10]: {jax_logits_np[:10]}")
print(f"PyTorch logits[:10]: {torch_logits_np[:10]}")

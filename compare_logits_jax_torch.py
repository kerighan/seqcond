"""
Compare logits from JAX and PyTorch for the same prompt.
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
prompt = "The quick"
tokens = tokenizer.encode(prompt)
print(f"\nPrompt: '{prompt}'")
print(f"Tokens: {tokens}")

# JAX: Process with step()
print("\n=== JAX ===")
jax_states = jax_model.apply(
    {"params": jax_params}, 1, method=lambda m, batch_size: m.init_state(batch_size)
)

for i, token_id in enumerate(tokens):
    token_array = jnp.array([[token_id]], dtype=jnp.int32)
    jax_logits, jax_states = jax_model.apply(
        {"params": jax_params},
        token_array,
        jax_states,
        i,
        deterministic=True,
        method=jax_model.step,
    )
    top_token = int(jnp.argmax(jax_logits[0]))
    print(
        f"  Token {i} (id={token_id}): top_token={top_token}, logits[:5]={jax_logits[0, :5]}"
    )

# PyTorch: Process with step()
print("\n=== PyTorch ===")
torch_states = torch_gen.model.init_state(batch_size=1, device="cuda")

for i, token_id in enumerate(tokens):
    token_tensor = torch.tensor([[token_id]], device="cuda")
    pos_tensor = torch.tensor([i], device="cuda")
    torch_logits, torch_states = torch_gen.model.step(
        token_tensor, torch_states, pos=pos_tensor
    )
    top_token = torch.argmax(torch_logits[0]).item()
    print(
        f"  Token {i} (id={token_id}): top_token={top_token}, logits[:5]={torch_logits[0, :5]}"
    )

# Compare final logits
print("\n=== Comparison ===")
jax_logits_np = np.array(jax_logits[0])
torch_logits_np = torch_logits[0].detach().cpu().numpy()

diff = np.abs(jax_logits_np - torch_logits_np)
print(f"Max diff: {np.max(diff):.6e}")
print(f"Mean diff: {np.mean(diff):.6e}")

# Check if argmax matches
jax_top = int(jnp.argmax(jax_logits[0]))
torch_top = torch.argmax(torch_logits[0]).item()
print(f"\nJAX top token: {jax_top}")
print(f"PyTorch top token: {torch_top}")
print(f"Match: {jax_top == torch_top}")

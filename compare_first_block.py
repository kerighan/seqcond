"""
Compare first block output between JAX and PyTorch.
"""

import jax
import jax.numpy as jnp
import torch
import pickle
import numpy as np
from seqcond.jax.model import SeqCondModel as JAXModel
from seqcond.torch.model import SeqCondModel as TorchModel
from seqcond.config import ModelConfig

# Load checkpoints
jax_checkpoint = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step60000.pkl"
with open(jax_checkpoint, "rb") as f:
    jax_data = pickle.load(f)

jax_params = jax_data["params"]
config_dict = jax_data["config"]["model"]

# Create models
jax_model = JAXModel(
    **{k: v for k, v in config_dict.items() if k not in ["model_type", "state_size"]}
)

torch_checkpoint = torch.load("checkpoints/seqcond_torch_60k.pt", map_location="cpu")
torch_config = ModelConfig(**torch_checkpoint["config"]["model"])
torch_model = TorchModel(torch_config)
torch_model.load_state_dict(torch_checkpoint["state_dict"])
torch_model.eval()

# Test input
token_id = 792  # "The"
token_jax = jnp.array([[token_id]], dtype=jnp.int32)
token_torch = torch.tensor([[token_id]])

# JAX: Get embedding
jax_emb = jax_model.apply(
    {"params": jax_params}, token_jax, method=lambda m, x: m.embedding(x)
)
print(f"JAX embedding shape: {jax_emb.shape}")
print(f"JAX embedding[0, 0, :5]: {jax_emb[0, 0, :5]}")

# PyTorch: Get embedding
torch_emb = torch_model.embedding(token_torch)
print(f"\nPyTorch embedding shape: {torch_emb.shape}")
print(f"PyTorch embedding[0, 0, :5]: {torch_emb[0, 0, :5]}")

# Compare embeddings
emb_diff = np.abs(np.array(jax_emb[0, 0, :]) - torch_emb[0, 0, :].detach().numpy())
print(f"\nEmbedding max diff: {np.max(emb_diff):.6e}")

# JAX: Initialize first block state
jax_states = jax_model.apply(
    {"params": jax_params}, 1, method=lambda m, batch_size: m.init_state(batch_size)
)
jax_state_0 = jax_states[0]

# PyTorch: Initialize first block state
torch_states = torch_model.init_state(batch_size=1, device="cpu")
torch_state_0 = torch_states[0]

print(f"\nJAX state_0 pos: {jax_state_0[3]}")
print(f"PyTorch state_0 pos: {torch_state_0[3]}")

# JAX: Run first block
jax_block = jax_model.blocks[0][1]  # (block_type, block)
jax_x = jax_emb[:, 0, :]  # (B, D)
jax_out, jax_new_state = jax_model.apply(
    {"params": jax_params},
    jax_x,
    jax_state_0,
    True,
    method=lambda m, x, s, d: m.blocks[0][1].step(x, s, deterministic=d),
)

print(f"\nJAX block 0 output shape: {jax_out.shape}")
print(f"JAX block 0 output[:5]: {jax_out[0, :5]}")

# PyTorch: Run first block
torch_block = torch_model.blocks[0]
torch_x = torch_emb[:, 0, :]  # (B, D)
torch_out, torch_new_state = torch_block.step(torch_x, torch_state_0)

print(f"\nPyTorch block 0 output shape: {torch_out.shape}")
print(f"PyTorch block 0 output[:5]: {torch_out[0, :5]}")

# Compare
block_diff = np.abs(np.array(jax_out[0, :]) - torch_out[0, :].detach().numpy())
print(f"\nBlock 0 output max diff: {np.max(block_diff):.6e}")
print(f"Block 0 output mean diff: {np.mean(block_diff):.6e}")

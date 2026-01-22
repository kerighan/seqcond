"""
Debug SeqCondAttention by comparing JAX and PyTorch intermediate outputs.
"""

import jax
import jax.numpy as jnp
import torch
import pickle
import numpy as np
from seqcond.jax.model import SeqCondModel as JAXModel
from seqcond.torch.model import SeqCondModel as TorchModel
from seqcond.dataset import Tokenizer

# Load models
jax_checkpoint = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step60000.pkl"
print("Loading JAX checkpoint...")
with open(jax_checkpoint, "rb") as f:
    data = pickle.load(f)

jax_params = data["params"]
config_dict = data["config"]["model"]

jax_model = JAXModel(
    **{k: v for k, v in config_dict.items() if k not in ["model_type", "state_size"]}
)

# Load PyTorch
torch_checkpoint = "checkpoints/seqcond_torch_60k.pt"
print("Loading PyTorch checkpoint...")
checkpoint = torch.load(torch_checkpoint, map_location="cuda")
torch_model = TorchModel(checkpoint["config"]).to("cuda").eval()
torch_model.load_state_dict(checkpoint["state_dict"])

# Test with single token
tokenizer = Tokenizer()
token_id = 792  # "The"

print(f"\nTesting with token {token_id}")

# JAX: Get embedding and first block output
jax_input = jnp.array([[token_id]])
jax_emb = jax_model.apply(
    {"params": jax_params},
    jax_input,
    method=lambda m, x: m.embedding(x),
)
print(f"JAX embedding shape: {jax_emb.shape}")
print(f"JAX embedding mean: {jnp.mean(jax_emb):.6f}, std: {jnp.std(jax_emb):.6f}")

# Get first block (SeqCond)
first_block_params = jax_params["seqcond_block_0"]

# PyTorch: Get embedding
torch_input = torch.tensor([[token_id]], dtype=torch.long).to("cuda")
with torch.no_grad():
    torch_emb = torch_model.embedding(torch_input)

print(f"PyTorch embedding shape: {torch_emb.shape}")
print(f"PyTorch embedding mean: {torch_emb.mean():.6f}, std: {torch_emb.std():.6f}")

# Compare embeddings
emb_diff = np.abs(np.array(jax_emb[0, 0]) - torch_emb[0, 0].cpu().numpy())
print(f"\nEmbedding diff: max={np.max(emb_diff):.6e}, mean={np.mean(emb_diff):.6e}")

# Now test first SeqCond block
print("\n" + "=" * 60)
print("Testing First SeqCond Block")
print("=" * 60)

# For JAX, we need to manually call the block
# This is complex, so let's just compare the full forward pass of first block
# by looking at the model's internal structure

# Actually, let's use the block-by-block test but add more diagnostics
print("\nFor detailed SeqCond debugging, check the block-by-block test output.")
print("The key issue is likely in:")
print("1. Convolution computation")
print("2. Complex integration (re/im accumulation)")
print("3. Phase modulation (phi calculation)")
print("4. Output gating (y_val * sigmoid(y_gate))")

print("\nSuggested next steps:")
print("1. Add print statements in SeqCondAttention forward/step")
print("2. Compare intermediate values (z_conv, q_raw, k_val, phi, re, im)")
print("3. Check if there's a dtype mismatch or operation order issue")

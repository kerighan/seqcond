"""
Check if weights are correctly converted from JAX to PyTorch.
"""

import pickle
import torch
import numpy as np

# Load JAX checkpoint
jax_checkpoint = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step50000.pkl"
with open(jax_checkpoint, "rb") as f:
    jax_data = pickle.load(f)

jax_params = jax_data["params"]

# Load PyTorch checkpoint
torch_checkpoint = "checkpoints/seqcond_torch_50k.pt"
torch_data = torch.load(torch_checkpoint, map_location="cpu")
torch_state_dict = torch_data["state_dict"]

print("=== Checking Embedding Weights ===")
jax_emb = np.array(jax_params["token_embedding"]["embedding"])
torch_emb = torch_state_dict["embedding.weight"].numpy()

print(f"JAX embedding shape: {jax_emb.shape}")
print(f"PyTorch embedding shape: {torch_emb.shape}")
print(f"JAX embedding[0, :5]: {jax_emb[0, :5]}")
print(f"PyTorch embedding[0, :5]: {torch_emb[0, :5]}")

emb_diff = np.abs(jax_emb - torch_emb)
print(f"Max diff: {np.max(emb_diff):.6e}")
print(f"Mean diff: {np.mean(emb_diff):.6e}")

print("\n=== Checking Output Projection Weights ===")
if "output_projection" in jax_params:
    jax_out = np.array(jax_params["output_projection"]["kernel"])
    print(f"JAX output_projection shape: {jax_out.shape}")
    print(f"JAX output_projection[:5, 0]: {jax_out[:5, 0]}")

    # PyTorch should transpose it
    torch_out = torch_state_dict["output_projection.weight"].numpy()
    print(f"PyTorch output_projection shape: {torch_out.shape}")
    print(f"PyTorch output_projection[0, :5]: {torch_out[0, :5]}")

    # Compare (need to transpose JAX to match PyTorch)
    jax_out_t = jax_out.T
    out_diff = np.abs(jax_out_t - torch_out)
    print(f"Max diff: {np.max(out_diff):.6e}")
    print(f"Mean diff: {np.mean(out_diff):.6e}")
else:
    print("JAX uses tied weights (no separate output_projection)")
    print("PyTorch should also use tied weights")
    print(
        f"PyTorch output_projection.weight is embedding.weight: {torch_state_dict['output_projection.weight'].data_ptr() == torch_state_dict['embedding.weight'].data_ptr()}"
    )

print("\n=== Checking First Block Weights ===")
# Check first SeqCond block
jax_block0 = jax_params["seqcond_block_0"]["SeqCondAttention_0"]
torch_block0_prefix = "blocks.0.attn."

print(f"JAX in_proj shape: {np.array(jax_block0['in_proj']['kernel']).shape}")
torch_in_proj = torch_state_dict[torch_block0_prefix + "in_proj.weight"].numpy()
print(f"PyTorch in_proj shape: {torch_in_proj.shape}")

jax_in_proj = np.array(jax_block0["in_proj"]["kernel"])
jax_in_proj_t = jax_in_proj.T

in_proj_diff = np.abs(jax_in_proj_t - torch_in_proj)
print(f"in_proj max diff: {np.max(in_proj_diff):.6e}")
print(f"in_proj mean diff: {np.mean(in_proj_diff):.6e}")

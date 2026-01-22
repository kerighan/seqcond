"""
Test to verify dtype handling in JAX -> NumPy -> PyTorch conversion.
"""

import jax.numpy as jnp
import numpy as np
import torch
import pickle

# Load checkpoint to check actual dtypes
CHECKPOINT_PATH = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step50000.pkl"

with open(CHECKPOINT_PATH, "rb") as f:
    data = pickle.load(f)

flax_params = data["params"]

# Check a few parameter dtypes
print("=== JAX Parameter Dtypes ===")
print(f"token_embedding: {flax_params['token_embedding']['embedding'].dtype}")
print(
    f"seqcond_block_0.in_proj: {flax_params['seqcond_block_0']['SeqCondAttention_0']['in_proj']['kernel'].dtype}"
)
print(
    f"seqcond_block_0.conv: {flax_params['seqcond_block_0']['SeqCondAttention_0']['conv']['kernel'].dtype}"
)

# Test conversion
print("\n=== Conversion Test ===")
test_param_bf16 = flax_params["token_embedding"]["embedding"]
print(f"Original JAX dtype: {test_param_bf16.dtype}")

# Convert to numpy
np_param = np.array(test_param_bf16)
print(f"After np.array(): {np_param.dtype}")

# Convert to torch
torch_param = torch.from_numpy(np_param)
print(f"After torch.from_numpy(): {torch_param.dtype}")

# What if we explicitly convert?
print("\n=== Explicit Conversion ===")
# JAX bfloat16 -> NumPy float32 -> PyTorch bfloat16
np_param_f32 = np.array(test_param_bf16, dtype=np.float32)
print(f"np.array(..., dtype=np.float32): {np_param_f32.dtype}")

torch_param_bf16 = torch.from_numpy(np_param_f32).to(torch.bfloat16)
print(f"torch.from_numpy().to(torch.bfloat16): {torch_param_bf16.dtype}")

# Check if values are preserved
print("\n=== Value Preservation ===")
print(f"Original JAX value [0,0]: {test_param_bf16[0, 0]}")
print(f"NumPy value [0,0]: {np_param[0, 0]}")
print(f"Torch value [0,0]: {torch_param[0, 0]}")
print(f"Torch bf16 value [0,0]: {torch_param_bf16[0, 0]}")

# Check difference
diff = abs(float(test_param_bf16[0, 0]) - float(torch_param_bf16[0, 0]))
print(f"\nDifference: {diff:.10e}")

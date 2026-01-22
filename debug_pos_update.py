"""
Debug script to check if pos is being updated correctly in PyTorch.
"""

import torch
from seqcond.torch.generator import TorchGenerator

checkpoint_path = "checkpoints/seqcond_torch_50k.pt"
gen = TorchGenerator(checkpoint_path)

prompt = "The quick"
tokens = gen.tokenizer([prompt])[0]

print(f"Prompt: '{prompt}'")
print(f"Tokens: {tokens}")

# Initialize states
states = gen.model.init_state(batch_size=1, device="cuda")

# Check initial pos values
print("\nInitial pos values:")
for i, s in enumerate(states):
    if isinstance(s, tuple):
        if len(s) == 5:  # SeqCond
            print(f"  Block {i} (SeqCond): pos = {s[3].item()}")
        else:  # Transformer
            print(f"  Block {i} (Transformer): pos = {s[2].item()}")

# Process first token
print("\n=== Processing first token ===")
token_tensor = torch.tensor([[tokens[0]]], device="cuda")
logits, states = gen.model.step(token_tensor, states)

print("After first step, pos values:")
for i, s in enumerate(states):
    if isinstance(s, tuple):
        if len(s) == 5:  # SeqCond
            print(f"  Block {i} (SeqCond): pos = {s[3].item()}")
        else:  # Transformer
            print(f"  Block {i} (Transformer): pos = {s[2].item()}")

# Process second token
print("\n=== Processing second token ===")
token_tensor = torch.tensor([[tokens[1]]], device="cuda")
logits, states = gen.model.step(token_tensor, states)

print("After second step, pos values:")
for i, s in enumerate(states):
    if isinstance(s, tuple):
        if len(s) == 5:  # SeqCond
            print(f"  Block {i} (SeqCond): pos = {s[3].item()}")
        else:  # Transformer
            print(f"  Block {i} (Transformer): pos = {s[2].item()}")

# Process third token
print("\n=== Processing third token ===")
token_tensor = torch.tensor(
    [[tokens[1]]], device="cuda"
)  # Same token to see if output changes
logits, states = gen.model.step(token_tensor, states)

print("After third step, pos values:")
for i, s in enumerate(states):
    if isinstance(s, tuple):
        if len(s) == 5:  # SeqCond
            print(f"  Block {i} (SeqCond): pos = {s[3].item()}")
        else:  # Transformer
            print(f"  Block {i} (Transformer): pos = {s[2].item()}")

print(f"\nLogits for same token at different positions:")
print(f"  First 5 logits: {logits[0, :5]}")

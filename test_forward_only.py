"""
Test PyTorch generation using forward() only (no step).
"""

import torch
from seqcond.torch.generator import TorchGenerator

checkpoint_path = "checkpoints/seqcond_torch_60k.pt"
gen = TorchGenerator(checkpoint_path)

prompt = "NAD+ is"
tokens = gen.tokenizer([prompt])[0]

print(f"Prompt: '{prompt}'")
print(f"Tokens: {tokens}")

# Use forward for entire sequence
input_ids = torch.tensor([tokens], dtype=torch.long).to("cuda")

with torch.no_grad():
    logits, _ = gen.model(input_ids)  # Forward pass returns (logits, states)

# Get prediction for last token
last_logits = logits[0, -1, :]
predicted_token = torch.argmax(last_logits).item()

print(f"\nPredicted next token: {predicted_token}")
print(f"Token text: '{gen.tokenizer.decode([predicted_token])}'")
print(f"Top 5 logits: {last_logits.topk(5).values.cpu().numpy()}")
print(f"Top 5 tokens: {last_logits.topk(5).indices.cpu().numpy()}")

"""
Test PyTorch generation using only step() method, bypassing prefill.
"""

import torch
from seqcond.torch.generator import TorchGenerator

checkpoint_path = "checkpoints/seqcond_torch_50k.pt"
gen = TorchGenerator(checkpoint_path)

prompt = "The quick brown fox"
tokens = gen.tokenizer([prompt])[0]

print(f"Prompt: '{prompt}'")
print(f"Tokens: {tokens}")

# Initialize states
states = gen.model.init_state(batch_size=1, device="cuda")

# Process prompt tokens one by one using step()
print("\nProcessing prompt with step()...")
for i, token_id in enumerate(tokens):
    token_tensor = torch.tensor([[token_id]], device="cuda")
    logits, states = gen.model.step(token_tensor, states)
    print(f"  Token {i}: {token_id}, logits shape: {logits.shape}")

# Generate new tokens
print("\nGenerating new tokens...")
generated = []
for step in range(50):
    # Sample next token (greedy for now)
    next_token = torch.argmax(logits[0]).item()
    generated.append(next_token)

    if step < 5:
        print(f"  Step {step}: token={next_token}, logits[:5]={logits[0, :5]}")

    # Step with new token
    token_tensor = torch.tensor([[next_token]], device="cuda")
    logits, states = gen.model.step(token_tensor, states)

# Decode
full_tokens = tokens + generated
text = gen.tokenizer.decode(full_tokens)

print(f"\nGenerated text:\n{text}")

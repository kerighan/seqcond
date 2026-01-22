"""
Test PyTorch generation without prefill - process prompt with step() only.
"""

import torch
from seqcond.torch.generator import TorchGenerator

checkpoint_path = "checkpoints/seqcond_torch_60k.pt"
gen = TorchGenerator(checkpoint_path)

prompt = "The quick brown fox"
tokens = gen.tokenizer([prompt])[0]

print(f"Prompt: '{prompt}'")
print(f"Tokens: {tokens}")

# Initialize states
states = gen.model.init_state(batch_size=1, device="cuda")

# Process ALL tokens (prompt + generation) with step() only
print("\nProcessing prompt with step()...")
current_pos = 0
for i, token_id in enumerate(tokens):
    token_tensor = torch.tensor([[token_id]], device="cuda")
    pos_tensor = torch.tensor([current_pos], device="cuda")
    logits, states = gen.model.step(token_tensor, states, pos=pos_tensor)
    current_pos += 1
    if i < 3:
        top_token = torch.argmax(logits[0]).item()
        print(
            f"  Token {i} (pos={current_pos-1}): input={token_id}, predicted={top_token}"
        )

# Generate new tokens
print("\nGenerating new tokens...")
generated = []
for step in range(50):
    # Sample next token with temperature
    logits_scaled = logits[0] / 0.8
    probs = torch.softmax(logits_scaled, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    generated.append(next_token)

    # Step with new token
    token_tensor = torch.tensor([[next_token]], device="cuda")
    pos_tensor = torch.tensor([current_pos], device="cuda")
    logits, states = gen.model.step(token_tensor, states, pos=pos_tensor)
    current_pos += 1

# Decode
full_tokens = tokens + generated
text = gen.tokenizer.decode(full_tokens)

print(f"\nGenerated text:\n{text}")

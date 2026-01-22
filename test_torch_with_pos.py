"""
Test PyTorch generation by passing pos explicitly to step().
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

# Process prompt tokens one by one using step() with explicit pos
print("\nProcessing prompt with step() and explicit pos...")
current_pos = 0
for i, token_id in enumerate(tokens):
    token_tensor = torch.tensor([[token_id]], device="cuda")
    pos_tensor = torch.tensor([current_pos], device="cuda")
    logits, states = gen.model.step(token_tensor, states, pos=pos_tensor)
    print(f"  Token {i} (pos={current_pos}): {token_id}, logits shape: {logits.shape}")
    current_pos += 1

# Generate new tokens
print("\nGenerating new tokens with explicit pos...")
generated = []
for step in range(50):
    # Sample next token (greedy)
    next_token = torch.argmax(logits[0]).item()
    generated.append(next_token)

    if step < 5:
        print(
            f"  Step {step} (pos={current_pos}): token={next_token}, logits[:5]={logits[0, :5]}"
        )

    # Step with new token and explicit pos
    token_tensor = torch.tensor([[next_token]], device="cuda")
    pos_tensor = torch.tensor([current_pos], device="cuda")
    logits, states = gen.model.step(token_tensor, states, pos=pos_tensor)
    current_pos += 1

# Decode
full_tokens = tokens + generated
text = gen.tokenizer.decode(full_tokens)

print(f"\nGenerated text:\n{text}")

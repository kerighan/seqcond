"""Test torch2 model loading and generation."""

import torch
from seqcond.torch2.model import SeqCondModel
from seqcond.dataset import Tokenizer

# Load checkpoint
checkpoint_path = "checkpoints/seqcond_torch2_60k.pt"
print(f"Loading {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location="cuda")

# Create model
print("Creating model...")
model = SeqCondModel(**checkpoint["config"]).to("cuda").eval()

# Load weights
print("Loading weights...")
missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
print(f"Missing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")
if missing:
    print(f"First 5 missing: {missing[:5]}")
if unexpected:
    print(f"First 5 unexpected: {unexpected[:5]}")

# Test generation
tokenizer = Tokenizer()
prompt = "NAD+ is"
tokens = tokenizer([prompt])[0]
print(f"\nPrompt: '{prompt}'")
print(f"Tokens: {tokens}")

# Initialize state
print("\nInitializing state...")
states = model.init_state(batch_size=1, device="cuda")

# Generate with step()
print("Generating...")
generated = []
for i, token_id in enumerate(tokens):
    token_tensor = torch.tensor([[token_id]], device="cuda")
    with torch.no_grad():
        logits, states = model.step(token_tensor, states)
    if i == len(tokens) - 1:
        next_token = torch.argmax(logits[0]).item()
        print(
            f"After prompt, predicted: {next_token} ('{tokenizer.decode([next_token])}')"
        )

# Generate more tokens
for step in range(50):
    token_tensor = torch.tensor([[next_token]], device="cuda")
    with torch.no_grad():
        logits, states = model.step(token_tensor, states)
    next_token = torch.argmax(logits[0]).item()
    generated.append(next_token)

full_text = prompt + tokenizer.decode(generated)
print(f"\nGenerated:\n{full_text}")

"""
Test HuggingFace model generation.
"""

from transformers import AutoModelForCausalLM
import torch
from seqcond.dataset import Tokenizer

model_path = "hf_checkpoints/seqcond-60k"

print(f"Loading HF model from {model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float32,
).to("cuda")

tokenizer = Tokenizer()

print("Model loaded successfully!")

# Test generation
prompt = "NAD+ is"
print(f"\nPrompt: '{prompt}'")

# Tokenize
tokens = tokenizer([prompt])[0]
input_ids = torch.tensor([tokens], dtype=torch.long).to("cuda")

print(f"Input tokens: {tokens}")
print("Generating...")

with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=False,  # Greedy
        pad_token_id=0,
    )

generated_tokens = outputs[0].cpu().tolist()
generated_text = tokenizer.decode(generated_tokens)
print(f"\nGenerated text:\n{generated_text}")

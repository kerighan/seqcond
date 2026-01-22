"""Simple debug script to compare JAX and PyTorch outputs.

This script loads both models and compares their outputs directly.
"""

import numpy as np
import jax
import torch
from seqcond.jax.generator import Generator as JAXGenerator
from seqcond.torch.generator import TorchGenerator
from seqcond.dataset import Tokenizer

CHECKPOINT_JAX = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step50000.pkl"
CHECKPOINT_TORCH = "checkpoints/seqcond_torch_50k.pt"
PROMPT = "NAD+ is"
MAX_TOKENS = 5

def cosine_similarity(a, b):
    """Compute cosine similarity between two arrays."""
    a = a.flatten()
    b = b.flatten()
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if (norm_a > 0 and norm_b > 0) else 0

def compare_outputs():
    """Compare outputs directly."""
    print("Loading models...")
    
    # Load JAX model
    jax_gen = JAXGenerator(CHECKPOINT_JAX)
    
    # Load PyTorch model
    torch_gen = TorchGenerator(CHECKPOINT_TORCH)
    
    # Tokenize prompt
    tokenizer = Tokenizer()
    prompt_tokens = tokenizer([PROMPT])[0]
    print(f"Prompt: {PROMPT}")
    print(f"Prompt tokens: {prompt_tokens}")
    
    # Generate with JAX
    print(f"\nGenerating with JAX...")
    jax_text = jax_gen.generate(
        prompt=PROMPT,
        max_new_tokens=MAX_TOKENS,
        temperature=0.0,
        verbose=True
    )
    
    # Generate with PyTorch
    print(f"\nGenerating with PyTorch...")
    torch_text = torch_gen.generate(
        prompt=PROMPT,
        max_new_tokens=MAX_TOKENS,
        temperature=0.0,
        verbose=True,
        use_cuda_graph=False
    )
    
    print(f"\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"JAX: {jax_text}")
    print(f"Torch: {torch_text}")
    print(f"Match: {jax_text == torch_text}")
    
    # Tokenize both outputs
    jax_tokens = tokenizer([jax_text])[0]
    torch_tokens = tokenizer([torch_text])[0]
    
    print(f"\nJAX tokens: {jax_tokens}")
    print(f"Torch tokens: {torch_tokens}")
    
    # Find first difference
    min_len = min(len(jax_tokens), len(torch_tokens))
    for i in range(min_len):
        if jax_tokens[i] != torch_tokens[i]:
            print(f"\nFirst difference at position {i}:")
            print(f"  JAX: {jax_tokens[i]} ({tokenizer.decode([jax_tokens[i]])})")
            print(f"  Torch: {torch_tokens[i]} ({tokenizer.decode([torch_tokens[i]])})")
            break

if __name__ == "__main__":
    try:
        compare_outputs()
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
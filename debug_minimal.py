"""Minimal debug script to compare JAX and PyTorch.

This script uses a very short prompt and minimal generation to avoid memory issues.
"""

import numpy as np
import jax
import torch
from seqcond.jax.generator import Generator as JAXGenerator
from seqcond.torch.generator import TorchGenerator
from seqcond.dataset import Tokenizer

CHECKPOINT_JAX = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step50000.pkl"
CHECKPOINT_TORCH = "checkpoints/seqcond_torch_50k.pt"
PROMPT = "A"
MAX_STEPS = 3

def compare_minimal():
    """Compare with minimal settings."""
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
    try:
        jax_text = jax_gen.generate(
            prompt=PROMPT,
            max_new_tokens=MAX_STEPS,
            temperature=0.0,
            verbose=True
        )
        print(f"JAX result: {jax_text}")
    except Exception as e:
        print(f"JAX failed: {e}")
        jax_text = None
    
    # Generate with PyTorch
    print(f"\nGenerating with PyTorch...")
    try:
        torch_text = torch_gen.generate(
            prompt=PROMPT,
            max_new_tokens=MAX_STEPS,
            temperature=0.0,
            verbose=True,
            use_cuda_graph=False
        )
        print(f"Torch result: {torch_text}")
    except Exception as e:
        print(f"Torch failed: {e}")
        torch_text = None
    
    # Compare
    if jax_text and torch_text:
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
        compare_minimal()
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
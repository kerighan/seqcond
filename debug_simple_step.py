"""Simple step-by-step debug script.

This script uses the existing generator methods to compare outputs.
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
MAX_STEPS = 10

def cosine_similarity(a, b):
    """Compute cosine similarity between two arrays."""
    a = a.flatten()
    b = b.flatten()
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if (norm_a > 0 and norm_b > 0) else 0

def compare_step_by_step():
    """Compare step by step using generator methods."""
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
    print(f"Prompt token strings: {[tokenizer.decode([t]) for t in prompt_tokens]}")
    
    # Generate step by step with JAX
    print("\n" + "="*80)
    print("JAX GENERATION")
    print("="*80)
    
    jax_text = jax_gen.generate(
        prompt=PROMPT,
        max_new_tokens=MAX_STEPS,
        temperature=0.0,
        verbose=True
    )
    
    # Generate step by step with PyTorch
    print("\n" + "="*80)
    print("PYTORCH GENERATION")
    print("="*80)
    
    torch_text = torch_gen.generate(
        prompt=PROMPT,
        max_new_tokens=MAX_STEPS,
        temperature=0.0,
        verbose=True,
        use_cuda_graph=False
    )
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    jax_tokens = tokenizer([jax_text])[0]
    torch_tokens = tokenizer([torch_text])[0]
    
    print(f"JAX: {jax_text}")
    print(f"Torch: {torch_text}")
    print(f"Match: {jax_text == torch_text}")
    
    print(f"\nJAX tokens: {jax_tokens}")
    print(f"Torch tokens: {torch_tokens}")
    
    # Find first difference
    min_len = min(len(jax_tokens), len(torch_tokens))
    for i in range(min_len):
        if jax_tokens[i] != torch_tokens[i]:
            print(f"\nFirst difference at position {i}:")
            print(f"  JAX: {jax_tokens[i]} ({tokenizer.decode([jax_tokens[i]])})")
            print(f"  Torch: {torch_tokens[i]} ({tokenizer.decode([torch_tokens[i]])})")
            
            # Show context around the difference
            start = max(0, i - 3)
            end = min(min_len, i + 4)
            print(f"\nContext around difference:")
            for j in range(start, end):
                jax_char = tokenizer.decode([jax_tokens[j]])
                torch_char = tokenizer.decode([torch_tokens[j]])
                match = "✓" if jax_tokens[j] == torch_tokens[j] else "✗"
                print(f"  Pos {j}: JAX={jax_tokens[j]:5d} ({jax_char:10s}) | Torch={torch_tokens[j]:5d} ({torch_char:10s}) {match}")
            break
    
    # Detailed analysis of the divergence point
    if jax_tokens != torch_tokens:
        print(f"\n" + "="*80)
        print("DETAILED ANALYSIS AT DIVERGENCE POINT")
        print("="*80)
        
        # Find the exact step where they diverge
        for i in range(min_len):
            if jax_tokens[i] != torch_tokens[i]:
                print(f"Divergence occurs at token position {i}")
                print(f"This is the {i - len(prompt_tokens) + 1}-th generated token")
                
                # Show what each model predicted
                if i > 0:
                    prev_token = jax_tokens[i-1]
                    print(f"Previous token: {prev_token} ('{tokenizer.decode([prev_token])}')")
                
                print(f"JAX chose: {jax_tokens[i]} ('{tokenizer.decode([jax_tokens[i]])}')")
                print(f"Torch chose: {torch_tokens[i]} ('{tokenizer.decode([torch_tokens[i]])}')")
                break

if __name__ == "__main__":
    try:
        compare_step_by_step()
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
"""Debug script to compare JAX and PyTorch step by step.

This script compares the internal states, logits, and token choices at each step.
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
    """Compare step by step."""
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
    
    # Initialize states
    print("\nInitializing states...")
    jax_states = jax_gen.model.init_state(1)
    torch_states = torch_gen.model.init_state(1, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"JAX states: {len(jax_states)} blocks")
    print(f"Torch states: {len(torch_states)} blocks")
    
    # Prefill phase - process prompt tokens
    print("\n" + "="*80)
    print("PREFILL PHASE")
    print("="*80)
    
    jax_embeddings = None
    torch_embeddings = None
    
    for step, token_id in enumerate(prompt_tokens):
        print(f"\n--- Prefill Step {step}: Token {token_id} ('{tokenizer.decode([token_id])}') ---")
        
        # JAX step
        token_array = jnp.array([[token_id]], dtype=jnp.int32)
        pos_array = jnp.array(step, dtype=jnp.int32)
        
        if jax_embeddings is None:
            # First token - get embeddings
            jax_embeddings = jax_gen.model.embed(token_array)
        
        jax_logits, jax_states = jax_gen.model.step(token_array, jax_embeddings, jax_states, pos_array)
        
        # PyTorch step
        torch_token = torch.tensor([[token_id]], device='cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch_embeddings is None:
            # First token - get embeddings
            torch_embeddings = torch_gen.model.embed(torch_token)
        
        with torch.no_grad():
            torch_logits, torch_states = torch_gen.model.step(torch_token, torch_embeddings, torch_states)
        
        # Convert to numpy for comparison
        jax_logits_np = np.array(jax_logits[0])
        torch_logits_np = torch_logits[0].cpu().numpy()
        
        # Compare logits
        logits_sim = cosine_similarity(jax_logits_np, torch_logits_np)
        print(f"Logits similarity: {logits_sim:.6f}")
        
        # Get top 5 predictions
        jax_top5 = np.argsort(jax_logits_np)[-5:][::-1]
        torch_top5 = np.argsort(torch_logits_np)[-5:][::-1]
        
        print(f"JAX top 5: {jax_top5} -> {[tokenizer.decode([t]) for t in jax_top5]}")
        print(f"Torch top 5: {torch_top5} -> {[tokenizer.decode([t]) for t in torch_top5]}")
        
        # Update embeddings for next step
        jax_embeddings = jax_logits  # Use logits as next input (this might not be correct)
        torch_embeddings = torch_logits
        
        # Early exit
        if step >= len(prompt_tokens) - 1:
            break
    
    print("\n" + "="*80)
    print("GENERATION PHASE")
    print("="*80)
    
    # Generation phase
    jax_tokens = prompt_tokens.copy()
    torch_tokens = prompt_tokens.copy()
    
    for step in range(MAX_STEPS):
        print(f"\n--- Generation Step {step} ---")
        
        # JAX generation
        jax_next_token = int(jnp.argmax(jax_logits[0]))
        jax_token_array = jnp.array([[jax_next_token]], dtype=jnp.int32)
        jax_pos_array = jnp.array(len(prompt_tokens) + step, dtype=jnp.int32)
        
        jax_logits, jax_states = jax_gen.model.step(jax_token_array, jax_logits, jax_states, jax_pos_array)
        
        # PyTorch generation
        torch_next_token = int(torch.argmax(torch_logits[0]))
        torch_token = torch.tensor([[torch_next_token]], device='cuda' if torch.cuda.is_available() else 'cpu')
        
        with torch.no_grad():
            torch_logits, torch_states = torch_gen.model.step(torch_token, torch_logits, torch_states)
        
        jax_tokens.append(jax_next_token)
        torch_tokens.append(torch_next_token)
        
        # Convert to numpy for comparison
        jax_logits_np = np.array(jax_logits[0])
        torch_logits_np = torch_logits[0].cpu().numpy()
        
        # Compare
        logits_sim = cosine_similarity(jax_logits_np, torch_logits_np)
        print(f"Token choice - JAX: {jax_next_token} ('{tokenizer.decode([jax_next_token])}'), Torch: {torch_next_token} ('{tokenizer.decode([torch_next_token])}')")
        print(f"Logits similarity: {logits_sim:.6f}")
        
        # Check if they diverged
        if jax_next_token != torch_next_token:
            print(f"⚠️  DIVERGENCE at step {step}!")
            
            # Show top predictions
            jax_top5 = np.argsort(jax_logits_np)[-5:][::-1]
            torch_top5 = np.argsort(torch_logits_np)[-5:][::-1]
            
            print(f"JAX top 5: {jax_top5} -> {[tokenizer.decode([t]) for t in jax_top5]}")
            print(f"Torch top 5: {torch_top5} -> {[tokenizer.decode([t]) for t in torch_top5]}")
            
            # Show actual logit values for the chosen tokens
            print(f"JAX logit for chosen token {jax_next_token}: {jax_logits_np[jax_next_token]:.6f}")
            print(f"Torch logit for chosen token {torch_next_token}: {torch_logits_np[torch_next_token]:.6f}")
            print(f"JAX logit for torch's choice {torch_next_token}: {jax_logits_np[torch_next_token]:.6f}")
            print(f"Torch logit for jax's choice {jax_next_token}: {torch_logits_np[jax_next_token]:.6f}")
            
            break
    
    print(f"\nFinal sequences:")
    print(f"JAX: {tokenizer.decode(jax_tokens)}")
    print(f"Torch: {tokenizer.decode(torch_tokens)}")

if __name__ == "__main__":
    try:
        compare_step_by_step()
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
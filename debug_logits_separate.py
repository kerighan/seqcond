"""Debug script to compare logits step by step (separate processes).

This script runs JAX and PyTorch in separate processes to avoid GPU memory issues.
"""

import numpy as np
import torch
import jax
import jax.numpy as jnp
import subprocess
import sys
from seqcond.dataset import Tokenizer

CHECKPOINT_JAX = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step50000.pkl"
CHECKPOINT_TORCH = "checkpoints/seqcond_torch_50k.pt"
PROMPT = "The quick brown fox"

def run_jax_steps():
    """Run JAX model and save logits at each step."""
    from seqcond.jax.generator import Generator as JAXGenerator, _make_step_fn
    
    print("[JAX] Loading model...")
    jax_gen = JAXGenerator(CHECKPOINT_JAX)
    tokenizer = Tokenizer()
    
    # Tokenize prompt
    tokens = tokenizer([PROMPT])[0]
    print(f"[JAX] Prompt tokens: {tokens}")
    
    # Initialize state
    jax_states = jax_gen.model.apply(
        {"params": jax_gen.params},
        batch_size=1,
        method=jax_gen.model.init_state,
    )
    
    # Create step function
    jax_step_fn = _make_step_fn(jax_gen.model, jax_gen.params)
    
    # Process each token and save logits
    jax_logits_list = []
    
    for step, token_id in enumerate(tokens):
        print(f"[JAX] Processing step {step}: token {token_id}")
        
        token_array = jnp.array([[token_id]], dtype=jnp.int32)
        pos_array = jnp.array(step, dtype=jnp.int32)
        jax_logits, jax_states = jax_step_fn(token_array, jax_states, pos_array)
        
        # Save logits
        jax_logits_np = np.array(jax_logits[0])
        jax_logits_list.append(jax_logits_np)
        
        # Save argmax
        argmax_token = int(np.argmax(jax_logits_np))
        print(f"[JAX] Step {step} argmax: {argmax_token} ({tokenizer.decode([argmax_token])!r})")
    
    # Save all logits
    np.save("/tmp/jax_step_logits.npy", np.array(jax_logits_list))
    print(f"[JAX] Saved logits for {len(jax_logits_list)} steps")

def run_torch_steps():
    """Run PyTorch model and save logits at each step."""
    from seqcond.torch.generator import TorchGenerator
    
    print("[PyTorch] Loading model...")
    torch_gen = TorchGenerator(CHECKPOINT_TORCH)
    tokenizer = Tokenizer()
    
    # Tokenize prompt
    tokens = tokenizer([PROMPT])[0]
    print(f"[PyTorch] Prompt tokens: {tokens}")
    
    # Initialize state
    torch_states = torch_gen.model.init_state(batch_size=1, device="cuda")
    
    # Process each token and save logits
    torch_logits_list = []
    
    for step, token_id in enumerate(tokens):
        print(f"[PyTorch] Processing step {step}: token {token_id}")
        
        torch_token = torch.tensor([[token_id]], device="cuda")
        with torch.no_grad():
            torch_logits, torch_states = torch_gen.model.step(torch_token, torch_states)
        
        # Save logits
        torch_logits_np = torch_logits[0].cpu().numpy()
        torch_logits_list.append(torch_logits_np)
        
        # Save argmax
        argmax_token = int(np.argmax(torch_logits_np))
        print(f"[PyTorch] Step {step} argmax: {argmax_token} ({tokenizer.decode([argmax_token])!r})")
    
    # Save all logits
    np.save("/tmp/torch_step_logits.npy", np.array(torch_logits_list))
    print(f"[PyTorch] Saved logits for {len(torch_logits_list)} steps")

def compare_step_logits():
    """Compare saved step logits."""
    jax_logits_all = np.load("/tmp/jax_step_logits.npy")
    torch_logits_all = np.load("/tmp/torch_step_logits.npy")
    
    tokenizer = Tokenizer()
    tokens = tokenizer([PROMPT])[0]
    
    print("\n" + "="*80)
    print("STEP-BY-STEP LOGITS COMPARISON")
    print("="*80)
    
    all_correlations = []
    all_max_diffs = []
    
    for step in range(len(tokens)):
        print(f"\n--- Step {step}: Token {tokens[step]} ('{tokenizer.decode([tokens[step]])}') ---")
        
        jax_logits = jax_logits_all[step]
        torch_logits = torch_logits_all[step]
        
        # Compute statistics
        diff = jax_logits - torch_logits
        correlation = np.corrcoef(jax_logits, torch_logits)[0,1]
        max_diff = np.abs(diff).max()
        mean_diff = np.abs(diff).mean()
        
        all_correlations.append(correlation)
        all_max_diffs.append(max_diff)
        
        print(f"Correlation: {correlation:.6f}")
        print(f"Max absolute diff: {max_diff:.6f}")
        print(f"Mean absolute diff: {mean_diff:.6f}")
        
        # Get top predictions
        jax_top3 = np.argsort(jax_logits)[-3:][::-1]
        torch_top3 = np.argsort(torch_logits)[-3:][::-1]
        
        print(f"JAX top 3: {[tokenizer.decode([t]) for t in jax_top3]}")
        print(f"Torch top 3: {[tokenizer.decode([t]) for t in torch_top3]}")
        
        # Check if argmax differs
        jax_argmax = np.argmax(jax_logits)
        torch_argmax = np.argmax(torch_logits)
        
        if jax_argmax != torch_argmax:
            print(f"⚠️  Argmax divergence at step {step}!")
            print(f"  JAX: {jax_argmax} ({tokenizer.decode([jax_argmax])!r})")
            print(f"  Torch: {torch_argmax} ({tokenizer.decode([torch_argmax])!r})")
        else:
            print(f"✓ Argmax matches: {jax_argmax} ({tokenizer.decode([jax_argmax])!r})")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"Overall correlation range: {min(all_correlations):.6f} to {max(all_correlations):.6f}")
    print(f"Overall max diff range: {min(all_max_diffs):.6f} to {max(all_max_diffs):.6f}")
    
    # Find step with worst correlation
    worst_corr_step = np.argmin(all_correlations)
    print(f"Worst correlation at step {worst_corr_step}: {all_correlations[worst_corr_step]:.6f}")
    
    # Find step with largest max diff
    worst_diff_step = np.argmax(all_max_diffs)
    print(f"Largest max diff at step {worst_diff_step}: {all_max_diffs[worst_diff_step]:.6f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "jax":
            run_jax_steps()
        elif sys.argv[1] == "torch":
            run_torch_steps()
        elif sys.argv[1] == "compare":
            compare_step_logits()
    else:
        print("="*80)
        print("Comparing Step-by-Step Logits: JAX vs PyTorch")
        print("="*80)
        
        # Run JAX in subprocess
        print("\n[1/2] Running JAX...")
        subprocess.run([sys.executable, __file__, "jax"], check=True)
        
        # Run PyTorch in subprocess
        print("\n[2/2] Running PyTorch...")
        subprocess.run([sys.executable, __file__, "torch"], check=True)
        
        # Compare
        compare_step_logits()
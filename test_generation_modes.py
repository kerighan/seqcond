"""
Compare text generation using step() vs __call__() with argmax sampling.

This tests if the divergence between __call__ and step affects actual token generation.
"""

import pickle
import jax
import jax.numpy as jnp
import numpy as np
from seqcond.jax.model import SeqCondModel
from seqcond.dataset import Tokenizer


def load_model():
    checkpoint_path = "checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step50000.pkl"
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    
    params = data["params"]
    config_dict = data["config"]["model"]
    
    model = SeqCondModel(
        vocab_size=config_dict["vocab_size"],
        d_model=config_dict["d_model"],
        d_ff=config_dict["d_ff"],
        num_layers=config_dict["num_layers"],
        num_heads=config_dict["num_heads"],
        num_kv_heads=config_dict.get("num_kv_heads"),
        maxlen=config_dict["maxlen"],
        dropout=0.0,
        tie_weights=config_dict.get("tie_weights", True),
        qk_norm=config_dict.get("qk_norm", False),
        qk_norm_eps=config_dict.get("qk_norm_eps", 1e-6),
        seqcond_heads=config_dict.get("seqcond_heads"),
        num_query_heads=config_dict.get("num_query_heads"),
        num_thetas=config_dict.get("num_thetas", 4),
        num_anchor_heads=config_dict.get("num_anchor_heads", 0),
        conv_kernel_size=config_dict.get("conv_kernel_size", 4),
        expand_factor=config_dict.get("expand_factor", 2),
        out_expand_factor=config_dict.get("out_expand_factor", 2),
        use_positional_embedding=config_dict.get("use_positional_embedding", False),
        seqcond_ratio=config_dict.get("seqcond_ratio", 3),
        use_square_matrix=config_dict.get("use_square_matrix", False),
        remat=False,
    )
    
    return model, params, config_dict


def generate_with_step(model, params, prompt_tokens, max_new_tokens=20):
    """Generate text using step() method (recurrent mode)."""
    B = 1
    tokens = list(prompt_tokens)
    
    # Initialize states
    states = model.apply(
        {"params": params},
        B,
        method=lambda m, batch_size: m.init_state(batch_size)
    )
    
    print(f"\n=== Generation with step() ===")
    print(f"Prompt: {tokens}")
    
    # Process prompt tokens one by one
    for i, token_id in enumerate(tokens):
        token_array = jnp.array([[token_id]], dtype=jnp.int32)
        logits, states = model.apply(
            {"params": params},
            token_array,
            states,
            i,
            deterministic=True,
            method=model.step,
        )
    
    # Generate new tokens
    for step in range(max_new_tokens):
        # Get next token (argmax)
        next_token = int(jnp.argmax(logits[0]))
        tokens.append(next_token)
        
        # Step with new token
        token_array = jnp.array([[next_token]], dtype=jnp.int32)
        logits, states = model.apply(
            {"params": params},
            token_array,
            states,
            len(tokens) - 1,
            deterministic=True,
            method=model.step,
        )
        
        if step < 5:  # Print first few logits for debugging
            print(f"  Step {step}: token={next_token}, logits[:5]={logits[0, :5]}")
    
    return tokens


def generate_with_call(model, params, prompt_tokens, max_new_tokens=20):
    """Generate text using __call__() method (parallel mode with growing sequence)."""
    tokens = list(prompt_tokens)
    
    print(f"\n=== Generation with __call__() ===")
    print(f"Prompt: {tokens}")
    
    for step in range(max_new_tokens):
        # Create input sequence (growing)
        inputs = jnp.array([tokens], dtype=jnp.int32)  # (1, L)
        
        # Run __call__ on entire sequence
        logits = model.apply(
            {"params": params},
            inputs,
            deterministic=True,
        )  # (1, L, vocab_size)
        
        # Get logits for last position
        last_logits = logits[0, -1, :]  # (vocab_size,)
        
        # Get next token (argmax)
        next_token = int(jnp.argmax(last_logits))
        tokens.append(next_token)
        
        if step < 5:  # Print first few logits for debugging
            print(f"  Step {step}: token={next_token}, logits[:5]={last_logits[:5]}")
    
    return tokens


def main():
    model, params, config = load_model()
    tokenizer = Tokenizer()
    
    # Test prompt
    prompt = "The quick brown"
    prompt_tokens = tokenizer.encode(prompt)
    
    print(f"Prompt text: '{prompt}'")
    print(f"Prompt tokens: {prompt_tokens}")
    
    # Generate with both methods
    max_new_tokens = 20
    
    tokens_step = generate_with_step(model, params, prompt_tokens, max_new_tokens)
    tokens_call = generate_with_call(model, params, prompt_tokens, max_new_tokens)
    
    # Decode and compare
    text_step = tokenizer.decode(tokens_step)
    text_call = tokenizer.decode(tokens_call)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nstep() generated tokens: {tokens_step}")
    print(f"step() generated text:\n{text_step}")
    
    print(f"\n__call__() generated tokens: {tokens_call}")
    print(f"__call__() generated text:\n{text_call}")
    
    # Compare token by token
    print("\n" + "="*80)
    print("TOKEN-BY-TOKEN COMPARISON")
    print("="*80)
    
    min_len = min(len(tokens_step), len(tokens_call))
    divergence_pos = None
    
    for i in range(min_len):
        match = "✓" if tokens_step[i] == tokens_call[i] else "✗"
        if tokens_step[i] != tokens_call[i] and divergence_pos is None:
            divergence_pos = i
        print(f"  Pos {i:2d}: step={tokens_step[i]:5d}, call={tokens_call[i]:5d} {match}")
    
    if divergence_pos is None:
        print("\n✅ SUCCESS: Both methods produced IDENTICAL token sequences!")
    else:
        print(f"\n❌ DIVERGENCE: First difference at position {divergence_pos}")
        print(f"   (position {divergence_pos - len(prompt_tokens)} in generated tokens)")


if __name__ == "__main__":
    main()

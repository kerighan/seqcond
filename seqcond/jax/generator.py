"""
Efficient text generation using step-by-step decoding.

This module provides a Generator class that uses O(1) step decoding for SeqCond blocks
and O(L) step decoding with KV cache for Transformer blocks, instead of recomputing
the full forward pass at each token.
"""

import pickle
from typing import Optional, Any, Tuple, List
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .model import SeqCondModel
from .rope import precompute_freqs, get_rope_embeddings
from ..dataset import Tokenizer
from ..config import ModelConfig


@dataclass
class GeneratorConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    seed: Optional[int] = None


@partial(jax.jit, static_argnums=(0,))
def _step_fn_jit(model, params, token_array, states, pos):
    """JIT-compiled step function with params as argument (not captured)."""
    return model.apply(
        {"params": params},
        token_array,
        states,
        pos,
        deterministic=True,
        method=model.step,
    )


def _make_step_fn(model, params):
    """Create a step function wrapper."""

    def step_fn(token_array, states, pos):
        return _step_fn_jit(model, params, token_array, states, pos)

    return step_fn


@partial(jax.jit, static_argnums=(0, 5))
def _generate_scan_jit(
    model, params, initial_token, initial_states, initial_pos, num_tokens
):
    """JIT-compiled generation loop using jax.lax.scan."""

    def scan_fn(carry, _):
        token, states, pos = carry
        token_array = token[None, :]  # (1, 1)
        logits, next_states = model.apply(
            {"params": params},
            token_array,
            states,
            pos,
            deterministic=True,
            method=model.step,
        )
        # Greedy sampling (argmax) inside the loop for maximum speed
        next_token = jnp.argmax(logits[0], axis=-1)
        return (next_token[None], next_states, pos + 1), next_token

    # Initial carry: (last_token, current_states, current_pos)
    init_carry = (initial_token, initial_states, initial_pos)
    _, generated_tokens = jax.lax.scan(scan_fn, init_carry, None, length=num_tokens)
    return generated_tokens


def generate_text_stepwise(
    model: SeqCondModel,
    params: dict,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    seed: Optional[int] = None,
    verbose: bool = True,
    use_scan: bool = False,  # New option
) -> str:
    """
    Generate text using efficient step-by-step decoding.

    Uses O(1) step for SeqCond blocks and O(L) KV cache for Transformer blocks.

    Args:
        model: SeqCondModel instance
        params: Model parameters
        tokenizer: Tokenizer instance
        prompt: Text prompt to continue
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty for repeated tokens
        seed: Random seed
        verbose: Print tokens as generated
        use_scan: Use jax.lax.scan for faster generation (Greedy only)

    Returns:
        Generated text including prompt
    """
    if seed is not None:
        np.random.seed(seed)

    # Tokenize prompt
    tokens = tokenizer([prompt])[0]

    # Initialize states
    batch_size = 1
    states = model.apply({"params": params}, batch_size, method=model.init_state)

    # Create JIT-compiled step function
    step_fn = _make_step_fn(model, params)

    if verbose:
        print(f"Compiling and prefilling {len(tokens)} tokens (first call is slow)...")

    # Prefill: run step for each prompt token to build up states
    import time

    t0 = time.time()
    last_logits = None
    for t, token_id in enumerate(tokens):
        token_array = jnp.array([[token_id]], dtype=jnp.int32)
        pos_array = jnp.array(t, dtype=jnp.int32)
        last_logits, states = step_fn(token_array, states, pos_array)

    # Block until prefill is done
    last_logits.block_until_ready()
    prefill_time = time.time() - t0

    if verbose:
        print(
            f"Prefill done in {prefill_time:.1f}s ({prefill_time/len(tokens)*1000:.1f}ms/token)"
        )
        print(prompt, end="", flush=True)

    pos = len(tokens)

    if use_scan:
        # Use fast lax.scan for generation (Argmax only for now)
        if verbose:
            print("\nGenerating with lax.scan (Greedy)...")

        # Sample first token from last prefill logits
        first_token_id = int(jnp.argmax(last_logits[0]))
        first_token = jnp.array([first_token_id], dtype=jnp.int32)

        t0_gen = time.time()
        generated_ids = _generate_scan_jit(
            model,
            params,
            first_token,
            states,
            jnp.array(pos, dtype=jnp.int32),
            max_new_tokens - 1,
        )
        generated_ids.block_until_ready()
        gen_time = time.time() - t0_gen

        # Prepend the first sampled token
        all_generated_ids = [first_token_id] + generated_ids.tolist()
        text = prompt + tokenizer.decode(all_generated_ids)

        if verbose:
            print(tokenizer.decode(all_generated_ids))
            print(
                f"Generation done in {gen_time:.1f}s ({gen_time/max_new_tokens*1000:.1f}ms/token)"
            )
        return text

    # Standard loop (supports temperature, top-k, etc.)
    generated_tokens = []
    text = prompt
    gen_start_time = time.time()
    logits = last_logits

    for i in range(max_new_tokens):
        # Sample next token from logits
        logits_np = np.array(logits[0], dtype=np.float64)

        # ... (sampling logic remains same)

        # Apply repetition penalty
        if repetition_penalty != 1.0 and generated_tokens:
            for token_id in set(generated_tokens):
                if logits_np[token_id] > 0:
                    logits_np[token_id] /= repetition_penalty
                else:
                    logits_np[token_id] *= repetition_penalty

        # Apply temperature
        logits_np = logits_np / temperature

        # Convert to probabilities
        probs = np.exp(logits_np - logits_np.max())
        probs = probs / probs.sum()

        # Top-k filtering
        if top_k > 0:
            top_k_idx = np.argpartition(probs, -top_k)[-top_k:]
            mask = np.zeros_like(probs)
            mask[top_k_idx] = 1
            probs = probs * mask
            probs = probs / probs.sum()

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff_idx = np.searchsorted(cumsum, top_p) + 1
            mask = np.zeros_like(probs)
            mask[sorted_idx[:cutoff_idx]] = 1
            probs = probs * mask
            probs = probs / probs.sum()

        # Sample
        token_id = int(np.random.choice(len(probs), p=probs))
        token = tokenizer.decode([token_id])

        tokens.append(token_id)
        generated_tokens.append(token_id)
        text += token

        if verbose:
            print(token, end="", flush=True)

        # Check max length
        if pos >= model.maxlen - 1:
            break

        # Step for next token (using JIT-compiled function)
        token_array = jnp.array([[token_id]], dtype=jnp.int32)
        pos_array = jnp.array(pos, dtype=jnp.int32)
        logits, states = step_fn(token_array, states, pos_array)
        pos += 1

    if verbose:
        gen_time = time.time() - gen_start_time
        num_gen = len(generated_tokens)
        if num_gen > 0:
            print(
                f"\nGeneration done in {gen_time:.1f}s ({gen_time/num_gen*1000:.1f}ms/token)"
            )
        else:
            print()

    return text


class Generator:
    """
    Efficient text generator using step-by-step decoding.

    Uses O(1) step decoding for SeqCond blocks and O(L) KV cache for Transformer blocks.
    """

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer: Optional[Tokenizer] = None,
    ):
        """
        Initialize generator from a checkpoint.

        Args:
            checkpoint_path: Path to the .pkl checkpoint file
            tokenizer: Optional tokenizer (uses default if not provided)
        """
        self.tokenizer = tokenizer or Tokenizer()

        # Load checkpoint
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)

        self.params = data["params"]
        self.config_dict = data["config"]
        self.checkpoint_step = data.get("step", 0)

        # Reconstruct model config
        model_config_dict = self.config_dict["model"]
        self.model_config = ModelConfig(**model_config_dict)

        # Create model
        self.model = SeqCondModel(
            vocab_size=self.model_config.vocab_size,
            d_model=self.model_config.d_model,
            num_layers=self.model_config.num_layers,
            num_heads=self.model_config.num_heads,
            num_kv_heads=self.model_config.num_kv_heads,
            d_ff=self.model_config.d_ff,
            maxlen=self.model_config.maxlen,
            dropout=0.0,  # No dropout during inference
            tie_weights=self.model_config.tie_weights,
            qk_norm=self.model_config.qk_norm,
            qk_norm_eps=self.model_config.qk_norm_eps,
            seqcond_heads=self.model_config.seqcond_heads,
            num_query_heads=self.model_config.num_query_heads,
            num_thetas=self.model_config.num_thetas,
            derivative_order=self.model_config.derivative_order,
            num_anchor_heads=self.model_config.num_anchor_heads,
            conv_kernel_size=self.model_config.conv_kernel_size,
            expand_factor=self.model_config.expand_factor,
            out_expand_factor=self.model_config.out_expand_factor,
            seqcond_ratio=self.model_config.seqcond_ratio,
            use_square_matrix=self.model_config.use_square_matrix,
            remat=False,  # No remat during inference
        )

        print(f"Generator initialized from checkpoint step {self.checkpoint_step}")
        print(
            f"Model: {self.model_config.num_layers} layers, d_model={self.model_config.d_model}"
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        seed: Optional[int] = None,
        verbose: bool = True,
        use_scan: bool = False,
    ) -> str:
        """
        Generate text from a prompt using efficient step-by-step decoding.

        Args:
            prompt: Text prompt to continue
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated tokens
            seed: Random seed
            verbose: Print tokens as generated
            use_scan: Use jax.lax.scan for faster generation (Greedy only)

        Returns:
            Generated text including the prompt
        """
        return generate_text_stepwise(
            model=self.model,
            params=self.params,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            verbose=verbose,
            use_scan=use_scan,
        )

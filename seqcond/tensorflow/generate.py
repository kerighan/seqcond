"""
Simple text generation utilities for TensorFlow models.
"""

import numpy as np
import tensorflow as tf


def generate_text(
    model,
    tokenizer,
    prompt: str = "<|im_start|>user\n",
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.5,
    maxlen: int = 768,
    verbose: bool = True,
) -> str:
    """
    Generate text from a language model.

    Args:
        model: Keras model that outputs logits
        tokenizer: Tokenizer with __call__ and decode methods
        prompt: Initial text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Number of top tokens to consider
        top_p: Cumulative probability threshold for nucleus sampling
        repetition_penalty: Penalty for repeating tokens
        maxlen: Maximum sequence length for the model
        verbose: Print tokens as they are generated

    Returns:
        Generated text string
    """
    tokens = tokenizer([prompt])[0]
    generated_tokens = []
    text = prompt

    for _ in range(max_new_tokens):
        # Pad to fixed length
        padded = tokens + [0] * (maxlen - len(tokens))
        padded = padded[:maxlen]
        input_array = np.array([padded], dtype=np.int32)

        # Get logits
        logits = model(input_array, training=False)
        logits_np = logits[0, len(tokens) - 1].numpy().astype(np.float64)

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

        if len(tokens) >= maxlen - 1:
            break

    if verbose:
        print()

    return text

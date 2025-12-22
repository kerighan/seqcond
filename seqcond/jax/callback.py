import os
from typing import Optional, Callable, List, Any

import jax
import jax.numpy as jnp
import numpy as np


class StepwiseGenerationCallback:
    """
    Callback that triggers autoregressive text generation every n steps.
    Uses the generate_text function internally.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        trigger_every_n_steps: int = 100,
        prompt: str = "<|im_start|>user\n",
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.5,
        maxlen: int = 768,
        seed: Optional[int] = None,
        model_name: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.trigger_every_n_steps = trigger_every_n_steps
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.maxlen = maxlen
        self.seed = seed
        self.step_counter = 0
        self.model_name = model_name
        self.log_file = None

    def on_train_begin(self):
        """Called at the beginning of training. Reset the log file."""
        if self.model_name is not None:
            os.makedirs("generations", exist_ok=True)
            self.log_file = f"generations/{self.model_name}.txt"
            with open(self.log_file, "w") as f:
                f.write(f"=== Generations for {self.model_name} ===\n")
                f.write(f"(dual pad modes: fixed, power2)\n\n")

    def on_train_batch_end(self, params: Any, batch: int):
        """Called at the end of a training batch."""
        self.step_counter += 1

        if self.step_counter % self.trigger_every_n_steps == 0:
            print(f"\n--- Generation at step {self.step_counter} ---")
            # Generate with fixed padding
            txt_fixed = generate_text(
                model=self.model,
                params=params,
                tokenizer=self.tokenizer,
                prompt=self.prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                maxlen=self.maxlen,
                pad_mode="fixed",
                verbose=True,
                seed=self.seed,
            )
            print("\n-----\n")
            # Generate with power-of-2 padding (to benefit from fewer recompilations)
            txt_power2 = generate_text(
                model=self.model,
                params=params,
                tokenizer=self.tokenizer,
                prompt=self.prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                maxlen=self.maxlen,
                pad_mode="power2",
                verbose=True,
                seed=self.seed,
            )

            if self.log_file is not None:
                with open(self.log_file, "a") as f:
                    f.write(f"--- Step {self.step_counter} (fixed) ---\n")
                    f.write(txt_fixed + "\n\n")
                    f.write("-----\n\n")
                    f.write(f"--- Step {self.step_counter} (power2) ---\n")
                    f.write(txt_power2 + "\n\n")


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def generate_text(
    model,
    params: Any,
    tokenizer: Any,
    prompt: str = "<|im_start|>user\n",
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.5,
    maxlen: int = 768,
    pad_mode: str = "fixed",  # "fixed" pads to maxlen, "power2" pads to next power of 2
    verbose: bool = True,
    seed: Optional[int] = None,
) -> str:
    """
    Generate text from a language model.

    Args:
        model: Flax model with apply method
        params: Model parameters
        tokenizer: Tokenizer with __call__ and decode methods
        prompt: Initial text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Number of top tokens to consider
        top_p: Cumulative probability threshold for nucleus sampling
        repetition_penalty: Penalty for repeating tokens
        maxlen: Maximum sequence length
        pad_mode: Padding strategy for generation ("fixed" or "power2")
        verbose: Print tokens as they are generated
        seed: Random seed for reproducibility

    Returns:
        Generated text string
    """
    if seed is not None:
        np.random.seed(seed)

    tokens = tokenizer([prompt])[0]
    generated_tokens = []
    text = prompt

    for _ in range(max_new_tokens):
        seq_len = len(tokens)
        if pad_mode == "power2":
            padded_len = min(next_power_of_2(seq_len), maxlen)
            padded = tokens + [0] * (padded_len - seq_len)
            padded = padded[:padded_len]
        else:
            # default to fixed padding
            padded = tokens + [0] * (maxlen - seq_len)
            padded = padded[:maxlen]
        input_array = jnp.array([padded], dtype=jnp.int32)

        # Get logits
        logits = model.apply({"params": params}, input_array, deterministic=True)
        logits_np = np.array(logits[0, len(tokens) - 1], dtype=np.float64)

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

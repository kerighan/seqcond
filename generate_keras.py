#!/usr/bin/env python3
"""Interactive text generation with Keras SeqCond model.

Usage:
    python generate_keras.py
    python generate_keras.py --prompt "What is 2+2?"
    python generate_keras.py --checkpoint checkpoints/seqcond_torch_762k.pt
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import time
import sys
import numpy as np


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def load_keras_model(torch_path):
    """Load Keras model with converted weights from torch checkpoint."""
    from convert_torch_to_keras import (
        load_torch_checkpoint,
        build_keras_model,
        convert_weights,
    )

    print(f"Loading checkpoint: {torch_path}")
    config, state_dict = load_torch_checkpoint(torch_path)
    print("Building Keras model...")
    keras_model = build_keras_model(config)
    print("Converting weights...")
    convert_weights(config, state_dict, keras_model)
    print(f"Model ready: {keras_model.count_params():,} parameters\n")
    return keras_model, config


def load_tokenizer():
    """Load tokenizer (same as generate.py)."""
    try:
        from seqcond.dataset import Tokenizer

        tokenizer = Tokenizer()
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Please install convectors: pip install convectors")
        return None


def sample_token(logits, temperature=0.6, top_p=0.9, top_k=50):
    """Sample next token from logits with temperature, top-p, and top-k."""
    logits = np.array(logits).astype(np.float64)

    if temperature <= 0.0:
        return int(np.argmax(logits))

    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
        top_k_logits = logits[top_k_indices]
        top_k_probs = np.exp(top_k_logits - np.max(top_k_logits))
        top_k_probs = top_k_probs / np.sum(top_k_probs)

        # Top-p (nucleus) filtering on top-k
        if top_p < 1.0:
            sorted_indices = np.argsort(top_k_probs)[::-1]
            cumsum = np.cumsum(top_k_probs[sorted_indices])
            cutoff = np.searchsorted(cumsum, top_p)
            nucleus_indices = sorted_indices[: cutoff + 1]
            nucleus_probs = top_k_probs[nucleus_indices]
            nucleus_probs = nucleus_probs / np.sum(nucleus_probs)
            sampled_idx = np.random.choice(nucleus_indices, p=nucleus_probs)
            return int(top_k_indices[sampled_idx])
        else:
            sampled_idx = np.random.choice(len(top_k_probs), p=top_k_probs)
            return int(top_k_indices[sampled_idx])
    else:
        # Softmax
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)

        # Top-p filtering
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_indices])
            cutoff = np.searchsorted(cumsum, top_p)
            nucleus_indices = sorted_indices[: cutoff + 1]
            nucleus_probs = probs[nucleus_indices]
            nucleus_probs = nucleus_probs / np.sum(nucleus_probs)
            sampled_idx = np.random.choice(nucleus_indices, p=nucleus_probs)
            return int(sampled_idx)
        else:
            return int(np.random.choice(len(probs), p=probs))


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=512,
    temperature=0.6,
    top_p=0.9,
    top_k=50,
    stream=True,
):
    """Generate text from prompt using Keras model."""

    # Encode prompt (returns list of token IDs)
    token_list = tokenizer.encode(prompt)
    input_ids = np.array([token_list], dtype=np.int32)  # (1, L)

    B = 1
    prompt_len = input_ids.shape[1]

    # Get EOS token ID
    eos_token_id = tokenizer.encode("<|im_end|>")[0]

    # Prefill: process prompt and initialize state
    states = model.init_state(batch_size=B)
    for t in range(prompt_len):
        _, states = model.step(input_ids[:, t : t + 1], states)

    # Get first token from forward pass
    fwd_logits = np.array(model(input_ids, training=False))
    first_token = sample_token(fwd_logits[0, -1, :], temperature, top_p, top_k)

    generated_tokens = [first_token]
    current_token = np.array([[first_token]], dtype=np.int32)

    # Stream first token
    if stream:
        text = tokenizer.decode([first_token])
        yield text

    # Generate remaining tokens
    for _ in range(max_new_tokens - 1):
        logits, states = model.step(current_token, states)
        logits_np = np.array(logits)[0]  # (vocab_size,)

        next_token = sample_token(logits_np, temperature, top_p, top_k)
        generated_tokens.append(next_token)
        current_token = np.array([[next_token]], dtype=np.int32)

        # Stream token
        if stream:
            text = tokenizer.decode([next_token])
            yield text

        # Stop on EOS
        if next_token == eos_token_id:
            break

    # Return full sequence if not streaming
    if not stream:
        yield tokenizer.decode(generated_tokens)


def format_prompt(question, use_synth_template=True):
    """Format question as chat prompt."""
    if use_synth_template:
        return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    else:
        return question


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with Keras SeqCond model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/seqcond_torch_762k.pt",
        help="Path to PyTorch checkpoint (will be converted to Keras)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text (if not provided, enters interactive mode)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.6,
        help="Sampling temperature (0.0 for greedy)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k filtering (0 to disable)",
    )
    parser.add_argument(
        "--no_template",
        action="store_true",
        help="Don't use chat template formatting",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Don't stream tokens (print all at once)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return 1

    # Load model and tokenizer
    model, config = load_keras_model(args.checkpoint)
    tokenizer = load_tokenizer()

    if tokenizer is None:
        print("\nError: Tokenizer is required for text generation.")
        return 1

    # Interactive or single-shot mode
    if args.prompt is None:
        # Interactive mode
        print("=" * 60)
        print("Interactive Keras SeqCond Generation")
        print("=" * 60)
        print("Type your question and press Enter. Type 'quit' to exit.\n")

        while True:
            try:
                question = input(f"{bcolors.OKGREEN}You:{bcolors.ENDC} ")
                if question.strip().lower() in ["quit", "exit", "q"]:
                    break

                if not question.strip():
                    continue

                prompt = format_prompt(
                    question, use_synth_template=not args.no_template
                )

                print(f"{bcolors.OKCYAN}Assistant:{bcolors.ENDC} ", end="", flush=True)
                start = time.time()

                current_color = bcolors.OKBLUE if not args.no_template else ""
                token_count = 0

                for token in generate(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temp,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    stream=not args.quiet,
                ):
                    token_count += 1

                    # Handle special tokens for coloring
                    if token == "<|think_end|>":
                        print()
                        current_color = ""
                    elif token == "<|think_start|>":
                        current_color = bcolors.OKBLUE
                    elif token == "<|im_end|>":
                        print()
                        break
                    else:
                        if current_color:
                            print(
                                current_color + token + bcolors.ENDC, end="", flush=True
                            )
                        else:
                            print(token, end="", flush=True)

                duration = time.time() - start
                print(
                    f"\n{bcolors.WARNING}[{token_count} tokens, {duration:.2f}s, {token_count/duration:.1f} tok/s]{bcolors.ENDC}\n"
                )

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except EOFError:
                break

    else:
        # Single-shot mode
        prompt = format_prompt(args.prompt, use_synth_template=not args.no_template)
        print(f"Prompt: {args.prompt}")
        print(f"Generating (temp={args.temp}, max_tokens={args.max_tokens})...\n")

        start = time.time()
        current_color = bcolors.OKBLUE if not args.no_template else ""
        token_count = 0

        for token in generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_p=args.top_p,
            top_k=args.top_k,
            stream=not args.quiet,
        ):
            token_count += 1

            if token == "<|think_end|>":
                print()
                current_color = ""
            elif token == "<|think_start|>":
                current_color = bcolors.OKBLUE
            elif token == "<|im_end|>":
                print()
                break
            else:
                if current_color:
                    print(current_color + token + bcolors.ENDC, end="", flush=True)
                else:
                    print(token, end="", flush=True)

        duration = time.time() - start
        print(f"\n\nStats:")
        print(f"  Tokens: {token_count}")
        print(f"  Time: {duration:.2f}s")
        print(f"  Speed: {token_count/duration:.1f} tokens/sec")

    return 0


if __name__ == "__main__":
    sys.exit(main())

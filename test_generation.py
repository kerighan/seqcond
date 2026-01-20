"""
Test text generation from a checkpoint.

Usage:
    python test_generation.py --checkpoint checkpoints/model_step10000.pkl --prompt "Hello"
    python test_generation.py --checkpoint checkpoints/model.pkl --max-tokens 256
"""

import argparse
import jax
import jax.numpy as jnp
import numpy as np

from seqcond.config import Config, ModelConfig
from seqcond.jax.train import create_model_from_config, load_checkpoint
from seqcond.jax.callback import generate_text
from seqcond.dataset import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step1.pkl",
        # required=True,
        help="Path to checkpoint file (.pkl)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="<|im_start|>user\nWhat is the capital of France?\n<|im_start|>assistant\n",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.2, help="Repetition penalty"
    )
    parser.add_argument(
        "--pad-mode",
        type=str,
        default="power2",
        choices=["fixed", "power2"],
        help="Padding mode for generation",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode for multiple prompts",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Loading checkpoint...")
    print("=" * 60)

    # Load checkpoint
    params, config_dict, step, _ = load_checkpoint(args.checkpoint)
    print(f"Loaded checkpoint from step {step}")

    # Reconstruct model config from checkpoint
    # Deep copy to avoid modifying original
    model_config_dict = config_dict["model"].copy()

    # Check if there are specific overrides needed for initialization
    # In some JAX/Flax setups, we need to match exactly the parameters
    model_config = ModelConfig(**model_config_dict)

    print(f"Model type: {model_config.model_type}")
    print(
        f"Model size: d_model={model_config.d_model}, layers={model_config.num_layers}"
    )
    print(f"Max sequence length: {model_config.maxlen}")
    print(f"Num heads: {model_config.num_heads}")
    print(f"Num query heads: {model_config.num_query_heads}")
    print(f"Seqcond heads: {model_config.seqcond_heads}")

    # Create model
    print("\nCreating model...")
    model = create_model_from_config(model_config)

    # Load tokenizer (tiktoken-based, same as training)
    print("Loading tokenizer...")
    tokenizer = Tokenizer()

    print("\n" + "=" * 60)
    print("Generation")
    print("=" * 60)

    if args.interactive:
        print("Interactive mode. Type 'quit' to exit.\n")
        while True:
            try:
                prompt = input("Prompt: ")
                if prompt.lower() == "quit":
                    break
                if not prompt:
                    continue

                print("\nGenerating...\n")
                text = generate_text(
                    model=model,
                    params=params,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    maxlen=model_config.maxlen,
                    pad_mode=args.pad_mode,
                    verbose=True,
                    seed=args.seed,
                )
                print("\n")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    else:
        print(f"Prompt: {args.prompt}\n")
        print("Generated text:\n")

        text = generate_text(
            model=model,
            params=params,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            maxlen=model_config.maxlen,
            pad_mode=args.pad_mode,
            verbose=True,
            seed=args.seed,
        )

        print("\n\n" + "=" * 60)
        print("Full output:")
        print("=" * 60)
        print(text)


if __name__ == "__main__":
    main()

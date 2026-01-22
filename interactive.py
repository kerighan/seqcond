#!/usr/bin/env python3
"""Interactive REPL for SeqCond model generation."""

import argparse
from seqcond.torch.generator import TorchGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Interactive text generation with SeqCond PyTorch model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/seqcond_torch2_60k.pt",
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=0.7, help="Temperature for sampling"
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument(
        "--rep_penalty", type=float, default=1.0, help="Repetition penalty"
    )
    parser.add_argument(
        "--freq_penalty", type=float, default=0.0, help="Frequency penalty"
    )
    parser.add_argument(
        "--no_repeat_ngram",
        type=int,
        default=0,
        help="Prevent repeating n-grams of this size",
    )
    parser.add_argument(
        "--no_cuda_graph",
        action="store_true",
        help="Disable CUDA graph optimization",
    )

    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    try:
        gen = TorchGenerator(args.checkpoint)
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    # Pre-capture CUDA graphs for fast generation
    if not args.no_cuda_graph:
        gen.precompute(max_seq_len=1024)
        print()

    print("=" * 60)
    print("Interactive SeqCond Generation")
    print("=" * 60)
    print(f"Settings: max_tokens={args.max_tokens}, temp={args.temp}")
    print("Type your prompt and press Enter. Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    print()

    while True:
        try:
            # Get user input
            prompt = input(">>> ").strip()

            if not prompt:
                continue

            if prompt.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            # Generate response
            output = gen.generate(
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temp,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.rep_penalty,
                frequency_penalty=args.freq_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram,
                verbose=False,
                use_cuda_graph=not args.no_cuda_graph,
            )

            # Stop at STOP token (user will replace STOP)
            stop_marker = "<|endoftext|>"
            if stop_marker in output:
                output = output.split(stop_marker)[0]

            # Print full response (prompt + generation)
            print(output)
            print()

        except KeyboardInterrupt:
            print("Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print()


if __name__ == "__main__":
    main()

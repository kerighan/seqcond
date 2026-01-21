import torch
import argparse
import time
from seqcond.torch.generator import TorchGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using SeqCond PyTorch model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/seqcond_torch_20k.pt",
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--prompt", type=str, default="The quick brown fox", help="Prompt text"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=0.7, help="Sampling temperature (0.0 for greedy)"
    )
    parser.add_argument(
        "--no_cuda_graph", action="store_true", help="Disable CUDA Graphs optimization"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Disable token streaming output"
    )

    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    try:
        gen = TorchGenerator(args.checkpoint)
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    print(f"\nPrompt: '{args.prompt}'")
    print(f"Generating {args.max_tokens} tokens (temp={args.temp})...\n")

    start = time.time()

    # Warmup if using CUDA graphs to ensure accurate timing of the actual run
    # and to capture the graph if it's the first run
    if not args.no_cuda_graph:
        if not args.quiet:
            print("Initializing CUDA Graphs (warmup)...")
        gen.generate(
            args.prompt,
            max_new_tokens=5,
            temperature=args.temp,
            verbose=False,
            use_cuda_graph=True,
        )
        if not args.quiet:
            print("Ready.\n")
        start = time.time()  # Reset start time after warmup

    output = gen.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
        verbose=not args.quiet,
        use_cuda_graph=not args.no_cuda_graph,
    )

    torch.cuda.synchronize()
    duration = time.time() - start

    print("-" * 60)
    if args.quiet:
        print(output)
        print("-" * 60)
    else:
        print()  # Newline after streaming output

    tps = args.max_tokens / duration
    print(f"\nStats:")
    print(f"  Time: {duration:.2f}s")
    print(f"  Speed: {tps:.2f} tokens/sec")
    print(f"  Latency: {duration*1000/args.max_tokens:.2f} ms/token")


if __name__ == "__main__":
    main()

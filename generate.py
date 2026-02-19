import torch
import argparse
import time
from seqcond.torch.generator import TorchGenerator


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


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using SeqCond PyTorch model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        # default="checkpoints/seqcond_opt_torch.pt",
        default="checkpoints/seqcond_torch_310k.pt",
        # default="checkpoints/thin_torch.pt",
        # default="checkpoints/transformer_torch.pt",
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--prompt", type=str, default="The quick brown fox", help="Prompt text"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=768,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=0.6, help="Sampling temperature (0.0 for greedy)"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Nucleus sampling top-p probability"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k filtering (default 50, 0 to disable)",
    )
    parser.add_argument(
        "--no_cuda_graph", action="store_true", help="Disable CUDA Graphs optimization"
    )
    parser.add_argument(
        "--rep_penalty",
        type=float,
        default=1.2,
        help="Repetition penalty (1.0 = no penalty, >1.0 penalizes repetition)",
    )
    parser.add_argument(
        "--freq_penalty",
        type=float,
        default=0.0,
        help="Frequency penalty (0.0 = no penalty, additive penalty per occurrence)",
    )
    parser.add_argument(
        "--no_repeat_ngram",
        type=int,
        default=0,
        help="Block repeated n-grams of this size (0 = disabled, try 3-6 for paragraphs)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Disable token streaming output"
    )
    parser.add_argument(
        "--use_triton",
        action="store_true",
        help="Use Triton kernels for SeqCond acceleration (requires triton package)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (float32, float16, bfloat16)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference (disables CUDA graphs)",
    )

    args = parser.parse_args()

    # CPU mode implies no CUDA graphs
    if args.cpu:
        args.no_cuda_graph = True

    device = "cpu" if args.cpu else "cuda"
    print(f"Loading model from {args.checkpoint} (device={device})...")
    try:
        gen = TorchGenerator(args.checkpoint, device=device, dtype=args.dtype)
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    print(f"\nPrompt: '{args.prompt}'")
    print(f"Generating {args.max_tokens} tokens (temp={args.temp})...\n")

    # Reset memory stats before generation
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.time()

    # Pre-capture CUDA graphs for fast generation
    if not args.no_cuda_graph:
        gen.precompute(max_seq_len=1024, use_triton=args.use_triton)
        start = time.time()  # Reset start time after precompute

    generator = gen.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.rep_penalty,
        frequency_penalty=args.freq_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram,
        verbose=not args.quiet,
        use_cuda_graph=not args.no_cuda_graph,
        use_triton=args.use_triton,
        use_synth_template=True,
    )
    output = []
    current_color = bcolors.OKBLUE
    print()
    for token in generator:
        if token == "<|think_end|>":
            print()
            current_color = ""
        elif token == "<|think_start|>":
            current_color = bcolors.OKBLUE
        elif token == "<|im_end|>":
            print()
            continue
        else:
            if current_color:
                print(current_color + token + bcolors.ENDC, end="", flush=True)
            else:
                print(token, end="", flush=True)
        output.append(token)

    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.synchronize()
    duration = time.time() - start
    print()
    print("-" * 60)
    if args.quiet:
        print("".join(output))
        print("-" * 60)
    else:
        print()  # Newline after streaming output

    # Count actual generated tokens (output length - prompt length)
    # prompt_tokens = len(gen.tokenizer.encode(args.prompt))
    if hasattr(gen, "tokenizer"):
        total_tokens = len(output)
        generated_tokens = max(1, total_tokens)
    else:
        generated_tokens = args.max_tokens

    tps = generated_tokens / duration
    print(f"\nStats:")
    print(f"  Tokens generated: {generated_tokens}")
    print(f"  Time: {duration:.2f}s")
    print(f"  Speed: {tps:.2f} tokens/sec")
    print(f"  Latency: {duration*1000/generated_tokens:.2f} ms/token")

    # Memory stats
    if torch.cuda.is_available() and not args.cpu:
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
        current_mem = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved_mem = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"\nMemory:")
        print(f"  Peak allocated: {peak_mem:.2f} GB")
        print(f"  Current allocated: {current_mem:.2f} GB")
        print(f"  Reserved: {reserved_mem:.2f} GB")


if __name__ == "__main__":
    main()

"""
Benchmark generation speed across batch sizes and backends.

Usage:
    python benchmark_gen.py
    python benchmark_gen.py --checkpoint checkpoints/seqcond_lin5.pt
    python benchmark_gen.py --max_tokens 256 --batch_sizes 1,2,4,8
"""

import argparse
import time

import torch
import numpy as np

from seqcond.torch.model import SeqCondModel
from seqcond.dataset import Tokenizer


PROMPT = "<|im_start|>user\nWhat is blood pressure?\n<|im_end|><|im_start|>assistant\n<|think_start|>"


def load_model(checkpoint_path, device="cuda"):
    data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = data["config"]
    model = SeqCondModel(**config).to(device).eval()
    model.load_state_dict(data["state_dict"], strict=False)
    n = sum(p.numel() for p in model.parameters())
    print(f"Loaded {checkpoint_path}  ({n:,} params, maxlen={config.get('maxlen')})")
    return model, config


def benchmark_once(model, tokenizer, prompt_toks, batch_size, max_tokens,
                   use_triton=False, use_seq_len=True, warmup=False):
    """Run one generation benchmark. Returns (tokens_generated, elapsed_seconds)."""
    device = next(model.parameters()).device
    eos_id = tokenizer.encode("<|im_end|>")[0]
    input_ids = torch.tensor([prompt_toks], device=device)

    with torch.no_grad():
        # Prefill
        logits, states = model.prefill(input_ids)
        logits = logits.squeeze(1)

        if batch_size > 1:
            logits = logits.repeat(batch_size, 1)
            states = [
                tuple(s.repeat(batch_size, *([1] * (s.ndim - 1))) for s in state)
                for state in states
            ]

        token_buf = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        current_seq_len = len(prompt_toks)

        # Warmup GPU (not timed)
        if warmup:
            for _ in range(3):
                current_seq_len += 1
                sl = current_seq_len if use_seq_len else None
                logits, states = model.step(
                    token_buf, states, seq_len=sl, use_triton=use_triton
                )
            # Re-prefill to reset states
            logits, states = model.prefill(input_ids)
            logits = logits.squeeze(1)
            if batch_size > 1:
                logits = logits.repeat(batch_size, 1)
                states = [
                    tuple(s.repeat(batch_size, *([1] * (s.ndim - 1))) for s in state)
                    for state in states
                ]
            current_seq_len = len(prompt_toks)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        total_tokens = 0
        for step in range(max_tokens):
            # Greedy sample (no fancy sampling — pure speed test)
            next_tokens = torch.argmax(logits, dim=-1)
            token_buf[:, 0] = next_tokens
            total_tokens += batch_size

            current_seq_len += 1
            sl = current_seq_len if use_seq_len else None
            logits, states = model.step(
                token_buf, states, seq_len=sl, use_triton=use_triton
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    return total_tokens, elapsed


def run_benchmark(model, tokenizer, prompt_toks, batch_sizes, max_tokens,
                  use_triton, use_seq_len, label, n_runs=3):
    """Run benchmark across batch sizes, return results dict."""
    results = {}
    for bs in batch_sizes:
        # Warmup run
        benchmark_once(model, tokenizer, prompt_toks, bs, min(max_tokens, 10),
                       use_triton=use_triton, use_seq_len=use_seq_len, warmup=True)

        speeds = []
        for run in range(n_runs):
            torch.cuda.empty_cache()
            total_tokens, elapsed = benchmark_once(
                model, tokenizer, prompt_toks, bs, max_tokens,
                use_triton=use_triton, use_seq_len=use_seq_len,
            )
            tok_per_sec = total_tokens / elapsed
            speeds.append(tok_per_sec)

        mean_speed = np.mean(speeds)
        std_speed = np.std(speeds)
        results[bs] = (mean_speed, std_speed)
        print(f"  {label}  batch={bs:2d}  →  {mean_speed:7.1f} ± {std_speed:4.1f} tok/s  "
              f"({mean_speed/bs:6.1f} tok/s/sample)  [{n_runs} runs × {max_tokens} steps]")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark SeqCond generation")
    parser.add_argument("--checkpoint", default="checkpoints/seqcond_lin5.pt")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Number of decode steps per run")
    parser.add_argument("--batch_sizes", default="1,2,4,8",
                        help="Comma-separated batch sizes to test")
    parser.add_argument("--n_runs", type=int, default=3,
                        help="Number of runs per configuration (for averaging)")
    parser.add_argument("--prompt", default=PROMPT,
                        help="Prompt string (already tokenized with chat template)")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    model, config = load_model(args.checkpoint)
    tokenizer = Tokenizer()
    prompt_toks = tokenizer([args.prompt])[0]
    print(f"Prompt length: {len(prompt_toks)} tokens")
    print(f"Decode steps: {args.max_tokens}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Runs per config: {args.n_runs}")
    print()

    # ── 1. PyTorch, no seq_len (baseline) ─────────────────────────────
    print("=" * 70)
    print("  PyTorch backend, NO seq_len (full KV cache scan — old behavior)")
    print("=" * 70)
    res_pt_nosl = run_benchmark(
        model, tokenizer, prompt_toks, batch_sizes, args.max_tokens,
        use_triton=False, use_seq_len=False, label="PT  no_sl", n_runs=args.n_runs,
    )

    # ── 2. PyTorch, with seq_len (new fix) ────────────────────────────
    print()
    print("=" * 70)
    print("  PyTorch backend, WITH seq_len (optimized KV cache — new behavior)")
    print("=" * 70)
    res_pt_sl = run_benchmark(
        model, tokenizer, prompt_toks, batch_sizes, args.max_tokens,
        use_triton=False, use_seq_len=True, label="PT  sl   ", n_runs=args.n_runs,
    )

    # ── 3. Triton, no seq_len ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Triton backend, NO seq_len (full KV cache scan)")
    print("=" * 70)
    res_tr_nosl = run_benchmark(
        model, tokenizer, prompt_toks, batch_sizes, args.max_tokens,
        use_triton=True, use_seq_len=False, label="TRI no_sl", n_runs=args.n_runs,
    )

    # ── 4. Triton, with seq_len (best config) ─────────────────────────
    print()
    print("=" * 70)
    print("  Triton backend, WITH seq_len (optimized — best config)")
    print("=" * 70)
    res_tr_sl = run_benchmark(
        model, tokenizer, prompt_toks, batch_sizes, args.max_tokens,
        use_triton=True, use_seq_len=True, label="TRI sl   ", n_runs=args.n_runs,
    )

    # ── Summary table ─────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  SUMMARY  (tok/s, higher is better)")
    print("=" * 70)
    header = f"{'batch':>5} | {'PT no_sl':>10} | {'PT sl':>10} | {'TRI no_sl':>10} | {'TRI sl':>10} | {'speedup':>8}"
    print(header)
    print("-" * len(header))
    for bs in batch_sizes:
        pt_nosl = res_pt_nosl[bs][0]
        pt_sl = res_pt_sl[bs][0]
        tr_nosl = res_tr_nosl[bs][0]
        tr_sl = res_tr_sl[bs][0]
        speedup = tr_sl / pt_nosl if pt_nosl > 0 else 0
        print(f"{bs:5d} | {pt_nosl:10.1f} | {pt_sl:10.1f} | {tr_nosl:10.1f} | {tr_sl:10.1f} | {speedup:7.2f}x")

    print()
    print("speedup = Triton+seq_len vs PyTorch baseline (no seq_len)")

    # Memory
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nPeak GPU memory: {peak:.2f} GB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark generation speed (tokens/sec) across models.

Compares SeqCond against HuggingFace models with:
1. Overall tokens/sec
2. Time per token at different positions (to show transformer O(n) scaling)
3. Clean summary tables
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from seqcond.torch.generator import TorchGenerator

# Model registry (same as eval_creative_writing.py)
MODEL_REGISTRY = {
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "lfm2-350m": "LiquidAI/LFM2-350M",
    "lfm2.5-350m": "LiquidAI/LFM2.5-350M",
    "granite-350m": "ibm-granite/granite-4.0-h-350M",
    "baguettotron": "PleIAs/Baguettotron",
}

# Prompts designed to elicit long responses
LONG_PROMPTS = [
    "Write a detailed 2000-word essay explaining the history of artificial intelligence from its origins to the present day, covering key milestones, important researchers, and major breakthroughs.",
    "Write an extremely long and detailed story about a space explorer who discovers an ancient alien civilization. Include vivid descriptions, dialogue, and plot twists. Make it at least 1500 words.",
    "Explain in great detail how a computer processor works, from transistors to machine code to high-level programming. Be thorough and technical, covering at least 1000 words.",
]


def count_parameters(model) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gpu_memory_mb() -> float:
    """Get current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def benchmark_seqcond(
    checkpoint: str,
    prompts: List[str],
    max_tokens: int,
    device: str = "cuda",
    use_triton: bool = False,
    use_cuda_graph: bool = True,
) -> Dict:
    """Benchmark SeqCond model and return timing statistics."""
    print(f"\n{'='*60}")
    print(f"Benchmarking SeqCond: {checkpoint}")
    print(f"{'='*60}")

    # Reset GPU memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    gen = TorchGenerator(checkpoint, device=device)
    num_params = count_parameters(gen.model)
    print(f"Parameters: {num_params / 1e6:.1f}M")

    # Measure model memory
    model_memory_mb = get_gpu_memory_mb()
    print(f"Model GPU memory: {model_memory_mb:.1f} MB")

    # Warmup with CUDA graph pre-capture
    if use_cuda_graph and device == "cuda":
        gen.precompute(max_seq_len=max_tokens + 512, use_triton=use_triton)
    else:
        # Simple warmup without CUDA graphs
        print("Warming up...")
        _ = list(
            gen.generate(
                "Hello",
                max_new_tokens=50,
                temperature=0.7,
                use_cuda_graph=False,
                use_triton=use_triton,
            )
        )
    torch.cuda.synchronize() if device == "cuda" else None

    all_times_per_token = []
    all_tokens_generated = []
    all_total_times = []
    all_prefill_times = []
    peak_memory_mb = model_memory_mb

    for i, prompt in enumerate(prompts):
        print(f"\n  Prompt {i+1}/{len(prompts)}...")

        # Track time per token
        token_times = []
        tokens_generated = []

        # Prefill timing
        torch.cuda.synchronize() if device == "cuda" else None
        start_total = time.perf_counter()

        token_count = 0
        last_time = start_total

        for token in gen.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.7,
            use_cuda_graph=use_cuda_graph,
            use_triton=use_triton,
        ):
            torch.cuda.synchronize() if device == "cuda" else None
            now = time.perf_counter()

            # Track peak memory
            if device == "cuda":
                current_mem = torch.cuda.max_memory_allocated() / 1024**2
                peak_memory_mb = max(peak_memory_mb, current_mem)

            if token_count == 0:
                # First token includes prefill
                prefill_time = now - start_total
                all_prefill_times.append(prefill_time)
            else:
                token_times.append(now - last_time)

            last_time = now
            token_count += 1

        torch.cuda.synchronize() if device == "cuda" else None
        total_time = time.perf_counter() - start_total

        all_times_per_token.append(token_times)
        all_tokens_generated.append(token_count)
        all_total_times.append(total_time)

        print(
            f"    Generated {token_count} tokens in {total_time:.2f}s ({token_count/total_time:.1f} tok/s)"
        )

    return {
        "model": "seqcond",
        "checkpoint": checkpoint,
        "num_params": num_params,
        "model_memory_mb": model_memory_mb,
        "peak_memory_mb": peak_memory_mb,
        "times_per_token": all_times_per_token,
        "tokens_generated": all_tokens_generated,
        "total_times": all_total_times,
        "prefill_times": all_prefill_times,
    }


def benchmark_hf_model(
    model_name: str,
    prompts: List[str],
    max_tokens: int,
    device: str = "cuda",
    warmup_runs: int = 1,
) -> Dict:
    """Benchmark HuggingFace model and return timing statistics."""
    print(f"\n{'='*60}")
    print(f"Benchmarking HF: {model_name}")
    print(f"{'='*60}")

    # Reset GPU memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    hf_path = MODEL_REGISTRY[model_name]
    print(f"Loading {hf_path}...")

    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_params = count_parameters(model)
    print(f"Parameters: {num_params / 1e6:.1f}M")

    # Measure model memory
    model_memory_mb = get_gpu_memory_mb()
    print(f"Model GPU memory: {model_memory_mb:.1f} MB")

    # Warmup
    print(f"Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        inputs = tokenizer("Hello", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}
        with torch.no_grad():
            _ = model.generate(
                **inputs, max_new_tokens=50, do_sample=True, temperature=0.7
            )
    torch.cuda.synchronize() if device == "cuda" else None

    all_times_per_token = []
    all_tokens_generated = []
    all_total_times = []
    all_prefill_times = []
    peak_memory_mb = model_memory_mb

    # Check for chat template
    has_chat_template = (
        hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    )

    for i, prompt in enumerate(prompts):
        print(f"\n  Prompt {i+1}/{len(prompts)}...")

        # Format prompt
        if has_chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        inputs = tokenizer(formatted, return_tensors="pt")
        # Filter out token_type_ids if present (not all models use it)
        inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}
        input_len = inputs["input_ids"].shape[1]

        # We need to generate token by token to measure per-token time
        # Use generate with output_scores and return_dict_in_generate
        torch.cuda.synchronize() if device == "cuda" else None
        start_total = time.perf_counter()

        token_times = []
        generated_ids = inputs["input_ids"].clone()
        past_key_values = None

        with torch.no_grad():
            # Prefill
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

            torch.cuda.synchronize() if device == "cuda" else None
            prefill_time = time.perf_counter() - start_total
            all_prefill_times.append(prefill_time)

            # Sample first token
            probs = torch.softmax(next_token_logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            last_time = time.perf_counter()

            # Generate remaining tokens
            for _ in range(max_tokens - 1):
                torch.cuda.synchronize() if device == "cuda" else None
                step_start = time.perf_counter()

                outputs = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]

                # Sample
                probs = torch.softmax(next_token_logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                torch.cuda.synchronize() if device == "cuda" else None
                token_times.append(time.perf_counter() - step_start)

                # Track peak memory
                if device == "cuda":
                    current_mem = torch.cuda.max_memory_allocated() / 1024**2
                    peak_memory_mb = max(peak_memory_mb, current_mem)

                # Check EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break

                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        torch.cuda.synchronize() if device == "cuda" else None
        total_time = time.perf_counter() - start_total
        token_count = generated_ids.shape[1] - input_len

        all_times_per_token.append(token_times)
        all_tokens_generated.append(token_count)
        all_total_times.append(total_time)

        print(
            f"    Generated {token_count} tokens in {total_time:.2f}s ({token_count/total_time:.1f} tok/s)"
        )

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "hf_path": hf_path,
        "num_params": num_params,
        "model_memory_mb": model_memory_mb,
        "peak_memory_mb": peak_memory_mb,
        "times_per_token": all_times_per_token,
        "tokens_generated": all_tokens_generated,
        "total_times": all_total_times,
        "prefill_times": all_prefill_times,
    }


def compute_stats(results: Dict) -> Dict:
    """Compute aggregate statistics from benchmark results."""
    total_tokens = sum(results["tokens_generated"])
    total_time = sum(results["total_times"])

    # Flatten all per-token times
    all_token_times = []
    for times in results["times_per_token"]:
        all_token_times.extend(times)

    # Compute time at different positions (binned)
    position_bins = {}
    for prompt_times in results["times_per_token"]:
        for pos, t in enumerate(prompt_times):
            bin_idx = pos // 100  # Bin by 100 tokens
            if bin_idx not in position_bins:
                position_bins[bin_idx] = []
            position_bins[bin_idx].append(t)

    position_stats = {}
    for bin_idx, times in sorted(position_bins.items()):
        position_stats[f"{bin_idx*100}-{(bin_idx+1)*100}"] = {
            "mean_ms": np.mean(times) * 1000,
            "std_ms": np.std(times) * 1000,
            "count": len(times),
        }

    return {
        "model": results["model"],
        "num_params_m": results["num_params"] / 1e6,
        "model_memory_mb": results["model_memory_mb"],
        "peak_memory_mb": results["peak_memory_mb"],
        "kv_cache_mb": results["peak_memory_mb"] - results["model_memory_mb"],
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "overall_tok_per_s": total_tokens / total_time if total_time > 0 else 0,
        "mean_prefill_ms": np.mean(results["prefill_times"]) * 1000,
        "mean_token_ms": np.mean(all_token_times) * 1000 if all_token_times else 0,
        "std_token_ms": np.std(all_token_times) * 1000 if all_token_times else 0,
        "position_stats": position_stats,
    }


def print_summary_table(all_stats: List[Dict]):
    """Print a clean summary table."""
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY")
    print("=" * 90)

    # Header
    print(
        f"{'Model':<20} {'Params':>8} {'Model':>8} {'Peak':>8} {'KV':>8} {'Tok/s':>8} {'Prefill':>9} {'Per-tok':>9}"
    )
    print(
        f"{'':20} {'(M)':>8} {'(MB)':>8} {'(MB)':>8} {'(MB)':>8} {'':>8} {'(ms)':>9} {'(ms)':>9}"
    )
    print("-" * 100)

    # Sort by tok/s descending
    sorted_stats = sorted(all_stats, key=lambda x: x["overall_tok_per_s"], reverse=True)

    for s in sorted_stats:
        print(
            f"{s['model']:<20} {s['num_params_m']:>8.1f} {s['model_memory_mb']:>8.0f} {s['peak_memory_mb']:>8.0f} "
            f"{s['kv_cache_mb']:>8.0f} {s['overall_tok_per_s']:>8.1f} {s['mean_prefill_ms']:>9.1f} {s['mean_token_ms']:>9.2f}"
        )

    print("=" * 100)


def print_position_table(all_stats: List[Dict]):
    """Print table showing time per token at different positions."""
    print("\n" + "=" * 100)
    print("TIME PER TOKEN BY POSITION (ms) - Shows transformer O(n) scaling")
    print("=" * 100)

    # Collect all position bins
    all_bins = set()
    for s in all_stats:
        all_bins.update(s["position_stats"].keys())
    sorted_bins = sorted(all_bins, key=lambda x: int(x.split("-")[0]))

    # Header
    header = f"{'Model':<20}"
    for bin_name in sorted_bins[:6]:  # Limit to first 6 bins
        header += f" {bin_name:>12}"
    print(header)
    print("-" * 100)

    for s in all_stats:
        row = f"{s['model']:<20}"
        for bin_name in sorted_bins[:6]:
            if bin_name in s["position_stats"]:
                mean_ms = s["position_stats"][bin_name]["mean_ms"]
                row += f" {mean_ms:>12.2f}"
            else:
                row += f" {'--':>12}"
        print(row)

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark generation speed across models"
    )
    parser.add_argument(
        "--seqcond-checkpoint",
        type=str,
        default="checkpoints/seqcond_torch_746k.pt",
        help="Path to SeqCond checkpoint",
    )
    parser.add_argument(
        "--hf-models",
        type=str,
        nargs="+",
        default=["baguettotron"],
        choices=list(MODEL_REGISTRY.keys()),
        help="HuggingFace models to benchmark",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate per prompt",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=3,
        help="Number of prompts to use",
    )
    parser.add_argument(
        "--use-triton",
        action="store_true",
        help="Use Triton kernels for SeqCond",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Disable CUDA graphs for SeqCond",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file for detailed results",
    )
    parser.add_argument(
        "--skip-seqcond",
        action="store_true",
        help="Skip SeqCond benchmark",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    prompts = LONG_PROMPTS[: args.num_prompts]
    print(f"Using {len(prompts)} prompts, max {args.max_tokens} tokens each")

    all_results = []
    all_stats = []

    # Benchmark SeqCond
    if not args.skip_seqcond:
        seqcond_results = benchmark_seqcond(
            args.seqcond_checkpoint,
            prompts,
            args.max_tokens,
            device=device,
            use_triton=args.use_triton,
            use_cuda_graph=(not args.no_cuda_graph and device == "cuda"),
        )
        all_results.append(seqcond_results)
        all_stats.append(compute_stats(seqcond_results))

    # Benchmark HF models
    for model_name in args.hf_models:
        try:
            hf_results = benchmark_hf_model(
                model_name,
                prompts,
                args.max_tokens,
                device=device,
            )
            all_results.append(hf_results)
            all_stats.append(compute_stats(hf_results))
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            continue

    # Print summary tables
    print_summary_table(all_stats)
    print_position_table(all_stats)

    # Save detailed results
    output_data = {
        "config": {
            "max_tokens": args.max_tokens,
            "num_prompts": len(prompts),
            "device": device,
        },
        "stats": all_stats,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()

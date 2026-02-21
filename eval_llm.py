#!/usr/bin/env python3
"""Benchmark external LLMs (HuggingFace models) on reasoning tasks.

This script evaluates external language models from HuggingFace on the same
benchmarks as eval_reasoning.py, allowing for direct comparison with SeqCond.
"""
import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Import evaluation functions from eval_reasoning
from eval_reasoning import (
    evaluate_winogrande,
    evaluate_openbookqa,
    evaluate_commonsenseqa,
    evaluate_hellaswag,
    evaluate_piqa,
    evaluate_arc,
    evaluate_gpqa,
    evaluate_gsm8k,
    _maybe_shuffle_dataset,
    _send_notification,
)


# Model registry mapping model names to HuggingFace checkpoint paths
MODEL_REGISTRY = {
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "lfm2-350m": "LiquidAI/LFM2-350M",
    "granite-350m": "ibm-granite/granite-4.0-h-350M",
    "baguettotron": "PleIAs/Baguettotron",
    "gemma3-270m": "google/gemma-3-270m",
}


class HFGenerator:
    """Wrapper around HuggingFace models to match TorchGenerator interface."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        max_batch_size: int = 16,
        use_chat_template: bool = True,
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.max_batch_size = max_batch_size
        self.use_chat_template = use_chat_template

        print(f"Loading model from {checkpoint}...")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        # Check if model supports chat templates
        self.has_chat_template = (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        )
        if self.use_chat_template and not self.has_chat_template:
            print(
                "Warning: Model does not have a chat template. Using direct prompts instead."
            )
            self.use_chat_template = False

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set left padding for decoder-only models
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        # Get model config for maxlen
        self.maxlen = getattr(self.model.config, "max_position_embeddings", 2048)

        print(f"Model loaded on {device}")
        print(f"Max position embeddings: {self.maxlen}")

        # Warmup
        print("Warming up...")
        self._warmup()
        print("Ready!")

    def _warmup(self):
        """Warmup generation to compile kernels."""
        test_prompt = "Hello, world!"
        if self.use_chat_template and self.has_chat_template:
            messages = [{"role": "user", "content": test_prompt}]
            test_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = self.tokenizer.encode(test_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model.generate(
                inputs,
                max_new_tokens=10,
                temperature=0.0,
                top_p=0.9,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        if self.device == "cuda":
            torch.cuda.synchronize()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        verbose: bool = False,
        use_cuda_graph: bool = False,
        use_triton: bool = False,
        use_synth_template: bool = False,
        **kwargs,
    ):
        """Generate tokens one at a time (streaming) for a single prompt.

        This is a generator function that yields decoded tokens as strings.
        Used for compatibility with eval_reasoning.py's _collect_generation.

        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter (ignored, uses 0.9)
            top_k: Top-k sampling parameter (ignored)
            verbose: Whether to print tokens (ignored)
            use_cuda_graph: Whether to use CUDA graphs (ignored)
            use_triton: Whether to use Triton kernels (ignored)
            use_synth_template: Whether to use synth template (ignored)
            **kwargs: Additional arguments (ignored)

        Yields:
            Decoded token strings
        """
        # Apply chat template if enabled and available
        if self.use_chat_template and self.has_chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.maxlen - max_new_tokens,
        )

        # Filter out token_type_ids if present
        inputs = {
            k: v.to(self.device) for k, v in inputs.items() if k != "token_type_ids"
        }

        # Generate all tokens at once (HF doesn't support true streaming easily)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]

        # Yield tokens one at a time
        for token_id in generated_ids:
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
            yield token_str

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        max_thinking_tokens: int = None,
        output_constraints: str = None,
    ) -> list[str]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature (0.0 = greedy)
            max_thinking_tokens: Ignored (for compatibility)
            output_constraints: Ignored (for compatibility)

        Returns:
            List of generated strings (without the prompt)
        """
        # Apply chat template if enabled and available
        if self.use_chat_template and self.has_chat_template:
            formatted_prompts = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                formatted_prompts.append(formatted)
        else:
            formatted_prompts = prompts

        # Tokenize with padding
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.maxlen - max_new_tokens,
        )

        # Filter out token_type_ids if present (not all models use it)
        inputs = {
            k: v.to(self.device) for k, v in inputs.items() if k != "token_type_ids"
        }

        input_length = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part (skip the prompt)
        generated_texts = []
        for output in outputs:
            generated = output[input_length:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts


def benchmark_generation_speed(
    model_name: str,
    checkpoint: str,
    device: str = "cuda",
    max_new_tokens: int = 1000,
    num_runs: int = 3,
):
    """Benchmark generation speed (tokens/second) for a model."""
    print(f"\n{'='*60}")
    print(f"Benchmarking generation speed: {model_name}")
    print(f"{'='*60}")

    gen = HFGenerator(checkpoint, device=device, use_chat_template=True)

    test_prompt = "Write a long poem about love."
    if gen.has_chat_template:
        messages = [{"role": "user", "content": test_prompt}]
        input_text = gen.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        input_text = test_prompt
    inputs = gen.tokenizer.encode(input_text, return_tensors="pt").to(device)

    results = []
    for run in range(num_runs):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start_time = time.time()
        with torch.no_grad():
            outputs = gen.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.9,
                do_sample=False,
                pad_token_id=gen.tokenizer.pad_token_id,
            )

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()

        input_length = inputs.shape[1]
        output_length = outputs.shape[1]
        new_tokens_generated = output_length - input_length
        generation_time = end_time - start_time
        tokens_per_second = new_tokens_generated / generation_time

        peak_memory_mb = 0
        if device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(device)
            peak_memory_mb = peak_memory / (1024 * 1024)

        results.append(
            {
                "tokens": new_tokens_generated,
                "time": generation_time,
                "tps": tokens_per_second,
                "memory_mb": peak_memory_mb,
            }
        )

        print(f"  Run {run + 1}/{num_runs}:")
        print(f"    Tokens generated: {new_tokens_generated}")
        print(f"    Time: {generation_time:.2f}s")
        print(f"    Tokens/sec: {tokens_per_second:.2f}")
        if device == "cuda":
            print(f"    Peak memory: {peak_memory_mb:.2f} MB")

    # Average results
    avg_tps = sum(r["tps"] for r in results) / len(results)
    avg_time = sum(r["time"] for r in results) / len(results)
    avg_memory = sum(r["memory_mb"] for r in results) / len(results)

    print(f"\n  Average over {num_runs} runs:")
    print(f"    Tokens/sec: {avg_tps:.2f}")
    print(f"    Time: {avg_time:.2f}s")
    if device == "cuda":
        print(f"    Peak memory: {avg_memory:.2f} MB")
    print(f"{'='*60}\n")

    return avg_tps, avg_memory


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate external LLMs on reasoning benchmarks"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()) + ["custom"],
        help="Model to evaluate (or 'custom' with --checkpoint)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="HuggingFace checkpoint path (required if --model=custom)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        help="Benchmark to run: winogrande, openbookqa, commonsenseqa, hellaswag, piqa, arc, arc:easy, arc:challenge, gpqa:diamond, speed, all",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation (default: 16)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate per benchmark",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Don't use chat template (for base models)",
    )
    parser.add_argument(
        "--verbose-examples",
        type=int,
        default=5,
        help="Number of examples to print debug info for",
    )
    parser.add_argument(
        "--answer-last",
        action="store_true",
        default=None,
        help="GSM8K: extract the last number in the answer instead of the first",
    )

    args = parser.parse_args()

    # Resolve checkpoint
    if args.model == "custom":
        if not args.checkpoint:
            parser.error("--checkpoint is required when --model=custom")
        checkpoint = args.checkpoint
        model_name = args.checkpoint.split("/")[-1]
    else:
        checkpoint = MODEL_REGISTRY[args.model]
        model_name = args.model

    # Speed benchmark
    if args.benchmark in ["speed", "all"]:
        benchmark_generation_speed(
            model_name=model_name,
            checkpoint=checkpoint,
            device=args.device,
            max_new_tokens=1000,
            num_runs=3,
        )
        if args.benchmark == "speed":
            return

    # Create generator
    gen = HFGenerator(
        checkpoint=checkpoint,
        device=args.device,
        max_batch_size=args.batch_size,
        use_chat_template=not args.no_chat_template,
    )

    results = {}

    # Run benchmarks
    benchmarks_to_run = []
    if args.benchmark == "all":
        benchmarks_to_run = [
            "winogrande",
            "openbookqa",
            "commonsenseqa",
            "hellaswag",
            "piqa",
            "arc:easy",
            "arc:challenge",
        ]
    else:
        benchmarks_to_run = [args.benchmark]

    for bench in benchmarks_to_run:
        print(f"\n{'='*60}")
        print(f"Running benchmark: {bench}")
        print(f"{'='*60}\n")

        if bench == "winogrande":
            dataset = load_dataset(
                "allenai/winogrande", "winogrande_xl", split="validation"
            )
            dataset = _maybe_shuffle_dataset(dataset, seed=42)
            acc = evaluate_winogrande(
                gen,
                dataset,
                max_samples=args.max_samples,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                verbose_examples=args.verbose_examples,
            )
            results["winogrande"] = acc

        elif bench == "openbookqa":
            dataset = load_dataset("openbookqa", "main", split="validation")
            dataset = _maybe_shuffle_dataset(dataset, seed=42)
            acc = evaluate_openbookqa(
                gen,
                dataset,
                max_samples=args.max_samples,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                verbose_examples=args.verbose_examples,
            )
            results["openbookqa"] = acc

        elif bench == "commonsenseqa":
            dataset = load_dataset("tau/commonsense_qa", split="validation")
            dataset = _maybe_shuffle_dataset(dataset, seed=42)
            acc = evaluate_commonsenseqa(
                gen,
                dataset,
                max_samples=args.max_samples,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                verbose_examples=args.verbose_examples,
            )
            results["commonsenseqa"] = acc

        elif bench == "hellaswag":
            dataset = load_dataset("hellaswag", split="validation")
            dataset = _maybe_shuffle_dataset(dataset, seed=42)
            acc = evaluate_hellaswag(
                gen,
                dataset,
                max_samples=args.max_samples,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                verbose_examples=args.verbose_examples,
            )
            results["hellaswag"] = acc

        elif bench == "piqa":
            dataset = load_dataset("ybisk/piqa", split="validation")
            dataset = _maybe_shuffle_dataset(dataset, seed=42)
            acc = evaluate_piqa(
                gen,
                dataset,
                max_samples=args.max_samples,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                verbose_examples=args.verbose_examples,
            )
            results["piqa"] = acc

        elif bench.startswith("arc"):
            if ":" in bench:
                subset = bench.split(":")[1]
            else:
                subset = "both"

            if subset in ["easy", "both"]:
                dataset_easy = load_dataset(
                    "allenai/ai2_arc", "ARC-Easy", split="validation"
                )
                dataset_easy = _maybe_shuffle_dataset(dataset_easy, seed=42)
                acc_easy = evaluate_arc(
                    gen,
                    dataset_easy,
                    max_samples=args.max_samples,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                    verbose_examples=args.verbose_examples,
                )
                results["arc:easy"] = acc_easy

            if subset in ["challenge", "both"]:
                dataset_challenge = load_dataset(
                    "allenai/ai2_arc", "ARC-Challenge", split="validation"
                )
                dataset_challenge = _maybe_shuffle_dataset(dataset_challenge, seed=42)
                acc_challenge = evaluate_arc(
                    gen,
                    dataset_challenge,
                    max_samples=args.max_samples,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                    verbose_examples=args.verbose_examples,
                )
                results["arc:challenge"] = acc_challenge

            if subset == "both":
                avg = (results["arc:easy"] + results["arc:challenge"]) / 2
                results["arc:average"] = avg

        elif bench == "gpqa:diamond":
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
            dataset = _maybe_shuffle_dataset(dataset, seed=0)
            acc = evaluate_gpqa(
                gen,
                dataset,
                max_samples=args.max_samples,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                verbose_examples=args.verbose_examples,
            )
            results["gpqa:diamond"] = acc

        elif bench == "gsm8k":
            dataset = load_dataset("openai/gsm8k", "main", split="test")
            dataset = _maybe_shuffle_dataset(dataset, seed=42)
            acc = evaluate_gsm8k(
                gen,
                dataset,
                max_samples=args.max_samples,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                verbose_examples=args.verbose_examples,
                answer_last=args.answer_last,
            )
            results["gsm8k"] = acc

    # Print summary
    if results:
        print(f"\n{'='*60}")
        print(f"SUMMARY - {model_name}")
        print(f"{'='*60}")
        for bench, acc in sorted(results.items()):
            print(f"  {bench:20s}: {acc:6.2f}%")
        print(f"{'='*60}\n")

        # Send notification
        summary = ", ".join(f"{k}={v:.1f}%" for k, v in sorted(results.items()))
        _send_notification(
            f"Eval Complete: {model_name}",
            summary,
        )


if __name__ == "__main__":
    main()

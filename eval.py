#!/usr/bin/env python3
"""Simple HellaSwag evaluation using optimized TorchGenerator."""
import time
import torch
from datasets import load_dataset

# from huggingface_hub import login
from seqcond.torch.generator import TorchGenerator


def evaluate_hellaswag(gen, dataset, max_samples=None):
    """Evaluate on HellaSwag using log probabilities."""
    correct = 0
    total = 0

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples...")
    start_time = time.time()

    for idx, example in enumerate(dataset):
        ctx = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])

        # Compute average log prob for each choice
        scores = []
        for ending in endings:
            choice_text = f"{ctx} {ending}"
            tokens = gen.tokenizer([choice_text])[0]

            if len(tokens) < 2:
                scores.append(float("-inf"))
                continue

            input_ids = torch.tensor([tokens], device=gen.device)
            with torch.no_grad():
                logits, _ = gen.model.prefill(input_ids, return_all_logits=True)

            # Compute sum of log probs for next token prediction
            log_probs = torch.log_softmax(logits[0], dim=-1)
            target_tokens = torch.tensor(tokens[1:], device=gen.device)
            seq_len = min(len(tokens) - 1, log_probs.shape[0])
            token_log_probs = log_probs[torch.arange(seq_len), target_tokens[:seq_len]]
            scores.append(token_log_probs.mean().item())

        # Best choice is highest score
        best_choice = max(range(4), key=lambda i: scores[i])
        if best_choice == label:
            correct += 1
        total += 1

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed
            acc = correct / total * 100
            print(
                f"  {idx + 1}/{len(dataset)} | Acc: {acc:.1f}% | Speed: {speed:.1f} samples/s"
            )

    elapsed = time.time() - start_time
    accuracy = correct / total * 100

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {total/elapsed:.1f} samples/s")
    print(f"{'='*60}")

    return accuracy


def evaluate_gpqa(gen, dataset, max_samples=None):
    """Evaluate on GPQA using log probabilities for multiple choice."""
    correct = 0
    total = 0

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples...")
    start_time = time.time()

    for idx, example in enumerate(dataset):
        question = example["Question"]
        choices = [
            example["Correct Answer"],
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]

        # Compute average log prob for each choice
        scores = []
        for choice in choices:
            choice_text = f"{question}\nAnswer: {choice}"
            tokens = gen.tokenizer([choice_text])[0]

            if len(tokens) < 2:
                scores.append(float("-inf"))
                continue

            input_ids = torch.tensor([tokens], device=gen.device)
            with torch.no_grad():
                logits, _ = gen.model.prefill(input_ids, return_all_logits=True)

            # Compute sum of log probs for next token prediction
            log_probs = torch.log_softmax(logits[0], dim=-1)
            target_tokens = torch.tensor(tokens[1:], device=gen.device)
            seq_len = min(len(tokens) - 1, log_probs.shape[0])
            token_log_probs = log_probs[torch.arange(seq_len), target_tokens[:seq_len]]
            scores.append(token_log_probs.mean().item())

        # Best choice is highest score, correct answer is always index 0
        best_choice = max(range(4), key=lambda i: scores[i])
        if best_choice == 0:
            correct += 1
        total += 1

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed
            acc = correct / total * 100
            print(
                f"  {idx + 1}/{len(dataset)} | Acc: {acc:.1f}% | Speed: {speed:.1f} samples/s"
            )

    elapsed = time.time() - start_time
    accuracy = correct / total * 100

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {total/elapsed:.1f} samples/s")
    print(f"{'='*60}")

    return accuracy


def evaluate_mmlu(gen, dataset, max_samples=None):
    """Evaluate on MMLU using log probabilities for multiple choice."""
    correct = 0
    total = 0

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples...")
    start_time = time.time()

    for idx, example in enumerate(dataset):
        question = example["question"]
        choices = example["choices"]
        label = example["answer"]  # Index of correct answer (0-3)

        # Compute average log prob for each choice
        scores = []
        for choice in choices:
            choice_text = f"{question}\nAnswer: {choice}"
            tokens = gen.tokenizer([choice_text])[0]

            if len(tokens) < 2:
                scores.append(float("-inf"))
                continue

            input_ids = torch.tensor([tokens], device=gen.device)
            with torch.no_grad():
                logits, _ = gen.model.prefill(input_ids, return_all_logits=True)

            # Compute sum of log probs for next token prediction
            log_probs = torch.log_softmax(logits[0], dim=-1)
            target_tokens = torch.tensor(tokens[1:], device=gen.device)
            seq_len = min(len(tokens) - 1, log_probs.shape[0])
            token_log_probs = log_probs[torch.arange(seq_len), target_tokens[:seq_len]]
            scores.append(token_log_probs.mean().item())

        # Best choice is highest score
        best_choice = max(range(len(choices)), key=lambda i: scores[i])
        if best_choice == label:
            correct += 1
        total += 1

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed
            acc = correct / total * 100
            print(
                f"  {idx + 1}/{len(dataset)} | Acc: {acc:.1f}% | Speed: {speed:.1f} samples/s"
            )

    elapsed = time.time() - start_time
    accuracy = correct / total * 100

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {total/elapsed:.1f} samples/s")
    print(f"{'='*60}")

    return accuracy


def evaluate_arc(gen, dataset, max_samples=None):
    """Evaluate on ARC using log probabilities for multiple choice."""
    correct = 0
    total = 0

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples...")
    start_time = time.time()

    for idx, example in enumerate(dataset):
        question = example["question"]
        choices = example["choices"]["text"]
        label = example["answerKey"]

        # Convert label (A, B, C, D, etc.) to index
        if label.isdigit():
            label_idx = int(label) - 1
        else:
            label_idx = ord(label.upper()) - ord("A")

        # Compute average log prob for each choice
        scores = []
        for choice in choices:
            choice_text = f"{question}\nAnswer: {choice}"
            tokens = gen.tokenizer([choice_text])[0]

            if len(tokens) < 2:
                scores.append(float("-inf"))
                continue

            input_ids = torch.tensor([tokens], device=gen.device)
            with torch.no_grad():
                logits, _ = gen.model.prefill(input_ids, return_all_logits=True)

            log_probs = torch.log_softmax(logits[0], dim=-1)
            target_tokens = torch.tensor(tokens[1:], device=gen.device)
            seq_len = min(len(tokens) - 1, log_probs.shape[0])
            token_log_probs = log_probs[torch.arange(seq_len), target_tokens[:seq_len]]
            scores.append(token_log_probs.mean().item())

        best_choice = max(range(len(choices)), key=lambda i: scores[i])
        if best_choice == label_idx:
            correct += 1
        total += 1

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed
            acc = correct / total * 100
            print(
                f"  {idx + 1}/{len(dataset)} | Acc: {acc:.1f}% | Speed: {speed:.1f} samples/s"
            )

    elapsed = time.time() - start_time
    accuracy = correct / total * 100

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {total/elapsed:.1f} samples/s")
    print(f"{'='*60}")

    return accuracy


def evaluate_piqa(gen, dataset, max_samples=None):
    """Evaluate on PIQA using log probabilities."""
    correct = 0
    total = 0

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples...")
    start_time = time.time()

    for idx, example in enumerate(dataset):
        goal = example["goal"]
        sol1 = example["sol1"]
        sol2 = example["sol2"]
        label = example["label"]

        # Compute log prob for each solution
        scores = []
        for solution in [sol1, sol2]:
            choice_text = f"{goal}\n{solution}"
            tokens = gen.tokenizer([choice_text])[0]

            if len(tokens) < 2:
                scores.append(float("-inf"))
                continue

            input_ids = torch.tensor([tokens], device=gen.device)
            with torch.no_grad():
                logits, _ = gen.model.prefill(input_ids, return_all_logits=True)

            log_probs = torch.log_softmax(logits[0], dim=-1)
            target_tokens = torch.tensor(tokens[1:], device=gen.device)
            seq_len = min(len(tokens) - 1, log_probs.shape[0])
            token_log_probs = log_probs[torch.arange(seq_len), target_tokens[:seq_len]]
            scores.append(token_log_probs.mean().item())

        best_choice = max(range(2), key=lambda i: scores[i])
        if best_choice == label:
            correct += 1
        total += 1

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed
            acc = correct / total * 100
            print(
                f"  {idx + 1}/{len(dataset)} | Acc: {acc:.1f}% | Speed: {speed:.1f} samples/s"
            )

    elapsed = time.time() - start_time
    accuracy = correct / total * 100

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {total/elapsed:.1f} samples/s")
    print(f"{'='*60}")

    return accuracy


def evaluate_winogrande(gen, dataset, max_samples=None):
    """Evaluate on Winogrande using log probabilities."""
    correct = 0
    total = 0

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples...")
    start_time = time.time()

    for idx, example in enumerate(dataset):
        sentence = example["sentence"]
        option1 = example["option1"]
        option2 = example["option2"]
        answer = example["answer"]

        # answer is "1" or "2"
        label = int(answer) - 1

        # Replace _ with each option
        scores = []
        for option in [option1, option2]:
            choice_text = sentence.replace("_", option)
            tokens = gen.tokenizer([choice_text])[0]

            if len(tokens) < 2:
                scores.append(float("-inf"))
                continue

            input_ids = torch.tensor([tokens], device=gen.device)
            with torch.no_grad():
                logits, _ = gen.model.prefill(input_ids, return_all_logits=True)

            log_probs = torch.log_softmax(logits[0], dim=-1)
            target_tokens = torch.tensor(tokens[1:], device=gen.device)
            seq_len = min(len(tokens) - 1, log_probs.shape[0])
            token_log_probs = log_probs[torch.arange(seq_len), target_tokens[:seq_len]]
            scores.append(token_log_probs.mean().item())

        best_choice = max(range(2), key=lambda i: scores[i])
        if best_choice == label:
            correct += 1
        total += 1

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed
            acc = correct / total * 100
            print(
                f"  {idx + 1}/{len(dataset)} | Acc: {acc:.1f}% | Speed: {speed:.1f} samples/s"
            )

    elapsed = time.time() - start_time
    accuracy = correct / total * 100

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {total/elapsed:.1f} samples/s")
    print(f"{'='*60}")

    return accuracy


def evaluate_gsm8k(gen, dataset, max_samples=None):
    """Evaluate on GSM8K using exact match on final answer."""
    import re

    correct = 0
    total = 0

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples...")
    start_time = time.time()

    for idx, example in enumerate(dataset):
        question = example["question"]
        answer = example["answer"]

        # Extract the numerical answer from the reference answer
        # GSM8K answers are in format "#### 42" at the end
        ref_match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", answer)
        if not ref_match:
            total += 1
            continue

        ref_answer = ref_match.group(1).replace(",", "")

        # Generate model's answer
        prompt = f"Question: {question}\nAnswer: Let's solve this step by step.\n"

        try:
            generated = gen.generate(
                prompt,
                max_new_tokens=256,
                temperature=0.0,  # Greedy decoding for math
                top_p=1.0,
                top_k=0,
                verbose=False,
                use_cuda_graph=False,
            )

            # Extract numerical answer from generated text
            # Look for patterns like "The answer is 42" or "#### 42"
            gen_match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", generated)
            if not gen_match:
                # Try other patterns
                gen_match = re.search(
                    r"(?:answer is|equals?|=)\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
                    generated,
                    re.IGNORECASE,
                )

            if gen_match:
                gen_answer = gen_match.group(1).replace(",", "")
                if gen_answer == ref_answer:
                    correct += 1
        except Exception as e:
            print(f"Error on example {idx}: {e}")

        total += 1

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed
            acc = correct / total * 100
            print(
                f"  {idx + 1}/{len(dataset)} | Acc: {acc:.1f}% | Speed: {speed:.1f} samples/s"
            )

    elapsed = time.time() - start_time
    accuracy = correct / total * 100

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {total/elapsed:.1f} samples/s")
    print(f"{'='*60}")

    return accuracy


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/seqcond_torch_80k.pt")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="hellaswag",
        # choices=["hellaswag"],
        help="Benchmark to evaluate on",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to evaluate (None for all)",
    )
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for gated datasets",
    )
    args = parser.parse_args()

    # # Login to HuggingFace if token provided or use cached credentials
    # if args.hf_token:
    #     login(token=args.hf_token)
    # else:
    #     # Try to use cached credentials
    #     try:
    #         login()
    #     except Exception:
    #         pass  # If no cached credentials, continue without login

    print("Loading model...")
    gen = TorchGenerator(args.checkpoint)

    if args.benchmark == "hellaswag":
        print(f"\nLoading HellaSwag dataset (split={args.split})...")
        dataset = load_dataset("hellaswag", split=args.split)
        print(f"Dataset loaded: {len(dataset)} examples\n")
        accuracy = evaluate_hellaswag(gen, dataset, max_samples=args.max_samples)
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
    elif args.benchmark == "gpqa:diamond":
        print(f"\nLoading GPQA Diamond dataset...")
        # Pass token explicitly to load_dataset
        token = os.environ.get("HF_TOKEN")
        dataset = load_dataset(
            "Idavidrein/gpqa", "gpqa_diamond", split="train", token=token
        )
        print(f"Dataset loaded: {len(dataset)} examples\n")
        accuracy = evaluate_gpqa(gen, dataset, max_samples=args.max_samples)
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
    elif args.benchmark.startswith("mmlu"):
        # Support mmlu:all or mmlu:subset_name
        parts = args.benchmark.split(":")
        if len(parts) == 2 and parts[1] != "all":
            subset = parts[1]
            print(f"\nLoading MMLU dataset (subset={subset}, split={args.split})...")
            dataset = load_dataset("cais/mmlu", subset, split=args.split)
            print(f"Dataset loaded: {len(dataset)} examples\n")
            accuracy = evaluate_mmlu(gen, dataset, max_samples=args.max_samples)
            print(f"\nFinal Accuracy: {accuracy:.2f}%")
        else:
            # Evaluate on all MMLU subsets
            print(f"\nLoading all MMLU subsets (split={args.split})...")
            from datasets import get_dataset_config_names

            subsets = get_dataset_config_names("cais/mmlu")

            all_correct = 0
            all_total = 0

            for subset in subsets:
                if subset == "all":
                    continue
                print(f"\n--- Evaluating {subset} ---")
                try:
                    dataset = load_dataset("cais/mmlu", subset, split=args.split)
                except ValueError as e:
                    # Skip subsets that don't have the requested split
                    print(f"Skipping {subset}: {e}")
                    continue
                subset_dataset = (
                    dataset
                    if not args.max_samples
                    else dataset.select(range(min(args.max_samples, len(dataset))))
                )

                correct = 0
                total = 0
                for example in subset_dataset:
                    question = example["question"]
                    choices = example["choices"]
                    label = example["answer"]

                    scores = []
                    for choice in choices:
                        choice_text = f"{question}\nAnswer: {choice}"
                        tokens = gen.tokenizer([choice_text])[0]
                        if len(tokens) < 2:
                            scores.append(float("-inf"))
                            continue
                        input_ids = torch.tensor([tokens], device=gen.device)
                        with torch.no_grad():
                            logits, _ = gen.model.prefill(
                                input_ids, return_all_logits=True
                            )
                        log_probs = torch.log_softmax(logits[0], dim=-1)
                        target_tokens = torch.tensor(tokens[1:], device=gen.device)
                        seq_len = min(len(tokens) - 1, log_probs.shape[0])
                        token_log_probs = log_probs[
                            torch.arange(seq_len), target_tokens[:seq_len]
                        ]
                        scores.append(token_log_probs.mean().item())

                    best_choice = max(range(len(choices)), key=lambda i: scores[i])
                    if best_choice == label:
                        correct += 1
                    total += 1

                subset_acc = correct / total * 100 if total > 0 else 0
                print(f"{subset}: {subset_acc:.2f}% ({correct}/{total})")
                all_correct += correct
                all_total += total

            overall_acc = all_correct / all_total * 100 if all_total > 0 else 0
            print(f"\n{'='*60}")
            print(
                f"Overall MMLU Accuracy: {overall_acc:.2f}% ({all_correct}/{all_total})"
            )
            print(f"{'='*60}")
    elif args.benchmark.startswith("arc"):
        # Support arc:easy, arc:challenge, or arc (both)
        parts = args.benchmark.split(":")
        if len(parts) == 2:
            subset = parts[1]  # "easy" or "challenge"
            print(f"\nLoading ARC-{subset} dataset (split={args.split})...")
            dataset = load_dataset(
                "allenai/ai2_arc", f"ARC-{subset.capitalize()}", split=args.split
            )
            print(f"Dataset loaded: {len(dataset)} examples\n")
            accuracy = evaluate_arc(gen, dataset, max_samples=args.max_samples)
            print(f"\nFinal Accuracy: {accuracy:.2f}%")
        else:
            # Evaluate both easy and challenge
            for subset in ["Easy", "Challenge"]:
                print(f"\n--- ARC-{subset} ---")
                dataset = load_dataset(
                    "allenai/ai2_arc", f"ARC-{subset}", split=args.split
                )
                print(f"Dataset loaded: {len(dataset)} examples\n")
                accuracy = evaluate_arc(gen, dataset, max_samples=args.max_samples)
                print(f"ARC-{subset} Accuracy: {accuracy:.2f}%")
    elif args.benchmark == "piqa":
        print(f"\nLoading PIQA dataset (split={args.split})...")
        dataset = load_dataset("ybisk/piqa", split=args.split)
        print(f"Dataset loaded: {len(dataset)} examples\n")
        accuracy = evaluate_piqa(gen, dataset, max_samples=args.max_samples)
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
    elif args.benchmark.startswith("winogrande"):
        # Support winogrande:xl, winogrande:l, etc.
        parts = args.benchmark.split(":")
        subset = parts[1] if len(parts) == 2 else "winogrande_xl"
        if not subset.startswith("winogrande_"):
            subset = f"winogrande_{subset}"
        print(f"\nLoading Winogrande dataset (subset={subset}, split={args.split})...")
        dataset = load_dataset("allenai/winogrande", subset, split=args.split)
        print(f"Dataset loaded: {len(dataset)} examples\n")
        accuracy = evaluate_winogrande(gen, dataset, max_samples=args.max_samples)
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
    elif args.benchmark == "gsm8k":
        print(f"\nLoading GSM8K dataset (split={args.split})...")
        dataset = load_dataset("openai/gsm8k", "main", split=args.split)
        print(f"Dataset loaded: {len(dataset)} examples\n")
        accuracy = evaluate_gsm8k(gen, dataset, max_samples=args.max_samples)
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
    else:
        print(f"Benchmark {args.benchmark} not implemented yet")


if __name__ == "__main__":
    main()

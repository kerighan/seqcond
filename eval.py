#!/usr/bin/env python3
"""Simple HellaSwag evaluation using optimized TorchGenerator."""
import time
import torch
from datasets import load_dataset
import os
import random
import subprocess

# from huggingface_hub import login
from seqcond.torch.generator import TorchGenerator


def _send_notification(title: str, message: str):
    """Send a Linux desktop notification."""
    try:
        subprocess.run(
            ["notify-send", title, message],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


def _common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _score_continuation_from_tokens(gen, full_tokens, prompt_prefix_len):
    if len(full_tokens) < 2:
        return float("-inf")

    input_ids = torch.tensor([full_tokens], device=gen.device)
    with torch.no_grad():
        logits, _ = gen.model.prefill(input_ids, return_all_logits=True)

    log_probs = torch.log_softmax(logits[0], dim=-1)
    target_tokens = input_ids[0, 1:]

    seq_len = min(target_tokens.shape[0], log_probs.shape[0])
    start_idx = max(int(prompt_prefix_len) - 1, 0)
    if start_idx >= seq_len:
        return float("-inf")

    relevant_log_probs = log_probs[start_idx:seq_len]
    relevant_targets = target_tokens[start_idx:seq_len]
    if relevant_targets.numel() == 0:
        return float("-inf")

    token_log_probs = relevant_log_probs[
        torch.arange(relevant_targets.shape[0], device=gen.device), relevant_targets
    ]
    return token_log_probs.mean().item()


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

        prompt_text = f"{ctx} "
        prompt_tokens = gen.tokenizer([prompt_text])[0]

        # Compute average log prob for each choice
        scores = []
        for ending in endings:
            full_text = prompt_text + ending
            full_tokens = gen.tokenizer([full_text])[0]
            prompt_prefix_len = _common_prefix_len(prompt_tokens, full_tokens)
            scores.append(
                _score_continuation_from_tokens(gen, full_tokens, prompt_prefix_len)
            )

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


def evaluate_openbookqa(gen, dataset, max_samples=None):
    """Evaluate on OpenBookQA using log probabilities for multiple choice."""
    correct = 0
    total = 0

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples...")
    start_time = time.time()

    for idx, example in enumerate(dataset):
        # OpenBookQA schema is typically:
        # question_stem: str
        # choices: {text: [...], label: [...]}  (labels are A/B/C/D)
        # answerKey: 'A'/'B'/'C'/'D'
        question = example.get("question_stem")
        if question is None and isinstance(example.get("question"), dict):
            question = example["question"].get("stem")
        if question is None:
            question = ""

        choices_obj = example.get("choices")
        if isinstance(choices_obj, dict):
            choices = choices_obj.get("text") or []
            labels = choices_obj.get("label") or []
        else:
            choices = []
            labels = []

        answer_key = example.get("answerKey")
        if answer_key is None:
            answer_key = example.get("answer")

        if answer_key is None:
            label_idx = None
        else:
            answer_key = str(answer_key)
            if answer_key.isdigit():
                label_idx = int(answer_key) - 1
            else:
                answer_key = answer_key.strip().upper()
                if labels and answer_key in labels:
                    label_idx = labels.index(answer_key)
                else:
                    label_idx = ord(answer_key) - ord("A")

        prompt_text = f"Question: {question}\nAnswer: "
        prompt_tokens = gen.tokenizer([prompt_text])[0]

        scores = []
        for choice in choices:
            full_text = prompt_text + choice
            full_tokens = gen.tokenizer([full_text])[0]
            prompt_prefix_len = _common_prefix_len(prompt_tokens, full_tokens)
            scores.append(
                _score_continuation_from_tokens(gen, full_tokens, prompt_prefix_len)
            )

        if scores and label_idx is not None and 0 <= label_idx < len(scores):
            best_choice = max(range(len(scores)), key=lambda i: scores[i])
            if best_choice == label_idx:
                correct += 1
        total += 1

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed
            acc = (correct / total * 100) if total > 0 else 0.0
            print(
                f"  {idx + 1}/{len(dataset)} | Acc: {acc:.1f}% | Speed: {speed:.1f} samples/s"
            )

    elapsed = time.time() - start_time
    accuracy = (correct / total * 100) if total > 0 else 0.0

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

    rng = random.Random(0)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples...")
    start_time = time.time()

    for idx, example in enumerate(dataset):
        question = example["Question"]
        choices_with_label = [
            (example["Correct Answer"], True),
            (example["Incorrect Answer 1"], False),
            (example["Incorrect Answer 2"], False),
            (example["Incorrect Answer 3"], False),
        ]
        rng.shuffle(choices_with_label)
        choices = [c for c, _ in choices_with_label]
        label_idx = next(
            i for i, (_, is_correct) in enumerate(choices_with_label) if is_correct
        )

        prompt_text = f"{question}\nAnswer: "
        prompt_tokens = gen.tokenizer([prompt_text])[0]

        # Compute average log prob for each choice
        scores = []
        for choice in choices:
            full_text = prompt_text + choice
            full_tokens = gen.tokenizer([full_text])[0]
            prompt_prefix_len = _common_prefix_len(prompt_tokens, full_tokens)
            scores.append(
                _score_continuation_from_tokens(gen, full_tokens, prompt_prefix_len)
            )

        # Best choice is highest score
        best_choice = max(range(4), key=lambda i: scores[i])
        if best_choice == label_idx:
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

        prompt_text = f"{question}\nAnswer: "
        prompt_tokens = gen.tokenizer([prompt_text])[0]

        # Compute average log prob for each choice
        scores = []
        for choice in choices:
            full_text = prompt_text + choice
            full_tokens = gen.tokenizer([full_text])[0]
            prompt_prefix_len = _common_prefix_len(prompt_tokens, full_tokens)
            scores.append(
                _score_continuation_from_tokens(gen, full_tokens, prompt_prefix_len)
            )

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

        prompt_text = f"{question}\nAnswer: "
        prompt_tokens = gen.tokenizer([prompt_text])[0]

        # Convert label (A, B, C, D, etc.) to index
        if label.isdigit():
            label_idx = int(label) - 1
        else:
            label_idx = ord(label.upper()) - ord("A")

        # Compute average log prob for each choice
        scores = []
        for choice in choices:
            full_text = prompt_text + choice
            full_tokens = gen.tokenizer([full_text])[0]
            prompt_prefix_len = _common_prefix_len(prompt_tokens, full_tokens)
            scores.append(
                _score_continuation_from_tokens(gen, full_tokens, prompt_prefix_len)
            )

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

        prompt_text = f"{goal}\n"
        prompt_tokens = gen.tokenizer([prompt_text])[0]

        # Compute log prob for each solution
        scores = []
        for solution in [sol1, sol2]:
            full_text = prompt_text + solution
            full_tokens = gen.tokenizer([full_text])[0]
            prompt_prefix_len = _common_prefix_len(prompt_tokens, full_tokens)
            scores.append(
                _score_continuation_from_tokens(gen, full_tokens, prompt_prefix_len)
            )

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

        prefix, sep, suffix = sentence.partition("_")
        prompt_text = prefix
        prompt_tokens = gen.tokenizer([prompt_text])[0]

        # answer is "1" or "2"
        label = int(answer) - 1

        # Replace _ with each option
        scores = []
        for option in [option1, option2]:
            full_text = prompt_text + option + suffix
            full_tokens = gen.tokenizer([full_text])[0]
            prompt_prefix_len = _common_prefix_len(prompt_tokens, full_tokens)
            scores.append(
                _score_continuation_from_tokens(gen, full_tokens, prompt_prefix_len)
            )

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


def evaluate_commonsenseqa(gen, dataset, max_samples=None):
    """Evaluate on CommonsenseQA using log probabilities."""
    correct = 0
    total = 0

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples...")
    start_time = time.time()

    for idx, example in enumerate(dataset):
        question = example["question"]
        choices = example["choices"]["text"]
        labels = example["choices"]["label"]
        answer_key = example["answerKey"]

        prompt_text = f"Question: {question}\nAnswer: "
        prompt_tokens = gen.tokenizer([prompt_text])[0]

        if (idx + 1) % 50 == 0:
            print(f"Progress: {idx + 1}/{len(dataset)}")

        # Compute log probability for each choice
        scores = []
        for choice in choices:
            full_text = prompt_text + choice
            full_tokens = gen.tokenizer([full_text])[0]
            prompt_prefix_len = _common_prefix_len(prompt_tokens, full_tokens)
            scores.append(
                _score_continuation_from_tokens(gen, full_tokens, prompt_prefix_len)
            )

        best_choice_idx = max(range(len(choices)), key=lambda i: scores[i])
        predicted_label = labels[best_choice_idx]

        if predicted_label == answer_key:
            correct += 1
        total += 1

    elapsed = time.time() - start_time
    accuracy = correct / total * 100

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {total/elapsed:.1f} samples/s")
    print(f"{'='*60}")

    return accuracy


def evaluate_triviaqa(gen, dataset, max_samples=None):
    """Evaluate on TriviaQA using exact match on generated answer."""
    import re

    correct = 0
    total = 0

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples...")
    start_time = time.time()

    for idx, example in enumerate(dataset):
        question = example["question"]
        answer_dict = example["answer"]

        # Get all valid answer aliases
        valid_answers = [answer_dict["value"]]
        if "aliases" in answer_dict and answer_dict["aliases"]:
            valid_answers.extend(answer_dict["aliases"])

        # Normalize answers
        valid_answers = [ans.lower().strip() for ans in valid_answers]

        if (idx + 1) % 50 == 0:
            print(f"Progress: {idx + 1}/{len(dataset)}")

        # Generate answer with better prompt
        prompt = f"Q: {question}\nA:"
        generated = gen.generate(
            prompt,
            max_new_tokens=20,
            temperature=0.0,
            verbose=False,
        )

        # Extract just the answer part (after "A:")
        if "A:" in generated:
            pred_answer = generated.split("A:")[-1].strip()
        else:
            pred_answer = generated.strip()

        # Take first line and remove trailing punctuation
        pred_answer = pred_answer.split("\n")[0].strip()
        # Remove common question starters if model continues generating
        for stop_word in ["Q:", "Question:", "\n"]:
            if stop_word in pred_answer:
                pred_answer = pred_answer.split(stop_word)[0].strip()
        pred_answer = pred_answer.rstrip(".,!?").lower()

        # Check if any valid answer is in the prediction
        is_correct = any(
            ans in pred_answer or pred_answer in ans for ans in valid_answers
        )

        if is_correct:
            correct += 1
        total += 1

    elapsed = time.time() - start_time
    accuracy = correct / total * 100

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {total/elapsed:.1f} samples/s")
    print(f"{'='*60}")

    return accuracy


def evaluate_hotpotqa(gen, dataset, max_samples=None):
    """Evaluate on HotpotQA using exact match on generated answer."""
    correct = 0
    total = 0

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples...")
    start_time = time.time()

    for idx, example in enumerate(dataset):
        question = example["question"]
        answer = example["answer"].lower().strip()

        if (idx + 1) % 50 == 0:
            print(f"Progress: {idx + 1}/{len(dataset)}")

        # Generate answer with better prompt
        prompt = f"Q: {question}\nA:"
        generated = gen.generate(
            prompt,
            max_new_tokens=20,
            temperature=0.0,
            verbose=False,
        )

        # Extract just the answer part
        if "A:" in generated:
            pred_answer = generated.split("A:")[-1].strip()
        else:
            pred_answer = generated.strip()

        # Take first line and remove trailing punctuation
        pred_answer = pred_answer.split("\n")[0].strip()
        # Remove common question starters if model continues generating
        for stop_word in ["Q:", "Question:", "\n"]:
            if stop_word in pred_answer:
                pred_answer = pred_answer.split(stop_word)[0].strip()
        pred_answer = pred_answer.rstrip(".,!?").lower()

        # Check exact match or substring match
        is_correct = (
            answer in pred_answer or pred_answer in answer or answer == pred_answer
        )

        if is_correct:
            correct += 1
        total += 1

    elapsed = time.time() - start_time
    accuracy = correct / total * 100

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {total/elapsed:.1f} samples/s")
    print(f"{'='*60}")

    return accuracy


# 40k = 38.36%
# 60k = 39.12%
# 80k = 39.74%
# 150k = 41.27%
# 170k = 41.87%
# 180k = 42.25%
# 190k = 41.87%


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/seqcond_torch_330k.pt")
    # parser.add_argument("--checkpoint", default="checkpoints/seqcond_opt_5_torch.pt")
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
    parser.add_argument(
        "--use_triton",
        action="store_true",
        help="Use Triton kernels for faster inference",
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

    # Pre-capture CUDA graphs with Triton if requested
    if args.use_triton:
        gen.precompute(max_seq_len=1024, use_triton=True)

    if args.benchmark == "hellaswag":
        print(f"\nLoading HellaSwag dataset (split={args.split})...")
        dataset = load_dataset("hellaswag", split=args.split)
        print(f"Dataset loaded: {len(dataset)} examples\n")
        accuracy = evaluate_hellaswag(gen, dataset, max_samples=args.max_samples)
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
        _send_notification("Évaluation terminée", f"HellaSwag: {accuracy:.2f}%")
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
        _send_notification("Évaluation terminée", f"GPQA Diamond: {accuracy:.2f}%")
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
            _send_notification("Évaluation terminée", f"MMLU {subset}: {accuracy:.2f}%")
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

                    prompt_text = f"{question}\nAnswer: "
                    prompt_tokens = gen.tokenizer([prompt_text])[0]

                    scores = []
                    for choice in choices:
                        full_text = prompt_text + choice
                        full_tokens = gen.tokenizer([full_text])[0]
                        prompt_prefix_len = _common_prefix_len(
                            prompt_tokens, full_tokens
                        )
                        scores.append(
                            _score_continuation_from_tokens(
                                gen, full_tokens, prompt_prefix_len
                            )
                        )

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
            _send_notification("Évaluation terminée", f"MMLU (all): {overall_acc:.2f}%")
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
            _send_notification("Évaluation terminée", f"ARC-{subset}: {accuracy:.2f}%")
        else:
            # Evaluate both easy and challenge
            accuracies = {}
            for subset in ["Easy", "Challenge"]:
                print(f"\n--- ARC-{subset} ---")
                dataset = load_dataset(
                    "allenai/ai2_arc", f"ARC-{subset}", split=args.split
                )
                print(f"Dataset loaded: {len(dataset)} examples\n")
                accuracy = evaluate_arc(gen, dataset, max_samples=args.max_samples)
                accuracies[subset] = accuracy
                print(f"ARC-{subset} Accuracy: {accuracy:.2f}%")

            # Calculate and display average
            avg_accuracy = sum(accuracies.values()) / len(accuracies)
            print(f"\n{'='*60}")
            print(f"ARC (average): {avg_accuracy:.2f}%")
            print(f"{'='*60}")
            _send_notification("Évaluation terminée", f"ARC (avg): {avg_accuracy:.2f}%")
    elif args.benchmark == "piqa":
        print(f"\nLoading PIQA dataset (split={args.split})...")
        dataset = load_dataset(
            "ybisk/piqa", split=args.split, revision="refs/convert/parquet"
        )
        print(f"Dataset loaded: {len(dataset)} examples\n")
        accuracy = evaluate_piqa(gen, dataset, max_samples=args.max_samples)
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
        _send_notification("Évaluation terminée", f"PIQA: {accuracy:.2f}%")
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
        _send_notification("Évaluation terminée", f"Winogrande: {accuracy:.2f}%")
    elif args.benchmark == "gsm8k":
        print(f"\nLoading GSM8K dataset (split={args.split})...")
        dataset = load_dataset("openai/gsm8k", "main", split=args.split)
        print(f"Dataset loaded: {len(dataset)} examples\n")
        accuracy = evaluate_gsm8k(gen, dataset, max_samples=args.max_samples)
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
        _send_notification("Évaluation terminée", f"GSM8K: {accuracy:.2f}%")
    elif args.benchmark == "commonsenseqa":
        print(f"\nLoading CommonsenseQA dataset (split={args.split})...")
        dataset = load_dataset("commonsense_qa", split=args.split)
        print(f"Dataset loaded: {len(dataset)} examples\n")
        accuracy = evaluate_commonsenseqa(gen, dataset, max_samples=args.max_samples)
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
        _send_notification("Évaluation terminée", f"CommonsenseQA: {accuracy:.2f}%")
    elif args.benchmark == "openbookqa":
        print(f"\nLoading OpenBookQA dataset (split={args.split})...")
        dataset = load_dataset("openbookqa", "main", split=args.split)
        print(f"Dataset loaded: {len(dataset)} examples\n")
        accuracy = evaluate_openbookqa(gen, dataset, max_samples=args.max_samples)
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
        _send_notification("Évaluation terminée", f"OpenBookQA: {accuracy:.2f}%")
    elif args.benchmark == "triviaqa":
        print(f"\nLoading TriviaQA dataset (split={args.split})...")
        dataset = load_dataset("trivia_qa", "unfiltered.nocontext", split=args.split)
        print(f"Dataset loaded: {len(dataset)} examples\n")
        accuracy = evaluate_triviaqa(gen, dataset, max_samples=args.max_samples)
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
        _send_notification("Évaluation terminée", f"TriviaQA: {accuracy:.2f}%")
    elif args.benchmark == "hotpotqa":
        print(f"\nLoading HotpotQA dataset (split={args.split})...")
        dataset = load_dataset("hotpot_qa", "distractor", split=args.split)
        print(f"Dataset loaded: {len(dataset)} examples\n")
        accuracy = evaluate_hotpotqa(gen, dataset, max_samples=args.max_samples)
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
        _send_notification("Évaluation terminée", f"HotpotQA: {accuracy:.2f}%")
    else:
        print(f"Benchmark {args.benchmark} not implemented yet")


if __name__ == "__main__":
    main()

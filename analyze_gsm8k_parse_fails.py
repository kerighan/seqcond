#!/usr/bin/env python3
"""Analyze GSM8K parse failures to ensure we're not losing points unnecessarily.

This script runs GSM8K evaluation and logs all parse failures with detailed information
about what went wrong in the extraction process.
"""
import argparse
import re
import json
from datasets import load_dataset
from generate import TorchGenerator


def _extract_answer_after_thinking(text):
    """Extract answer after </reasoning> tag if present, otherwise return full text."""
    if "</reasoning>" in text:
        parts = text.split("</reasoning>", 1)
        if len(parts) > 1:
            return parts[1]
    return text


def _extract_number(text, answer_last=None):
    """Extract a number from text. Tries structured patterns first, then falls back."""
    if "answer is " in text:
        res = text.split("answer is ")[1].split(".")[0].strip()
        res = re.sub(r"[^0-9.-]", "", res)
        return res
    if "\\boxed{" in text:
        res = text.split("\\boxed{")[1].split("}")[0].strip()
        res = re.sub(r"[^0-9.-]", "", res)
        return res

    _NUM = r"-?\d+(?:,\d{3})*(?:\.\d+)?"

    # Try #### pattern first (GSM8K format)
    m = re.search(rf"####\s*({_NUM})", text)
    if m:
        return m.group(1).replace(",", "")
    # Try "Answer: X" pattern (with optional markdown bold **)
    m = re.search(rf"\*?\*?[Aa]nswer:?\*?\*?\s*\$?\s*({_NUM})", text)
    if m:
        return m.group(1).replace(",", "")
    # Try "answer is X" / "equals X" / "= X" pattern
    m = re.search(
        rf"(?:answer is|equals?|=)\s*\$?\s*({_NUM})",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).replace(",", "")
    # Fallback: pick first or last number depending on answer_last
    numbers = re.findall(rf"{_NUM}", text)
    if numbers:
        idx = -1 if answer_last else 0
        return numbers[idx].replace(",", "")
    return None


def analyze_gsm8k_parse_failures(
    checkpoint,
    max_samples=None,
    max_new_tokens=512,
    batch_size=16,
    answer_last=None,
    output_file="gsm8k_parse_fails.jsonl",
):
    """Run GSM8K evaluation and log all parse failures."""
    print(f"Loading model from {checkpoint}...")
    gen = TorchGenerator(checkpoint)
    
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    n = len(dataset)
    print(f"\nAnalyzing {n} samples for parse failures...")
    print(f"Batch size: {batch_size}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Answer extraction: {'last number' if answer_last else 'first number (default)'}\n")
    
    correct = 0
    total = 0
    parse_failures = []
    
    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch = dataset.select(range(batch_start, batch_end))
        
        prompts = []
        ref_answers = []
        questions = []
        
        for example in batch:
            question = example["question"]
            answer = example["answer"]
            questions.append(question)
            
            # Extract reference numerical answer (format: "#### 42")
            ref_match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", answer)
            ref_val = ref_match.group(1).replace(",", "") if ref_match else None
            ref_answers.append(ref_val)
            
            prompt = (
                f"Solve the following math problem. Give your final numerical answer after your reasoning.\n\n{question}"
            )
            # Truncate if too long
            model_maxlen = getattr(gen.model, "maxlen", 2048)
            max_prompt_tokens = model_maxlen // 2
            prompt_tokens = gen.tokenizer.encode(prompt)
            if len(prompt_tokens) > max_prompt_tokens:
                prompt = gen.tokenizer.decode(prompt_tokens[:max_prompt_tokens])
            prompts.append(prompt)
        
        # Batched generation
        if batch_size == 1:
            outputs = []
            for prompt in prompts:
                output_tokens = []
                for token in gen.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                ):
                    output_tokens.append(token)
                outputs.append("".join(output_tokens))
        else:
            outputs = gen.generate_batch(
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
        
        for i, (output, ref_val, question) in enumerate(zip(outputs, ref_answers, questions)):
            idx = batch_start + i
            answer_text = _extract_answer_after_thinking(output)
            predicted = _extract_number(answer_text, answer_last=answer_last)
            
            is_correct = False
            parse_failed = False
            
            if ref_val is None:
                parse_failed = True
                failure_reason = "Reference answer could not be parsed from dataset"
            elif predicted is None:
                parse_failed = True
                failure_reason = "Could not extract number from model output"
            else:
                try:
                    if float(predicted) == float(ref_val):
                        is_correct = True
                        correct += 1
                except ValueError:
                    parse_failed = True
                    failure_reason = f"ValueError when comparing predicted='{predicted}' with ref='{ref_val}'"
            
            if parse_failed:
                parse_failures.append({
                    "idx": idx,
                    "question": question,
                    "reference_answer": ref_val,
                    "predicted_answer": predicted,
                    "full_output": output,
                    "answer_text": answer_text,
                    "failure_reason": failure_reason,
                })
            
            total += 1
        
        acc = correct / total * 100 if total > 0 else 0
        print(f"  {total}/{n} | Acc: {acc:.1f}% | Parse fails: {len(parse_failures)} ({len(parse_failures)/total*100:.1f}%)")
    
    # Save parse failures to file
    if parse_failures:
        print(f"\n{'='*60}")
        print(f"PARSE FAILURE ANALYSIS")
        print(f"{'='*60}")
        print(f"Total samples: {total}")
        print(f"Correct: {correct} ({correct/total*100:.1f}%)")
        print(f"Parse failures: {len(parse_failures)} ({len(parse_failures)/total*100:.1f}%)")
        print(f"\nSaving parse failures to: {output_file}")
        
        with open(output_file, "w") as f:
            for failure in parse_failures:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")
        
        print(f"\n{'='*60}")
        print("SAMPLE PARSE FAILURES (first 5):")
        print(f"{'='*60}")
        for i, failure in enumerate(parse_failures[:5]):
            print(f"\n[{i+1}] Index: {failure['idx']}")
            print(f"Question: {failure['question'][:100]}...")
            print(f"Reference: {failure['reference_answer']}")
            print(f"Predicted: {failure['predicted_answer']}")
            print(f"Reason: {failure['failure_reason']}")
            print(f"Answer text (first 200 chars): {failure['answer_text'][:200]!r}")
            print("-" * 60)
    else:
        print(f"\n{'='*60}")
        print("NO PARSE FAILURES! All answers were successfully extracted.")
        print(f"{'='*60}")
    
    accuracy = correct / total * 100 if total > 0 else 0.0
    print(f"\nFinal Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}\n")
    
    return accuracy, parse_failures


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GSM8K parse failures to identify extraction issues"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/seqcond_torch_100k.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation (default: 16)",
    )
    parser.add_argument(
        "--answer-last",
        action="store_true",
        help="Extract the last number instead of the first as fallback",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gsm8k_parse_fails.jsonl",
        help="Output file for parse failures (default: gsm8k_parse_fails.jsonl)",
    )
    
    args = parser.parse_args()
    
    analyze_gsm8k_parse_failures(
        checkpoint=args.checkpoint,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        answer_last=args.answer_last,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()

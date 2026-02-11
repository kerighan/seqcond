#!/usr/bin/env python3
"""Rigorous A/B test: compare old vs new _parse_choice on real model outputs."""
import re
import random
from datasets import load_dataset
from seqcond.torch.generator import TorchGenerator
from eval_reasoning import _parse_choice as _parse_choice_CURRENT


def _extract_answer_after_thinking(text):
    if "<|think_end|>" in text:
        text = text.split("<|think_end|>")[-1]
    elif "<|think_start|>" in text:
        before = text.split("<|think_start|>")[0].strip()
        if before:
            text = before
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    return text.strip()


def _parse_choice_OLD(answer_text, valid_choices, options=None):
    """ORIGINAL parser (before any of my changes)."""
    letters = "".join(sorted(valid_choices))
    answer_text = answer_text.strip()

    if answer_text.upper() in valid_choices:
        return answer_text.upper()

    letter_class = f"[{letters}]"
    for pattern in [
        rf"(?:answer|option|choice)\s*(?:is\s*)?({letter_class})",
        rf"(?:correct\s+(?:answer|option|choice)\s*(?:is\s*)?)({letter_class})",
        rf"\b([{letters}])\b",
        rf"\b([{letters}])\.",
    ]:
        m = re.search(pattern, answer_text, re.IGNORECASE)
        if m and m.group(1).upper() in valid_choices:
            return m.group(1).upper()

    if options:
        lower = answer_text.lower()
        letter_labels = sorted(valid_choices)
        matches = [
            lbl for lbl, opt in zip(letter_labels, options) if opt.lower() in lower
        ]
        if len(matches) == 1:
            return matches[0]

    return None


def _parse_choice_NEW(answer_text, valid_choices, options=None):
    """CURRENT parser imported from eval_reasoning.py."""
    return _parse_choice_CURRENT(answer_text, valid_choices, options=options)


def main():
    gen = TorchGenerator("checkpoints/seqcond_torch_120k.pt")
    dataset = load_dataset("commonsense_qa", split="validation")
    dataset = dataset.shuffle(seed=42)

    N = 20
    rng = random.Random(42)

    old_correct = 0
    new_correct = 0
    old_parse_fail = 0
    new_parse_fail = 0
    old_wrong = 0
    new_wrong = 0
    disagree = 0

    prompts = []
    metadata = []

    for i in range(N):
        ex = dataset[i]
        question = ex["question"]
        choices = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        answer_key = ex["answerKey"]

        indexed = list(enumerate(choices))
        rng.shuffle(indexed)
        shuffled_choices = [c for _, c in indexed]
        orig_correct_idx = (
            labels.index(answer_key)
            if answer_key in labels
            else (ord(answer_key) - ord("A"))
        )
        new_correct_idx = next(
            j for j, (orig_i, _) in enumerate(indexed) if orig_i == orig_correct_idx
        )
        correct_letter = chr(ord("A") + new_correct_idx)
        num_choices = len(shuffled_choices)

        prompt = (
            f"Question: {question}\n\n"
            + "\n".join(
                f"{chr(ord('A') + j)}. {c}" for j, c in enumerate(shuffled_choices)
            )
            + "\n\nAnswer with a single letter: "
            + ", ".join(chr(ord("A") + j) for j in range(num_choices))
            + "."
        )
        prompts.append(prompt)
        metadata.append((shuffled_choices, correct_letter, question))

    # Generate in batch
    outputs = gen.generate_batch(prompts, max_new_tokens=512, temperature=0.0)

    for i, (output, (choices, correct_letter, question)) in enumerate(
        zip(outputs, metadata)
    ):
        answer_text = _extract_answer_after_thinking(output)
        valid = {chr(ord("A") + j) for j in range(len(choices))}

        pred_old = _parse_choice_OLD(answer_text, valid, options=choices)
        pred_new = _parse_choice_NEW(answer_text, valid, options=choices)

        if pred_old != pred_new:
            disagree += 1

        if pred_old is None:
            old_parse_fail += 1
        elif pred_old == correct_letter:
            old_correct += 1
        else:
            old_wrong += 1

        if pred_new is None:
            new_parse_fail += 1
        elif pred_new == correct_letter:
            new_correct += 1
        else:
            new_wrong += 1

        # Print details when parsers disagree OR parse fails
        if pred_old != pred_new or pred_old is None or pred_new is None:
            print(f"\n{'='*60}")
            print(f"[{i}] Q: {question[:80]}...")
            print(f"  Correct: {correct_letter}")
            print(f"  OLD predicted: {pred_old}")
            print(f"  NEW predicted: {pred_new}")
            print(f"  answer_text (first 200): {answer_text[:200]}")
            print(f"  answer_text (last 200):  {answer_text[-200:]}")

    print(f"\n{'='*60}")
    print(f"SUMMARY ({N} samples)")
    print(f"{'='*60}")
    print(
        f"  OLD parser: correct={old_correct} wrong={old_wrong} parse_fail={old_parse_fail} acc={old_correct/N*100:.1f}%"
    )
    print(
        f"  NEW parser: correct={new_correct} wrong={new_wrong} parse_fail={new_parse_fail} acc={new_correct/N*100:.1f}%"
    )
    print(f"  Disagreements: {disagree}/{N}")


if __name__ == "__main__":
    main()

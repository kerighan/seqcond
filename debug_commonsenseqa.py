#!/usr/bin/env python3
import argparse
import random

from datasets import load_dataset

from seqcond.torch.generator import TorchGenerator
from eval_reasoning import _extract_answer_after_thinking, _parse_choice


def _wrap_synth(user_prompt: str) -> str:
    return (
        "<|im_start|>user\n"
        + user_prompt
        + "\n<|im_end|><|im_start|>assistant\n<|think_start|>"
    )


def _collect_generation(
    gen: TorchGenerator, prompt: str, max_new_tokens: int, temperature: float
) -> str:
    toks = []
    print(prompt)
    for tok in gen.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=1.0,
        top_k=0,
        verbose=False,
        use_cuda_graph=True,
        use_synth_template=True,
        use_triton=True,
    ):
        toks.append(tok)
    print(len(toks))
    return "".join(toks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/seqcond_torch_190k.pt")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--pause", action="store_true")
    parser.add_argument("--show_wrapped_prompt", action="store_true")
    args = parser.parse_args()

    print("Loading model...")
    gen = TorchGenerator(args.checkpoint)

    print(f"Loading CommonsenseQA dataset (split={args.split})...")
    # dataset = load_dataset("commonsense_qa", split=args.split)
    dataset = load_dataset(
        "ybisk/piqa", split=args.split, revision="refs/convert/parquet"
    )
    # dataset = load_dataset("commonsense_qa", split=args.split)
    dataset = dataset.shuffle(seed=args.seed)

    end = min(args.start + args.n, len(dataset))
    rng = random.Random(args.seed)

    for idx in range(args.start, end):
        ex = dataset[idx]
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

        prompt = f"{question}\n\n" + "\n".join(
            f"{chr(ord('A') + j)}. {c}" for j, c in enumerate(shuffled_choices)
        )

        # wrapped = _wrap_synth(prompt)
        output = _collect_generation(
            gen,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print("\n" + "=" * 80)
        print(f"IDX {idx} | correct={correct_letter}")
        print("-" * 80)
        print("PROMPT (user text):\n")
        print(prompt)
        if args.show_wrapped_prompt:
            print("\n" + "-" * 80)
            print("PROMPT (wrapped actually sent to model):\n")
            print(wrapped)
        print("\n" + "-" * 80)
        print("MODEL OUTPUT (full):\n")
        print(output)
        valid = {chr(ord("A") + j) for j in range(len(shuffled_choices))}
        answer_text = _extract_answer_after_thinking(output)
        predicted = _parse_choice(answer_text, valid, options=shuffled_choices)
        match = "OK" if predicted == correct_letter else "WRONG"
        print(f"Answer text: {repr(answer_text[:200]) if answer_text else None}")
        print(f"Parsed: {predicted} | Correct: {correct_letter} | {match}")
        # break
        print("\n" + "-" * 80)

        if args.pause:
            input("\n[enter] next... ")


if __name__ == "__main__":
    main()

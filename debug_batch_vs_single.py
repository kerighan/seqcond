#!/usr/bin/env python3
"""Compare generate() vs generate_batch() on the same prompts.
If they diverge, there's a batching bug."""
import random
from datasets import load_dataset
from seqcond.torch.generator import TorchGenerator


def collect_single(gen, prompt, max_new_tokens=512):
    """Use generate() one-by-one."""
    toks = []
    for tok in gen.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        verbose=False,
        use_cuda_graph=False,
        use_synth_template=True,
    ):
        toks.append(tok)
    return "".join(toks)


def main():
    gen = TorchGenerator("checkpoints/seqcond_torch_120k.pt")
    dataset = load_dataset("commonsense_qa", split="validation").shuffle(seed=42)
    rng = random.Random(42)

    N = 5
    prompts = []
    for i in range(N):
        ex = dataset[i]
        question = ex["question"]
        choices = ex["choices"]["text"]
        indexed = list(enumerate(choices))
        rng.shuffle(indexed)
        shuffled = [c for _, c in indexed]
        prompt = f"{question}\n\n" + "\n".join(
            f"{chr(ord('A') + j)}. {c}" for j, c in enumerate(shuffled)
        )
        prompts.append(prompt)

    # --- generate_batch ---
    print("=" * 60)
    print("BATCH (generate_batch, bs=5)")
    print("=" * 60)
    batch_outputs = gen.generate_batch(prompts, max_new_tokens=512, temperature=0.0)
    for i, out in enumerate(batch_outputs):
        print(f"\n[BATCH {i}] first 200 chars:")
        print(out[:200])
        print(f"[BATCH {i}] last 200 chars:")
        print(out[-200:])

    # --- generate (single) ---
    print("\n" + "=" * 60)
    print("SINGLE (generate, one by one)")
    print("=" * 60)
    single_outputs = []
    for i, p in enumerate(prompts):
        out = collect_single(gen, p, max_new_tokens=512)
        single_outputs.append(out)
        print(f"\n[SINGLE {i}] first 200 chars:")
        print(out[:200])
        print(f"[SINGLE {i}] last 200 chars:")
        print(out[-200:])

    # --- Compare ---
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    for i in range(N):
        match = batch_outputs[i] == single_outputs[i]
        b_len = len(batch_outputs[i])
        s_len = len(single_outputs[i])
        print(f"  [{i}] match={match} | batch_len={b_len} single_len={s_len}")
        if not match:
            # Find first divergence
            for j in range(min(b_len, s_len)):
                if batch_outputs[i][j] != single_outputs[i][j]:
                    print(f"       First diff at char {j}:")
                    print(f"       BATCH:  ...{batch_outputs[i][max(0,j-20):j+20]}...")
                    print(f"       SINGLE: ...{single_outputs[i][max(0,j-20):j+20]}...")
                    break


if __name__ == "__main__":
    main()

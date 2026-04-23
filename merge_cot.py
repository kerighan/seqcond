"""
merge_cot.py — Fusionne tous les *_cot.jsonl en un mega JSONL équilibré.

Prend au maximum N exemples par fichier (tirage aléatoire sans remise),
mélange tout et écrit dans data/merged_cot.jsonl.

Usage:
    python merge_cot.py                    # N=5000 par défaut
    python merge_cot.py --max_per_file 2000
    python merge_cot.py --output data/train_cot.jsonl
    python merge_cot.py --data_dir data/
"""
import argparse
import glob
import json
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/", help="Dossier contenant les *_cot.jsonl")
    parser.add_argument("--max_per_file", type=int, default=5000, help="Nombre max d'exemples par fichier")
    parser.add_argument("--output", default="data/train.jsonl", help="Fichier de sortie")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    files = sorted(glob.glob(os.path.join(args.data_dir, "*_cot.jsonl")))
    if not files:
        print(f"Aucun fichier *_cot.jsonl trouvé dans {args.data_dir}")
        return

    all_examples = []
    for path in files:
        name = os.path.basename(path)
        with open(path) as f:
            lines = [l for l in f if l.strip()]
        examples = [json.loads(l) for l in lines]
        # Normalise les entrées qui ont reasoning + answer séparés
        for ex in examples:
            if "answer" in ex and "reasoning" in ex:
                reasoning = ex["reasoning"].rstrip()
                answer    = ex["answer"]
                if "<|think_end|>" not in reasoning:
                    ex["reasoning"] = reasoning + "<|think_end|>\n" + answer
                del ex["answer"]
                ex.setdefault("reward", 1.0)
                ex.setdefault("advantage", 1.0)
        if len(examples) > args.max_per_file:
            examples = random.sample(examples, args.max_per_file)
        print(f"  {name}: {len(examples)} exemples")
        all_examples.extend(examples)

    random.shuffle(all_examples)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_examples)} exemples → {args.output}")


if __name__ == "__main__":
    main()

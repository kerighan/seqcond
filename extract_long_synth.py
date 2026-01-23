#!/usr/bin/env python3
import json
import os

os.environ["HF_TOKEN"] = ""
import tqdm
from seqcond.dataset import iterate_synth, tokenizer


def extract_long_samples(
    threshold=1024, output_file="/media/maixent/2To/corpus/long_synth_samples.jsonl"
):
    """
    Iterates through the SYNTH dataset, tokenizes samples,
    and saves those longer than the threshold to a JSONL file.
    """
    print(f"Extracting SYNTH samples longer than {threshold} tokens...")
    print(f"Output file: {output_file}")

    # We use shard_data=False to ensure we see the whole dataset
    # We use streaming=True (default in iterate_synth)
    iterator = iterate_synth(
        max_samples=None, tokenize=True, tok=tokenizer, shard_data=False
    )

    count_long = 0
    count_total = 0

    # Open file in append mode just in case of resume,
    # but for a fresh run you might want to delete it first.
    if os.path.exists(output_file):
        print(f"Warning: {output_file} already exists. Appending to it.")

    try:
        with open(output_file, "a", encoding="utf-8") as f:
            # We don't know the total size for tqdm because it's streaming,
            # but we can estimate or just show progress.
            pbar = tqdm.tqdm(unit=" samples")
            for tokens in iterator:
                count_total += 1
                length = len(tokens)

                if length > threshold:
                    count_long += 1
                    # Store as a simple JSON object per line
                    # We store the tokens directly as requested
                    entry = {"tokens": tokens, "length": length}
                    f.write(json.dumps(entry) + "\n")

                    if count_long % 100 == 0:
                        pbar.set_description(f"Found {count_long} long samples")

                pbar.update(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print(f"\nExtraction complete.")
        print(f"Total samples scanned: {count_total}")
        print(f"Long samples saved: {count_long}")


if __name__ == "__main__":
    extract_long_samples()

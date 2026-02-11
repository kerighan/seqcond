#!/usr/bin/env python3
"""Iterate over PleIAs/SYNTH and log one full record per unique exercise type."""
import json
from datasets import load_dataset

OUTPUT_FILE = "synth_exercise_samples.jsonl"

dataset = load_dataset("PleIAs/SYNTH", split="train", streaming=True)

seen_exercises = set()

with open(OUTPUT_FILE, "w") as f:
    for item in dataset:
        exercise = item.get("exercise", "")
        if exercise not in seen_exercises:
            seen_exercises.add(exercise)
            print(f"New exercise type: {exercise!r} (total unique: {len(seen_exercises)})")
            # Write full record as pretty JSON
            f.write("=" * 80 + "\n")
            f.write(f"EXERCISE TYPE: {exercise}\n")
            f.write("=" * 80 + "\n")
            for key, val in item.items():
                val_str = str(val)
                if len(val_str) > 2000:
                    val_str = val_str[:2000] + "... [TRUNCATED]"
                f.write(f"\n--- {key} ---\n{val_str}\n")
            f.write("\n\n")
            f.flush()

print(f"\nDone. Found {len(seen_exercises)} unique exercise types.")
print(f"Output written to {OUTPUT_FILE}")

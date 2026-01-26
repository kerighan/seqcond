
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from datasets import load_dataset
from seqcond.dataset import tokenizer, format_synth_item
import time

def check_synth():
    print("Loading PleIAs/SYNTH (streaming=True)...")
    try:
        dataset = load_dataset("PleIAs/SYNTH", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Iterating...")
    count = 0
    start = time.time()
    
    try:
        for i, item in enumerate(dataset):
            if i == 0:
                print("First item keys:", item.keys())
                print("First item text length:", len(format_synth_item(item)))
            
            count += 1
            if count % 1000 == 0:
                print(f"Read {count} items...", end="\r")
            
            # Just check the first 5000 items to see if it crashes or stops early
            if count >= 5000:
                print("\nReached 5000 items without issue.")
                break
    except Exception as e:
        print(f"\nStopped with error at index {count}: {e}")

    elapsed = time.time() - start
    print(f"\nProcessed {count} items in {elapsed:.2f}s")

if __name__ == "__main__":
    check_synth()

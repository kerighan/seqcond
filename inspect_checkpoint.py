#!/usr/bin/env python3
"""Inspect checkpoint to find Unicode string data causing broadcast issues."""
import pickle
import sys
import jax.tree_util as tree_util


def find_unicode_strings(tree, path=""):
    """Recursively find Unicode string arrays in a tree structure."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            find_unicode_strings(v, f"{path}.{k}" if path else k)
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            find_unicode_strings(v, f"{path}[{i}]")
    elif hasattr(tree, "dtype"):
        if tree.dtype.kind == "U":  # Unicode string
            print(
                f"Found Unicode string at {path}: dtype={tree.dtype}, shape={tree.shape}"
            )
            print(f"  Value: {tree}")
    elif isinstance(tree, str):
        print(f"Found string at {path}: {repr(tree)}")


if __name__ == "__main__":
    ckpt_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "checkpoints/seqcond-l24-d1024-th16-sh16-m2-r2-o0-a0_step20000.pkl"
    )

    print(f"Loading checkpoint: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        try:
            data = pickle.load(f)
        except:
            f.seek(0)
            data = pickle.load(f, encoding="latin1")

    print("\nCheckpoint keys:", data.keys())
    print("\nSearching for Unicode strings...")

    for key in data.keys():
        print(f"\n=== Checking {key} ===")
        find_unicode_strings(data[key], key)

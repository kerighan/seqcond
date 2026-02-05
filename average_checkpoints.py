#!/usr/bin/env python3
"""Average multiple checkpoints with weighted combination."""

import argparse
import pickle
import os
import glob
import jax.numpy as jnp
from jax import tree_util


def find_checkpoint(checkpoint_dir: str, step: int) -> str:
    """Find checkpoint file for a given step."""
    pattern = os.path.join(checkpoint_dir, f"*_step{step}.pkl")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found for step {step} in {checkpoint_dir}"
        )
    if len(matches) > 1:
        raise ValueError(f"Multiple checkpoints found for step {step}: {matches}")
    return matches[0]


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint file."""
    print(f"  Loading: {os.path.basename(path)}")
    with open(path, "rb") as f:
        try:
            data = pickle.load(f)
        except (UnicodeDecodeError, ValueError):
            f.seek(0)
            data = pickle.load(f, encoding="latin1")
    return data


def accumulate_weighted_params(accumulated, new_params, weight):
    """Add weighted params to accumulated result."""
    if accumulated is None:
        # First checkpoint: multiply by weight
        return tree_util.tree_map(lambda p: jnp.array(p) * weight, new_params)
    else:
        # Add weighted params to accumulator
        return tree_util.tree_map(
            lambda acc, p: acc + jnp.array(p) * weight, accumulated, new_params
        )


def main():
    parser = argparse.ArgumentParser(
        description="Average checkpoints with weighted combination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python average_checkpoints.py 200000,1.0 220000,0.75 240000,0.5 -o averaged.pkl
  
This will create a weighted average of checkpoints at steps 200000, 220000, 240000
with weights 1.0, 0.75, 0.5 (L1 normalized to sum to 1).
        """,
    )
    parser.add_argument(
        "step_weights",
        nargs="+",
        help="Step,weight pairs (e.g., '200000,1.0' '220000,0.75')",
    )
    parser.add_argument(
        "-d",
        "--checkpoint-dir",
        default="/media/maixent/2To/seqcond_checkpoints",
        help="Directory containing checkpoints (default: /media/maixent/2To/seqcond_checkpoints)",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output path for averaged checkpoint"
    )

    args = parser.parse_args()

    # Parse step,weight pairs
    step_weight_pairs = []
    for sw in args.step_weights:
        parts = sw.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid step,weight pair: {sw}. Expected format: step,weight"
            )
        step = int(parts[0])
        weight = float(parts[1])
        step_weight_pairs.append((step, weight))

    # L1 normalize weights
    weights = [w for _, w in step_weight_pairs]
    total_weight = sum(weights)
    norm_weights = [w / total_weight for w in weights]

    print(f"Averaging {len(step_weight_pairs)} checkpoints:")
    for (step, weight), norm_w in zip(step_weight_pairs, norm_weights):
        print(f"  Step {step}: weight {weight:.4f} (normalized: {norm_w:.4f})")

    # Process checkpoints one by one
    print("\nProcessing checkpoints incrementally...")
    accumulated_params = None
    base_config = None

    for i, ((step, _), norm_weight) in enumerate(zip(step_weight_pairs, norm_weights)):
        print(f"  [{i+1}/{len(step_weight_pairs)}] Loading step {step}...")
        ckpt_path = find_checkpoint(args.checkpoint_dir, step)
        ckpt = load_checkpoint(ckpt_path)

        # Save config from first checkpoint
        if base_config is None:
            base_config = ckpt["config"]

        # Accumulate weighted params
        print(f"       Accumulating with weight {norm_weight:.4f}...")
        accumulated_params = accumulate_weighted_params(
            accumulated_params, ckpt["params"], norm_weight
        )

        # Free memory
        del ckpt
        print(f"       Done (freed memory)")

    print("\nWeighted averaging complete!")
    averaged_params = accumulated_params

    # Create output checkpoint
    output_data = {
        "params": averaged_params,
        "config": base_config,
        # Note: opt_state is not averaged, omitted from output
    }

    # Add metadata about the averaging
    output_data["averaged_from"] = [
        {"step": step, "weight": weight} for step, weight in step_weight_pairs
    ]

    # Save
    print(f"\nSaving to: {args.output}")
    with open(args.output, "wb") as f:
        pickle.dump(output_data, f)

    print("Done!")


if __name__ == "__main__":
    main()

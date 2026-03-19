#!/usr/bin/env python3
"""Plot llm_avg evolution from training results."""

import re
import sys
from pathlib import Path

import numpy as np


def parse_results(filepath):
    """Extract step and llm_avg from results file."""
    steps = []
    llm_avgs = []

    with open(filepath) as f:
        for line in f:
            # Match: "Step   42/1000 | loss=... | llm_avg=0.525 | ..."
            m = re.search(r"Step\s+(\d+)/\d+.*reward_avg=([\d.]+)", line)
            if m:
                steps.append(int(m.group(1)))
                llm_avgs.append(float(m.group(2)))

    return np.array(steps), np.array(llm_avgs)


def smooth(y, window=5):
    """Simple moving average smoothing."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    smoothed = np.convolve(y, kernel, mode="valid")
    # Pad start to match original length
    pad = len(y) - len(smoothed)
    return np.concatenate([y[:pad], smoothed])


def main():
    results_file = Path("results.txt")
    if not results_file.exists():
        print(f"ERROR: {results_file} not found")
        sys.exit(1)

    steps, llm_avgs = parse_results(results_file)
    if len(steps) == 0:
        print("ERROR: No training steps found in results.txt")
        sys.exit(1)

    print(f"Parsed {len(steps)} steps")
    print(f"  Min llm_avg: {llm_avgs.min():.3f}")
    print(f"  Max llm_avg: {llm_avgs.max():.3f}")
    print(f"  Mean llm_avg: {llm_avgs.mean():.3f}")
    print()

    # Try matplotlib first, fall back to ASCII
    try:
        import matplotlib.pyplot as plt

        smoothed = smooth(llm_avgs, window=10)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(steps, llm_avgs, alpha=0.3, s=20, label="Raw (noisy)")
        ax.plot(steps, smoothed, color="red", linewidth=2, label="Smoothed (window=10)")
        ax.set_xlabel("Step")
        ax.set_ylabel("llm_avg")
        ax.set_title("Training Progress: LLM Reward Average")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig("training_plot.png", dpi=100)
        print("✓ Saved: training_plot.png")
        plt.show()
    except ImportError:
        print("matplotlib not found, using ASCII plot instead")
        print()
        ascii_plot(steps, llm_avgs)


def ascii_plot(steps, values, height=20, width=80):
    """Simple ASCII plot."""
    if len(values) == 0:
        return

    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        vmax = vmin + 1

    # Downsample if too many points
    if len(values) > width:
        indices = np.linspace(0, len(values) - 1, width, dtype=int)
        plot_vals = values[indices]
        plot_steps = steps[indices]
    else:
        plot_vals = values
        plot_steps = steps

    # Normalize to [0, height]
    normalized = (plot_vals - vmin) / (vmax - vmin) * (height - 1)

    # Build plot
    grid = [[" " for _ in range(len(plot_vals))] for _ in range(height)]
    for i, val in enumerate(normalized):
        row = height - 1 - int(val)
        grid[row][i] = "●"

    # Print
    print(f"  {vmax:.3f} ┤", end="")
    print(grid[0])
    for row in grid[1:-1]:
        print("        ┤", end="")
        print("".join(row))
    print(f"  {vmin:.3f} ┤", end="")
    print(grid[-1])
    print("        └" + "─" * len(plot_vals))
    print(f"        {plot_steps[0]}...{plot_steps[-1]}")


if __name__ == "__main__":
    main()

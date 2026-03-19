#!/usr/bin/env python3
"""Plot llm_avg evolution from training results."""

import re
import sys
from pathlib import Path

import numpy as np


def parse_results(filepath):
    """Extract step, reward_avg, and correct ratio from results file."""
    steps = []
    reward_avgs = []
    correct_rates = []

    with open(filepath) as f:
        for line in f:
            m = re.search(
                r"Step\s+(\d+)/\d+.*reward_avg=([\d.]+).*correct=(\d+)/(\d+)",
                line,
            )
            if m:
                steps.append(int(m.group(1)))
                reward_avgs.append(float(m.group(2)))
                num_correct = int(m.group(3))
                num_total = int(m.group(4))
                correct_rates.append(num_correct / max(num_total, 1))

    return np.array(steps), np.array(reward_avgs), np.array(correct_rates)


def smooth(y, window=5):
    """Simple moving average smoothing."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    left = window // 2
    right = window - 1 - left
    padded = np.pad(y, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main():
    results_file = Path("results.txt")
    if not results_file.exists():
        print(f"ERROR: {results_file} not found")
        sys.exit(1)

    steps, reward_avgs, correct_rates = parse_results(results_file)
    if len(steps) == 0:
        print("ERROR: No training steps found in results.txt")
        sys.exit(1)

    print(f"Parsed {len(steps)} steps")
    print(f"  Min reward_avg: {reward_avgs.min():.3f}")
    print(f"  Max reward_avg: {reward_avgs.max():.3f}")
    print(f"  Mean reward_avg: {reward_avgs.mean():.3f}")
    print(f"  Mean correct rate: {correct_rates.mean():.3f}")
    print()

    try:
        import matplotlib.pyplot as plt

        reward_smoothed = smooth(reward_avgs, window=50)
        correct_smoothed = smooth(correct_rates, window=50)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax2 = ax.twinx()

        ax.scatter(
            steps, reward_avgs, alpha=0.2, s=20, color="tab:red", label="Reward raw"
        )
        reward_line = ax.plot(
            steps,
            reward_smoothed,
            color="tab:red",
            linewidth=2,
            label="Reward smooth",
        )
        correct_line = ax2.plot(
            steps,
            correct_smoothed,
            color="tab:blue",
            linewidth=2,
            label="Correct smooth",
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("reward_avg", color="tab:red")
        ax2.set_ylabel("correct rate", color="tab:blue")
        ax2.set_ylim(0.0, 1.0)
        ax.set_title("Training Progress: reward_avg and correct rate")
        ax.grid(True, alpha=0.3)
        handles = reward_line + correct_line
        labels = [line.get_label() for line in handles]
        ax.legend(handles, labels)

        plt.tight_layout()
        plt.savefig("training_plot.png", dpi=100)
        print("✓ Saved: training_plot.png")
        plt.show()
    except ImportError:
        print("matplotlib not found, using ASCII plot instead")
        print()
        print("reward_avg")
        ascii_plot(steps, reward_avgs)
        print()
        print("correct rate")
        ascii_plot(steps, correct_rates)


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

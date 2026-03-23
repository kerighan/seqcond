#!/usr/bin/env python3
"""Plot reward_avg and overall_avg from RLAI training results."""

import re
import sys
from pathlib import Path

import numpy as np


def parse_results(filepath):
    """Extract step, reward_avg, and overall_avg from results file."""
    steps = []
    reward_avgs = []
    overall_avgs = []

    with open(filepath) as f:
        for line in f:
            m = re.search(
                r"Step\s+(\d+)/\d+.*reward_avg=([\d.]+).*overall_avg=([\d.]+)",
                line,
            )
            if m:
                steps.append(int(m.group(1)))
                reward_avgs.append(float(m.group(2)))
                overall_avgs.append(float(m.group(3)))

    return np.array(steps), np.array(reward_avgs), np.array(overall_avgs)


def smooth(y, window=5):
    """Simple moving average smoothing."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    left = window // 2
    right = window - 1 - left
    padded = np.pad(y, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def rolling_std(y, window=50):
    """Rolling standard deviation (same padding as smooth)."""
    if len(y) < window:
        return np.zeros_like(y)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(y, (left, right), mode="edge")
    out = np.empty(len(y))
    for i in range(len(y)):
        out[i] = np.std(padded[i : i + window])
    return out


def main():
    results_file = Path("results.txt")
    if not results_file.exists():
        print(f"ERROR: {results_file} not found")
        sys.exit(1)

    steps, reward_avgs, overall_avgs = parse_results(results_file)
    if len(steps) == 0:
        print("ERROR: No training steps found in results.txt")
        sys.exit(1)

    print(f"Parsed {len(steps)} steps")
    print(
        f"  reward_avg:  min={reward_avgs.min():.3f}  max={reward_avgs.max():.3f}  mean={reward_avgs.mean():.3f}"
    )
    print(
        f"  overall_avg: min={overall_avgs.min():.1f}  max={overall_avgs.max():.1f}  mean={overall_avgs.mean():.1f}"
    )
    print()

    try:
        import matplotlib.pyplot as plt

        w = 50
        reward_smoothed = smooth(reward_avgs, window=w)
        overall_smoothed = smooth(overall_avgs, window=w)
        reward_std = rolling_std(reward_avgs, window=w)
        overall_std = rolling_std(overall_avgs, window=w)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax2 = ax.twinx()

        # reward_avg: line + ±1σ band
        reward_line = ax.plot(
            steps,
            reward_smoothed,
            color="tab:red",
            linewidth=2,
            label="reward_avg",
        )
        ax.fill_between(
            steps,
            reward_smoothed - reward_std,
            reward_smoothed + reward_std,
            color="tab:red",
            alpha=0.15,
            label="±1σ",
        )

        # overall_avg: line + ±1σ band
        overall_line = ax2.plot(
            steps,
            overall_smoothed,
            color="tab:blue",
            linewidth=2,
            label="overall_avg",
        )
        ax2.fill_between(
            steps,
            overall_smoothed - overall_std,
            overall_smoothed + overall_std,
            color="tab:blue",
            alpha=0.12,
            label="±1σ",
        )

        ax.set_xlabel("Step")
        ax.set_ylabel("reward_avg", color="tab:red")
        ax2.set_ylabel("overall_avg", color="tab:blue")
        ax.set_title("RLAI Training: reward_avg & overall_avg (±1σ band)")
        ax.grid(True, alpha=0.3)
        handles = reward_line + overall_line
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
        print("overall_avg")
        ascii_plot(steps, overall_avgs)


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

#!/usr/bin/env python3
"""Plot GRPO training: correct rate per step + smoothed curve + batch boundaries + eval markers."""

import re
import sys
from pathlib import Path

import numpy as np


def parse_log(filepath):
    """Parse GRPO training log. Returns steps, correct, total, optimizer_steps, eval_results."""
    steps, correct, total = [], [], []
    optimizer_steps = []  # steps where optimizer.step happened
    eval_results = []  # (step, pass_k, acc%)

    with open(filepath) as f:
        for line in f:
            # Training step: Step  17/5000 | ... | correct=20/48
            m = re.search(r"Step\s+(\d+)/\d+.*correct=(\d+)/(\d+)", line)
            if m:
                steps.append(int(m.group(1)))
                correct.append(int(m.group(2)))
                total.append(int(m.group(3)))
                continue

            # Optimizer step
            m = re.search(r"optimizer\.step \| train_step=(\d+)", line)
            if m:
                optimizer_steps.append(int(m.group(1)))
                continue

            # Eval: step=64  pass@3: 43.3%  correct=65/150
            m = re.search(r"step=(\d+)\s+pass@(\d+):\s+([\d.]+)%\s+correct=(\d+)/(\d+)", line)
            if m:
                eval_results.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))

    return (np.array(steps), np.array(correct), np.array(total),
            sorted(set(optimizer_steps)), eval_results)


def smooth(y, window):
    if len(y) < window:
        return y.copy()
    kernel = np.ones(window) / window
    left = window // 2
    right = window - 1 - left
    padded = np.pad(y, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main():
    # Accept log file as argument, or search for common names
    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
    else:
        candidates = [
            Path("checkpoints/grpo_train.log"),
            Path("checkpoints/seqcond_lin5_grpo_eval.log"),
            Path("training.log"),
            Path("results.txt"),
        ]
        filepath = None
        for c in candidates:
            if c.exists():
                filepath = c
                break
        if filepath is None:
            # Try to find any file with Step lines
            import glob
            for f in glob.glob("**/*.log", recursive=True):
                with open(f) as fh:
                    if "correct=" in fh.read(2000):
                        filepath = Path(f)
                        break
        if filepath is None:
            print("ERROR: No log file found. Pass as argument: python plot_training.py <logfile>")
            sys.exit(1)

    print(f"Reading: {filepath}")
    steps, correct, total, opt_steps, eval_results = parse_log(filepath)
    if len(steps) == 0:
        print("ERROR: No training steps found")
        sys.exit(1)

    G = total[0] if len(total) > 0 else 1
    rate = correct / np.maximum(total, 1)  # correct rate per step

    # Stats
    n_zero = np.sum(correct == 0)
    n_full = np.sum(correct == total)
    n_skip = np.sum(rate == 0) + np.sum(rate == 1)  # zero-variance (would be skipped by Dr.GRPO)
    print(f"  {len(steps)} steps, G={G}, {len(opt_steps)} optimizer updates")
    print(f"  Correct rate: mean={rate.mean():.1%}  median={np.median(rate):.1%}")
    print(f"  Zero batches (0/{G}): {n_zero} ({n_zero/len(steps):.0%})")
    print(f"  Full batches ({G}/{G}): {n_full} ({n_full/len(steps):.0%})")
    print(f"  Zero-variance (skip): {n_skip} ({n_skip/len(steps):.0%})")
    if eval_results:
        print(f"  Evals: {len(eval_results)}")
        for s, k, acc in eval_results:
            print(f"    step={s}  pass@{k}: {acc:.1f}%")
    print()

    # Also parse loss values
    losses = []
    with open(filepath) as f:
        for line in f:
            m = re.search(r"Step\s+\d+/\d+.*loss=([-\d.]+)", line)
            if m:
                losses.append(float(m.group(1)))
    losses = np.array(losses[:len(steps)])

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Adaptive smoothing: ~1 batch worth of steps, minimum 3
    grad_accum = opt_steps[1] - opt_steps[0] if len(opt_steps) >= 2 else 8
    smooth_window = max(4, min(grad_accum, len(steps) // 4))
    rate_smooth = smooth(rate, window=smooth_window)

    fig, (ax, ax_loss) = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[3, 1],
                                       sharex=True, gridspec_kw={"hspace": 0.08})

    # ── Top: Correct rate ──
    # Per-step correct rate as thin bars with color coding
    for i in range(len(steps)):
        if correct[i] == 0:
            color, alpha = "#d62728", 0.5  # red for zero
        elif correct[i] == total[i]:
            color, alpha = "#2ca02c", 0.5  # green for full
        else:
            color, alpha = "#4a90d9", 0.35  # blue for partial
        ax.bar(steps[i], rate[i], width=0.8, color=color, alpha=alpha, linewidth=0)

    # Smoothed curve
    ax.plot(steps, rate_smooth, color="#1a5276", linewidth=2.5,
            label=f"Smoothed (w={smooth_window})", zorder=5)

    # Optimizer step boundaries (vertical lines) — visible
    for i, os_step in enumerate(opt_steps):
        ax.axvline(x=os_step, color="#e74c3c", alpha=0.6, linewidth=1.2, linestyle="-",
                   label="Optimizer step" if i == 0 else None)
        ax_loss.axvline(x=os_step, color="#e74c3c", alpha=0.6, linewidth=1.2, linestyle="-")

    # Eval markers
    if eval_results:
        eval_steps_list = [e[0] for e in eval_results]
        eval_accs = [e[2] / 100 for e in eval_results]
        ax.scatter(eval_steps_list, eval_accs, color="#ff7f0e", s=100, zorder=10,
                   marker="D", edgecolors="black", linewidth=0.8, label="Eval pass@k")
        for s, k, acc in eval_results:
            ax.annotate(f"{acc:.1f}%", (s, acc/100), textcoords="offset points",
                        xytext=(6, 10), fontsize=9, color="#ff7f0e", fontweight="bold")

    ax.set_ylabel("Correct rate (per step)", fontsize=11)
    ax.set_title(f"GRPO Training — G={G}, {len(steps)} steps, {len(opt_steps)} updates, "
                 f"accum={grad_accum}", fontsize=13)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=9)

    # Stats text box
    stats_text = (f"mean={rate.mean():.1%}  zero={n_zero/len(steps):.0%}  "
                  f"full={n_full/len(steps):.0%}  skip={n_skip/len(steps):.0%}")
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    # ── Bottom: Loss ──
    if len(losses) > 0:
        loss_smooth = smooth(losses, window=smooth_window)
        ax_loss.bar(steps[:len(losses)], losses, width=0.8, color="#7f8c8d", alpha=0.3, linewidth=0)
        ax_loss.plot(steps[:len(losses)], loss_smooth, color="#2c3e50", linewidth=2)
        ax_loss.set_ylabel("Loss", fontsize=11)
        ax_loss.grid(True, alpha=0.2)
    ax_loss.set_xlabel("Step", fontsize=11)

    plt.tight_layout()
    out = str(filepath).replace(".log", "_plot.png").replace(".txt", "_plot.png")
    if out == str(filepath):
        out = "training_plot.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()

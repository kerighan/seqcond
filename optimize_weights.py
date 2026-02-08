#!/usr/bin/env python3
"""Find optimal checkpoint weights to maximize benchmark performance."""

import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import argparse
import subprocess
import os


# Example benchmark data: rows are checkpoints, columns are benchmarks
# Format: step (in k), then benchmark scores
EXAMPLE_DATA = """
200	42.57	53.04	68.88	34.07	31.4	44.48	22.22
230	43.21	51.85	68.17	33.82	30.6	45.72	19.7
240	43.01	52.57	68.23	34	30.2	46.08	25.25
250	43.78	54.14	69.26	34.97	31.4	44.9	22.73
260	43.42	51.93	68.34	34	30.5	45.5	21.5
270	43.83	52.25	68.28	33.99	30.8	46.01	20.71
"""


def parse_data(data_str: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse benchmark data string into steps and scores arrays."""
    lines = [l.strip() for l in data_str.strip().split("\n") if l.strip()]
    rows = []
    steps = []
    for line in lines:
        # Handle both comma and dot decimals, tab/space separators
        line = line.replace(",", ".")
        parts = line.split()
        # Filter out empty parts
        parts = [p for p in parts if p]
        print(parts)
        steps.append(int(float(parts[0])))
        scores = [float(p) for p in parts[1:] if p]
        rows.append(scores)

    # Ensure all rows have same length (pad with NaN if needed)
    max_len = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_len:
            r.append(np.nan)

    return np.array(steps), np.array(rows)


def weighted_scores(scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted average scores for each benchmark."""
    # weights should sum to 1
    weights = weights / weights.sum()
    return np.dot(weights, scores)


def objective_total_score(weights: np.ndarray, scores: np.ndarray) -> float:
    """Objective: maximize total score (return negative for minimization)."""
    w = weights / weights.sum()
    combined = np.dot(w, scores)
    # Mean across benchmarks (ignoring NaN)
    return -np.nanmean(combined)


def compute_ranks(scores: np.ndarray) -> np.ndarray:
    """Compute ranks for each checkpoint within each benchmark (1 = best)."""
    n_ckpts, n_benchmarks = scores.shape
    ranks = np.zeros_like(scores)
    for j in range(n_benchmarks):
        col = scores[:, j]
        # Higher score = better = lower rank
        valid_mask = ~np.isnan(col)
        if valid_mask.sum() > 0:
            # argsort of negative gives descending order
            order = np.argsort(-col[valid_mask])
            rank_values = np.zeros(valid_mask.sum())
            rank_values[order] = np.arange(1, valid_mask.sum() + 1)
            ranks[valid_mask, j] = rank_values
            ranks[~valid_mask, j] = np.nan
    return ranks


def objective_rank(
    weights: np.ndarray, scores: np.ndarray, all_scores_with_weighted: callable
) -> float:
    """Objective: minimize average rank of weighted model across benchmarks."""
    w = weights / weights.sum()
    combined = np.dot(w, scores)

    # For each benchmark, compute rank of combined score among original scores
    n_benchmarks = scores.shape[1]
    ranks = []
    for j in range(n_benchmarks):
        col = scores[:, j]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            continue
        # How many original scores are >= combined score?
        rank = 1 + np.sum(col[valid] > combined[j])
        ranks.append(rank)

    avg_rank = np.mean(ranks) if ranks else float("inf")
    # Add small penalty based on negative mean score to break ties
    # This favors higher scores when ranks are equal
    score_penalty = -np.nanmean(combined) * 0.001
    return avg_rank + score_penalty


def objective_hybrid(
    weights: np.ndarray, scores: np.ndarray, alpha: float = 0.5
) -> float:
    """Hybrid objective: ranks in units, scores in decimals.

    Returns rank + 0.001 * (-score) so that:
    - Integer part = average rank (minimize)
    - Decimal part = score contribution (maximize score = minimize negative score)

    Args:
        weights: Checkpoint weights
        scores: Score matrix
        alpha: Unused, kept for compatibility
    """
    w = weights / weights.sum()
    combined = np.dot(w, scores)

    # Rank component (in units)
    n_benchmarks = scores.shape[1]
    ranks = []
    for j in range(n_benchmarks):
        col = scores[:, j]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            continue
        rank = 1 + np.sum(col[valid] > combined[j])
        ranks.append(rank)

    avg_rank = np.mean(ranks) if ranks else float("inf")

    # Score component (in decimals, 0.001 scale)
    # Negative because we want to maximize score (minimize negative score)
    # Mean score is typically 0-100, so 0.001 * score gives 0.000 to 0.100
    score_component = -np.nanmean(combined) * 0.001

    # Combine: rank (units) + score penalty (decimals)
    return avg_rank + score_component


def topn_softmax_weights(
    scores: np.ndarray,
    n: int,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select top N checkpoints by mean score and weight them using softmax.

    Args:
        scores: (n_checkpoints, n_benchmarks) array of scores
        n: Number of top checkpoints to select
        temperature: Softmax temperature (lower = sharper distribution)

    Returns:
        indices: Indices of selected checkpoints
        weights: Softmax weights based on mean scores
    """
    means = np.nanmean(scores, axis=1)

    # Get top N indices by mean score
    top_indices = np.argsort(means)[-n:][::-1]  # Descending order
    top_means = means[top_indices]

    # Softmax weighting
    # Shift for numerical stability
    shifted = (top_means - top_means.max()) / temperature
    exp_scores = np.exp(shifted)
    weights = exp_scores / exp_scores.sum()

    print(f"Top {n} checkpoints by mean score:")
    for idx, (i, m, w) in enumerate(zip(top_indices, top_means, weights)):
        print(f"  {idx+1}. Index {i}: mean={m:.2f}, weight={w:.4f}")

    return top_indices, weights


def optimize_weights(
    scores: np.ndarray,
    n_checkpoints: int = None,
    method: str = "total",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Find optimal weights for checkpoint averaging.

    Args:
        scores: (n_checkpoints, n_benchmarks) array of scores
        n_checkpoints: Number of checkpoints to use (None = use all)
        method: "total" for max total score, "rank" for best average rank, "hybrid" for balanced approach
        verbose: Print progress

    Returns:
        best_indices: Indices of selected checkpoints
        best_weights: Optimal weights (L1 normalized)
        best_score: Objective value
    """
    n_total = scores.shape[0]

    if n_checkpoints is None:
        n_checkpoints = n_total

    if n_checkpoints > n_total:
        n_checkpoints = n_total

    best_result = None
    best_indices = None
    best_obj = float("inf")

    # Try all combinations of n_checkpoints
    n_combinations = len(list(combinations(range(n_total), n_checkpoints)))
    if verbose:
        print(f"\nTesting {n_combinations} combinations...")

    for comb_idx, indices in enumerate(combinations(range(n_total), n_checkpoints)):
        indices = list(indices)
        sub_scores = scores[indices]

        # Bounds: weights in [0, 1]
        bounds = [(0.0, 1.0)] * n_checkpoints

        # Constraint: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}

        if method == "total":
            obj_fn = lambda w: objective_total_score(w, sub_scores)
        elif method == "rank":
            obj_fn = lambda w: objective_rank(w, sub_scores, None)
        else:  # hybrid
            obj_fn = lambda w: objective_hybrid(w, sub_scores)

        # Try multiple random initializations to avoid local minima
        best_local_result = None
        best_local_obj = float("inf")

        # Try uniform initialization
        w0 = np.ones(n_checkpoints) / n_checkpoints
        result = minimize(
            obj_fn,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )
        if result.fun < best_local_obj:
            best_local_obj = result.fun
            best_local_result = result

        # Try 5 random initializations
        np.random.seed(42 + comb_idx)  # Reproducible but different per combination
        for _ in range(5):
            # Random weights that sum to 1
            w_random = np.random.dirichlet(np.ones(n_checkpoints))
            result = minimize(
                obj_fn,
                w_random,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )
            if result.fun < best_local_obj:
                best_local_obj = result.fun
                best_local_result = result

        # Try initialization favoring best individual checkpoint
        individual_means = np.nanmean(sub_scores, axis=1)
        best_idx = np.argmax(individual_means)
        w_biased = np.ones(n_checkpoints) * 0.1
        w_biased[best_idx] = 1.0 - 0.1 * (n_checkpoints - 1)
        result = minimize(
            obj_fn,
            w_biased,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )
        if result.fun < best_local_obj:
            best_local_obj = result.fun
            best_local_result = result

        if verbose and n_combinations <= 20:
            print(
                f"  Combination {comb_idx+1}/{n_combinations}: indices={indices}, obj={best_local_obj:.4f}, weights={best_local_result.x}"
            )

        if best_local_obj < best_obj:
            best_obj = best_local_obj
            best_result = best_local_result
            best_indices = indices

    if verbose and best_result is not None:
        print(f"\nBest combination: checkpoints at indices {best_indices}")
        print(f"Optimal weights: {best_result.x}")
        print(f"Objective value: {-best_obj if method == 'total' else best_obj:.4f}")

    return np.array(best_indices), best_result.x, best_obj


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal checkpoint weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python optimize_weights.py --n 3 --method total
  
  This finds the best 3 checkpoints and their weights to maximize total score.
        """,
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="checkpoints/benchmarks.tsv",
        help="Path to TSV file with benchmark data (step, scores...)",
    )
    parser.add_argument(
        "--n",
        "-n",
        type=int,
        default=4,
        help="Number of checkpoints to use (default: all)",
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["total", "rank", "hybrid"],
        default="hybrid",
        help="Optimization method: 'total' (max score), 'rank' (best avg rank), or 'hybrid' (balanced)",
    )
    parser.add_argument(
        "--generate",
        "-g",
        action="store_true",
        help="Generate the averaged checkpoint (runs average_checkpoints.py and convert_jax_to_torch.py)",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=None,
        help="Use top N checkpoints by mean score with softmax weighting (alternative to optimization)",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=1.0,
        help="Temperature for softmax weighting with --topn (lower = sharper, default=1.0)",
    )

    args = parser.parse_args()

    # Load data
    if args.data:
        with open(args.data) as f:
            data_str = f.read()
    else:
        print("Using example data:")
        print(EXAMPLE_DATA)
        data_str = EXAMPLE_DATA

    steps, scores = parse_data(data_str)

    print(f"\nLoaded {len(steps)} checkpoints with {scores.shape[1]} benchmarks")
    print(f"Steps: {steps}")
    print(f"Scores shape: {scores.shape}")

    # Show individual checkpoint performance
    print("\n--- Individual checkpoint scores ---")
    print(f"{'Step':>6} | {'Mean':>8} | {'Benchmarks...'}")
    for i, (step, row) in enumerate(zip(steps, scores)):
        mean = np.nanmean(row)
        print(f"{step:>6} | {mean:>8.2f} | {row}")

    # Optimize or use top-N with softmax
    if args.topn:
        print(
            f"\n--- Using top {args.topn} checkpoints with softmax weighting (T={args.temperature}) ---"
        )
        best_indices, best_weights = topn_softmax_weights(
            scores, n=args.topn, temperature=args.temperature
        )
        best_obj = None
    else:
        print(f"\n--- Optimizing with method='{args.method}', n={args.n or 'all'} ---")
        best_indices, best_weights, best_obj = optimize_weights(
            scores, n_checkpoints=args.n, method=args.method
        )

    # Show results
    print("\n" + "=" * 60)
    print("OPTIMAL WEIGHTING")
    print("=" * 60)

    selected_steps = steps[best_indices]
    print("\nCheckpoints to average:")
    for step, weight in zip(selected_steps, best_weights):
        if weight > 0.001:  # Only show non-zero weights
            print(f"  Step {step}k: weight = {weight:.4f}")

    # Compute final combined scores
    combined = np.dot(best_weights, scores[best_indices])
    print(f"\nPredicted combined scores per benchmark: {combined}")
    print(f"Predicted mean score: {np.nanmean(combined):.2f}")

    # Compare to best single checkpoint
    single_means = np.nanmean(scores, axis=1)
    best_single_idx = np.argmax(single_means)
    print(
        f"\nBest single checkpoint: step {steps[best_single_idx]}k with mean {single_means[best_single_idx]:.2f}"
    )

    # Generate command for average_checkpoints.py
    n_label = args.topn if args.topn else args.n
    output_pkl = f"checkpoints/seqcond_opt_{n_label}.pkl"
    output_torch = f"checkpoints/seqcond_opt_{n_label}_torch.pt"

    cmd_avg_parts = ["python", "average_checkpoints.py"]
    for step, weight in zip(selected_steps, best_weights):
        if weight > 0.001:
            cmd_avg_parts.append(f"{int(step * 1000)},{weight:.4f}")
    cmd_avg_parts.extend(["-o", output_pkl])

    cmd_convert_parts = [
        "python",
        "convert_jax_to_torch.py",
        output_pkl,
        "--torch_path",
        output_torch,
    ]

    print("\n--- Command to create averaged checkpoint ---")
    print(" ".join(cmd_avg_parts))
    print(" ".join(cmd_convert_parts))

    if args.generate:
        print("\n--- Generating averaged checkpoint ---")
        print(f"Running: {' '.join(cmd_avg_parts)}")
        result = subprocess.run(
            cmd_avg_parts, cwd=os.path.dirname(os.path.abspath(__file__)) or "."
        )
        if result.returncode != 0:
            print(f"Error: average_checkpoints.py failed with code {result.returncode}")
            return

        print(f"\nRunning: {' '.join(cmd_convert_parts)}")
        result = subprocess.run(
            cmd_convert_parts, cwd=os.path.dirname(os.path.abspath(__file__)) or "."
        )
        if result.returncode != 0:
            print(
                f"Error: convert_jax_to_torch.py failed with code {result.returncode}"
            )
            return

        print(f"\n✓ Generated: {output_torch}")


if __name__ == "__main__":
    main()

"""Score every method by gap to the brute-force optimum.

Peer comparison alone cannot tell you whether the methods are all wrong
together. On universes small enough to enumerate, this measures each method
against ground truth.

    python scripts/validate_against_optimum.py [--json PATH]

By default the script prints and writes no files. With `--json PATH` it also
writes the per-problem gaps and the summary table to PATH. QAOA is not part of
this comparison: it is capped at 14 assets and these problems have 12 to 20.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantlab.evaluation.ground_truth import exhaustive_best, subset_count
from quantlab.optimizers import ALL_OPTIMIZERS, Constraints, PortfolioProblem

TRADING_DAYS = 252


def synthetic_problem(
    n: int, seed: int, cap: int, risk_aversion: float = 25.0, max_weight: float = 0.4
) -> PortfolioProblem:
    rng = np.random.default_rng(seed)
    daily_mu = rng.normal(0.0008, 0.0006, n)
    loadings = rng.normal(0, 1, (n, 3))
    daily_cov = loadings @ loadings.T * 1e-5 + np.eye(n) * 1e-4
    daily_cov = (daily_cov + daily_cov.T) / 2
    return PortfolioProblem(
        expected_returns=daily_mu * TRADING_DAYS,
        covariance=daily_cov * TRADING_DAYS,
        symbols=[f"A{i:02d}" for i in range(n)],
        constraints=Constraints(
            max_positions=cap, max_weight=max_weight, risk_aversion=risk_aversion
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, default=None, help="also write results here")
    args = parser.parse_args()

    cases = [(12, 4), (16, 4), (16, 6), (20, 5)]
    seeds = [1, 2, 3, 4, 5]

    print("Gap to brute-force optimum (lower is better; 0.000% = optimal)\n")
    header = (
        f"{'universe':>9} {'cap':>4} {'subsets':>9}  "
        + "  ".join(f"{cls.method:>16}" for cls in ALL_OPTIMIZERS)
    )
    print(header)
    print("-" * len(header))

    totals: dict[str, list[float]] = {cls.method: [] for cls in ALL_OPTIMIZERS}
    optimal_hits: dict[str, int] = {cls.method: 0 for cls in ALL_OPTIMIZERS}
    per_problem: list[dict[str, object]] = []

    for n, cap in cases:
        for seed in seeds:
            problem = synthetic_problem(n, seed, cap)
            truth = exhaustive_best(problem)
            cells = []
            record: dict[str, object] = {"n": n, "cap": cap, "seed": seed, "gaps": {}}
            for cls in ALL_OPTIMIZERS:
                result = cls().solve(problem)
                gap = truth.gap_to(result.objective_value)
                totals[cls.method].append(gap)
                record["gaps"][cls.method] = gap  # type: ignore[index]
                if gap < 1e-9:
                    optimal_hits[cls.method] += 1
                cells.append(f"{gap * 100:>15.3f}%")
            per_problem.append(record)
            if seed == seeds[0]:
                label = f"{n:>9} {cap:>4} {subset_count(n, cap):>9,}"
            else:
                label = f"{'':>9} {'':>4} {'':>9}"
            print(f"{label}  " + "  ".join(cells))
        print()

    runs = len(cases) * len(seeds)
    print("=" * len(header))
    print(f"{'MEAN GAP':>24}  " + "  ".join(
        f"{np.mean(totals[cls.method]) * 100:>15.3f}%" for cls in ALL_OPTIMIZERS
    ))
    print(f"{'WORST GAP':>24}  " + "  ".join(
        f"{np.max(totals[cls.method]) * 100:>15.3f}%" for cls in ALL_OPTIMIZERS
    ))
    print(f"{'OPTIMAL HIT RATE':>24}  " + "  ".join(
        f"{optimal_hits[cls.method] / runs * 100:>15.1f}%" for cls in ALL_OPTIMIZERS
    ))

    if args.json is not None:
        payload = {
            "problems": runs,
            "cases": [{"n": n, "cap": cap} for n, cap in cases],
            "seeds": seeds,
            "methods": {
                cls.method: {
                    "mean_gap": float(np.mean(totals[cls.method])),
                    "worst_gap": float(np.max(totals[cls.method])),
                    "optimal_hit_rate": optimal_hits[cls.method] / runs,
                }
                for cls in ALL_OPTIMIZERS
            },
            "per_problem": per_problem,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()

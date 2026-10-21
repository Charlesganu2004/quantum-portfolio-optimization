"""What does QAOA actually need to be competitive? Measure, don't guess.

Sweeps circuit depth and penalty strength across six problems (the `CASES`
list below; seeds 1 to 6) and reports the configuration with the best worst-case
gap. These tuning cases are not the cases `tests/test_qaoa.py` uses (holdout
seeds 101 to 110): `scripts/validate_qaoa_holdout.py` scores the chosen
configuration on those, which is the check that matters.
"""

from __future__ import annotations

import itertools

import numpy as np

from quantlab.evaluation.ground_truth import exhaustive_best
from quantlab.optimizers import Constraints, PortfolioProblem, QaoaOptimizer

TRADING_DAYS = 252
CASES = [(8, 4, 1), (8, 4, 2), (8, 3, 3), (10, 3, 4), (10, 4, 5), (12, 4, 6)]


def problem(n: int, seed: int, cap: int) -> PortfolioProblem:
    rng = np.random.default_rng(seed)
    daily_mu = rng.normal(0.0008, 0.0006, n)
    loadings = rng.normal(0, 1, (n, 3))
    daily_cov = loadings @ loadings.T * 1e-5 + np.eye(n) * 1e-4
    return PortfolioProblem(
        expected_returns=daily_mu * TRADING_DAYS,
        covariance=(daily_cov + daily_cov.T) / 2 * TRADING_DAYS,
        symbols=[f"A{i:02d}" for i in range(n)],
        constraints=Constraints(max_positions=cap, max_weight=0.4, risk_aversion=25.0),
    )


def main() -> None:
    depths = (1, 2, 3)
    penalties = (1.1, 5.0, 20.0)
    iterations = 250

    header = (
        f"{'reps':>5} {'penalty':>8} {'mean gap':>10} {'worst gap':>11} "
        f"{'feas.mass':>10} {'mean ms':>9}"
    )
    print(f"QAOA configuration sweep over {len(CASES)} cases, maxiter={iterations}\n")
    print(header)
    print("-" * len(header))

    best = None
    for reps, penalty in itertools.product(depths, penalties):
        gaps, masses, times = [], [], []
        for n, cap, seed in CASES:
            p = problem(n, seed, cap)
            truth = exhaustive_best(p)
            optimizer = QaoaOptimizer(
                reps=reps, max_iterations=iterations, penalty_multiplier=penalty
            )
            result = optimizer.solve(p)
            gaps.append(truth.gap_to(result.objective_value))
            masses.append(result.diagnostics.get("feasible_probability_mass", np.nan))
            times.append(result.runtime_seconds * 1000)

        mean_gap, worst_gap = float(np.mean(gaps)), float(np.max(gaps))
        print(
            f"{reps:>5} {penalty:>8.1f} {mean_gap * 100:>9.3f}% "
            f"{worst_gap * 100:>10.3f}% {np.mean(masses):>10.4f} {np.mean(times):>9.1f}"
        )
        if best is None or worst_gap < best[0]:
            best = (worst_gap, reps, penalty, mean_gap)

    assert best is not None
    worst, reps, penalty, mean_gap = best
    print(
        f"\nBest by worst-case: reps={reps} penalty={penalty} "
        f"(mean {mean_gap:.3%}, worst {worst:.3%})"
    )


if __name__ == "__main__":
    main()

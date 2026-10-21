"""Divergence check: do the methods differ, and where does the cap actually bind?

If every method returns the same portfolio, the Project II comparison has nothing
to report. Two things have to hold for the experiment to be meaningful:

1. Risk aversion must be scaled so the risk term competes with the return term.
   Inputs are annualized (x252) so lambda sits in the conventional 1-10 range.
2. The unconstrained solution must hold MORE names than the cardinality cap,
   otherwise the cap is inert and every method trivially agrees.

Run this after adding any optimizer.
"""

from __future__ import annotations

import numpy as np

from quantlab.optimizers import ALL_OPTIMIZERS, Constraints, PortfolioProblem

TRADING_DAYS = 252


def synthetic_problem(
    n: int, seed: int, cap: int | None, risk_aversion: float, max_weight: float = 0.4
) -> PortfolioProblem:
    """Annualized synthetic universe with a genuine factor structure."""
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


def sweep_risk_aversion(n: int = 32) -> None:
    """Find the lambda range where the solution is diversified enough to cap."""
    print("Risk-aversion sweep (uncapped, n=32) -- looking for a regime where the")
    print("solution holds enough names that a cardinality cap would bite.\n")
    print(f"{'lambda':>8} {'positions':>10} {'return':>9} {'risk':>9} {'max wt':>8}")
    print("-" * 48)
    for lam in (1, 2, 5, 10, 25, 50, 100):
        problem = synthetic_problem(n, seed=42, cap=None, risk_aversion=float(lam))
        result = ALL_OPTIMIZERS[0]().solve(problem)
        print(
            f"{lam:>8} {result.n_positions:>10} {result.expected_return:>9.4f} "
            f"{result.expected_risk:>9.4f} {result.weights.max():>8.3f}"
        )
    print()


def compare(n: int = 32, risk_aversion: float = 25.0) -> None:
    header = (
        f"{'cap':>5} {'method':<15} {'objective':>13} "
        f"{'return':>9} {'risk':>9} {'pos':>4} {'ms':>8}"
    )
    print(f"Method comparison at lambda={risk_aversion}, n={n}\n")
    print(header)
    print("-" * len(header))

    for cap in (None, 8, 4):
        results = []
        for cls in ALL_OPTIMIZERS:
            result = cls().solve(synthetic_problem(n, 42, cap, risk_aversion))
            results.append(result)
            print(
                f"{str(cap):>5} {result.method:<15} {result.objective_value:>13.6f} "
                f"{result.expected_return:>9.4f} {result.expected_risk:>9.4f} "
                f"{result.n_positions:>4} {result.runtime_seconds * 1000:>8.1f}"
            )

        objectives = [r.objective_value for r in results]
        identical = all(
            np.allclose(results[0].weights, r.weights, atol=1e-6) for r in results
        )
        best = max(objectives)
        worst = min(objectives)
        gap = (best - worst) / abs(best) * 100 if best else 0.0
        verdict = "IDENTICAL" if identical else f"differ, {gap:.3f}% objective gap"
        print(f"{'':>5} -> {verdict}\n")


if __name__ == "__main__":
    sweep_risk_aversion()
    compare()

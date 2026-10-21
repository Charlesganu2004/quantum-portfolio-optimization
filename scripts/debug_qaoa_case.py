"""Inspect a single QAOA case in detail.

Prints the percentage gap beside the absolute shortfall. The percentage divides
by |optimum|, so it explodes when the optimum is near zero (the scalability
sweep's n=8, repeat 2 problem has an optimum of -0.002158 and a QAOA gap above
5000% for an absolute shortfall of about 0.117). The shortfall is the number to
trust in that situation.

    python scripts/debug_qaoa_case.py [--n 8] [--cap 3] [--seed 3]

The default case (8 assets, cap 3, seed 3) is one of the six tuning cases, and
the shipped configuration reaches its optimum.
"""

from __future__ import annotations

import argparse

import numpy as np

from quantlab.evaluation.ground_truth import exhaustive_best
from quantlab.optimizers import Constraints, PortfolioProblem, QaoaOptimizer, QuboOptimizer
from quantlab.optimizers._qp import objective_value, solve_weights

TRADING_DAYS = 252


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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=8, help="number of assets")
    parser.add_argument("--cap", type=int, default=3, help="position cap")
    parser.add_argument("--seed", type=int, default=3, help="problem seed")
    args = parser.parse_args()
    n, cap, seed = args.n, args.cap, args.seed
    p = problem(n, seed, cap)
    c = p.constraints

    truth = exhaustive_best(p)
    qaoa = QaoaOptimizer().solve(p)
    qubo = QuboOptimizer().solve(p)

    print(f"case: n={n} cap={cap} seed={seed}\n")
    print(f"optimal objective : {truth.objective_value: .8f}  subset {truth.selected}")
    print(f"qubo objective    : {qubo.objective_value: .8f}")
    print(f"qaoa objective    : {qaoa.objective_value: .8f}")
    print(f"reported gap      : {truth.gap_to(qaoa.objective_value):.2%}")
    print(f"absolute shortfall: {truth.shortfall_to(qaoa.objective_value): .8f}")

    # Is the denominator simply tiny?
    print(f"\n|optimal| = {abs(truth.objective_value):.8f}")
    print("If |optimal| is near zero, a percentage gap is unstable and the")
    print("absolute shortfall is the meaningful number.\n")

    # Best achievable if readout were perfect over feasible states.
    matrix, _ = QuboOptimizer()._build_qubo(p, cap)
    best_surrogate, best_true = None, -np.inf
    from itertools import combinations

    surrogate_ranked = []
    for subset in combinations(range(n), cap):
        idx = np.asarray(subset)
        state = np.zeros(n)
        state[idx] = 1.0
        energy = float(state @ (matrix @ state))
        sub = solve_weights(
            p.expected_returns[idx],
            p.covariance[np.ix_(idx, idx)],
            c.risk_aversion,
            c.min_weight,
            c.max_weight,
            c.fully_invested,
        )
        weights = np.zeros(n)
        weights[idx] = sub
        value = objective_value(
            weights, p.expected_returns, p.covariance, c.risk_aversion
        )
        surrogate_ranked.append((energy, value, subset))
        if value > best_true:
            best_true = value

    surrogate_ranked.sort()
    print("feasible subsets ranked by QUBO energy (best 5):")
    for energy, value, subset in surrogate_ranked[:5]:
        print(f"  energy {energy: .6f}  true objective {value: .8f}  {subset}")
    print(f"\nlowest-energy subset true objective : {surrogate_ranked[0][1]: .8f}")
    print(f"true optimum                        : {best_true: .8f}")
    surrogate_gap = (best_true - surrogate_ranked[0][1]) / abs(best_true)
    print(f"=> surrogate (model) gap            : {surrogate_gap:.2%}")


if __name__ == "__main__":
    main()

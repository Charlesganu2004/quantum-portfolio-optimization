"""Locate where the QUBO method loses to ground truth.

Two independent failure modes produce the same bad number, and they need
opposite fixes:

  A. The annealer fails to find the lowest-energy state  -> fix the search
     (more sweeps, better schedule, penalty rescale).
  B. The annealer finds it, but the lowest-energy state is not the best
     portfolio -> fix the *model*. The QUBO assumes equal weights w = x/K, while
     the reported objective uses QP-optimal weights. If those two disagree on
     which subset is best, no amount of annealing effort helps.

This enumerates every subset, scores it both ways, and reports which mode is
responsible.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

from quantlab.evaluation.ground_truth import exhaustive_best
from quantlab.optimizers import Constraints, PortfolioProblem, QuboOptimizer
from quantlab.optimizers._qp import objective_value, solve_weights

TRADING_DAYS = 252


def synthetic_problem(n: int, seed: int, cap: int) -> PortfolioProblem:
    rng = np.random.default_rng(seed)
    daily_mu = rng.normal(0.0008, 0.0006, n)
    loadings = rng.normal(0, 1, (n, 3))
    daily_cov = loadings @ loadings.T * 1e-5 + np.eye(n) * 1e-4
    daily_cov = (daily_cov + daily_cov.T) / 2
    return PortfolioProblem(
        expected_returns=daily_mu * TRADING_DAYS,
        covariance=daily_cov * TRADING_DAYS,
        symbols=[f"A{i:02d}" for i in range(n)],
        constraints=Constraints(max_positions=cap, max_weight=0.4, risk_aversion=25.0),
    )


def analyse(n: int, cap: int, seed: int) -> dict[str, float]:
    problem = synthetic_problem(n, seed, cap)
    optimizer = QuboOptimizer()
    matrix, _penalty = optimizer._build_qubo(problem, cap)
    c = problem.constraints

    energies: list[float] = []
    objectives: list[float] = []
    subsets: list[tuple[int, ...]] = []

    for subset in combinations(range(n), cap):
        idx = np.asarray(subset, dtype=int)
        state = np.zeros(n)
        state[idx] = 1.0
        energies.append(float(state @ (matrix @ state)))

        sub = solve_weights(
            problem.expected_returns[idx],
            problem.covariance[np.ix_(idx, idx)],
            c.risk_aversion,
            c.min_weight,
            c.max_weight,
            c.fully_invested,
        )
        weights = np.zeros(n)
        weights[idx] = sub
        objectives.append(
            objective_value(
                weights, problem.expected_returns, problem.covariance, c.risk_aversion
            )
        )
        subsets.append(subset)

    energies_arr = np.asarray(energies)
    objectives_arr = np.asarray(objectives)

    lowest_energy_at = int(np.argmin(energies_arr))
    best_objective_at = int(np.argmax(objectives_arr))

    truth = exhaustive_best(problem)
    annealed = optimizer.solve(problem)
    annealed_subset = tuple(sorted(np.flatnonzero(annealed.weights > 1e-8).tolist()))

    # Objective reachable if the annealer perfectly minimized the surrogate.
    surrogate_ceiling = objectives_arr[lowest_energy_at]

    # Spearman: does low energy actually rank good portfolios highly?
    correlation = float(spearmanr(-energies_arr, objectives_arr).statistic)

    return {
        "model_loss": truth.gap_to(surrogate_ceiling),
        "search_loss": max(
            truth.gap_to(annealed.objective_value) - truth.gap_to(surrogate_ceiling),
            0.0,
        ),
        "total_gap": truth.gap_to(annealed.objective_value),
        "rank_correlation": correlation,
        "found_min_energy": float(
            annealed_subset == subsets[lowest_energy_at]
        ),
        "surrogate_agrees": float(lowest_energy_at == best_objective_at),
    }


def main() -> None:
    print("QUBO failure attribution -- where does the gap come from?\n")
    print("  model_loss  : gap left even if annealing perfectly minimized energy")
    print("  search_loss : extra gap from annealing not reaching minimum energy")
    print("  rank corr   : Spearman(-energy, true objective); 1.0 = perfect proxy\n")

    header = (
        f"{'n':>4} {'cap':>4} {'seed':>5} {'model_loss':>11} {'search_loss':>12} "
        f"{'total':>9} {'rank corr':>10} {'min E?':>7} {'agrees?':>8}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for n, cap in [(12, 4), (16, 4), (16, 6)]:
        for seed in (1, 2, 3):
            r = analyse(n, cap, seed)
            rows.append(r)
            print(
                f"{n:>4} {cap:>4} {seed:>5} {r['model_loss'] * 100:>10.3f}% "
                f"{r['search_loss'] * 100:>11.3f}% {r['total_gap'] * 100:>8.3f}% "
                f"{r['rank_correlation']:>10.3f} "
                f"{'yes' if r['found_min_energy'] else 'no':>7} "
                f"{'yes' if r['surrogate_agrees'] else 'no':>8}"
            )

    print("-" * len(header))
    model = np.mean([r["model_loss"] for r in rows]) * 100
    search = np.mean([r["search_loss"] for r in rows]) * 100
    corr = np.mean([r["rank_correlation"] for r in rows])
    print(f"{'MEAN':>15} {model:>10.3f}% {search:>11.3f}%")
    print(f"\nMean rank correlation: {corr:.3f}")
    verdict = "MODEL (equal-weight surrogate)" if model > search else "SEARCH (annealing)"
    print(f"Dominant failure mode: {verdict}")


if __name__ == "__main__":
    main()

"""Markowitz mean-variance optimization -- the classical baseline.

Without a cardinality cap this is a convex quadratic program and the solution is
the global optimum. Adding a cap makes it mixed-integer and NP-hard, which is
exactly the difficulty the QUBO formulation exists to attack. Here the cap is
handled by the standard two-stage relaxation: solve continuous, keep the largest
holdings, re-solve on that subset. That is a heuristic, it is reported as one in
`diagnostics`, and the gap it leaves is a headline result of the comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._qp import required_positions, solve_weights, InfeasibleProblem
from .base import Optimizer, PortfolioProblem


class MeanVarianceOptimizer(Optimizer):
    method = "mean_variance"

    def _solve(self, problem: PortfolioProblem) -> tuple[np.ndarray, dict[str, Any]]:
        c = problem.constraints
        weights = solve_weights(
            problem.expected_returns,
            problem.covariance,
            c.risk_aversion,
            c.min_weight,
            c.max_weight,
            c.fully_invested,
        )

        cap = c.max_positions
        held = int(np.count_nonzero(weights > 1e-8))
        if cap is None or held <= cap:
            return weights, {"relaxation_used": False, "continuous_positions": held}

        needed = required_positions(c.max_weight)
        if cap < needed:
            raise InfeasibleProblem(
                f"max_positions={cap} cannot reach a fully-invested portfolio "
                f"with max_weight={c.max_weight}; at least {needed} are required"
            )

        # Stage two: keep the largest holdings, re-optimize among them only.
        keep = np.argsort(weights)[::-1][:cap]
        keep.sort()
        restricted = solve_weights(
            problem.expected_returns[keep],
            problem.covariance[np.ix_(keep, keep)],
            c.risk_aversion,
            c.min_weight,
            c.max_weight,
            c.fully_invested,
        )
        final = np.zeros(problem.n_assets)
        final[keep] = restricted
        return final, {
            "relaxation_used": True,
            "continuous_positions": held,
            "selected": [problem.symbols[i] for i in keep],
        }

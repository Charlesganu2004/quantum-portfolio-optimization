"""Exhaustive optimum for small cardinality-constrained problems.

Every heuristic in this project claims to approximate the best K-asset portfolio.
On universes small enough to enumerate, that claim is checkable rather than
assumed: this module computes the true optimum by brute force so each method can
be scored by its *gap to optimal* instead of only against the other methods.

A method that merely beats its peers may still be far from the optimum, and all
of them can be wrong together. This is the only measurement in the comparison
that is not relative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from quantlab.optimizers._qp import objective_value, solve_weights, InfeasibleProblem
from quantlab.optimizers.base import PortfolioProblem

#: Refuse beyond this many subsets. C(26,6)=230,230 is accepted and C(27,6)=296,010
#: is refused; C(32,8) is 10.5 million and would never finish. Callers get an
#: error, not a hang.
MAX_SUBSETS = 250_000


class TooLargeToEnumerate(ValueError):
    """Raised when the subset count exceeds `MAX_SUBSETS`."""


@dataclass(frozen=True)
class GroundTruth:
    """The provably best allocation for a cardinality-constrained problem."""

    objective_value: float
    weights: np.ndarray
    selected: tuple[int, ...]
    subsets_evaluated: int
    subsets_infeasible: int

    def gap_to(self, objective: float) -> float:
        """Relative shortfall of `objective` against the optimum, as a fraction.

        Zero means optimal. Positive means worse. Slightly negative values within
        solver tolerance are clamped to zero rather than reported as beating a
        brute-force optimum, which would be nonsense.

        The denominator is ``|optimum|``, so the ratio is unstable when the
        optimum is close to zero: an optimum of -0.002 turns a shortfall of 0.12
        into a gap above 5000%. Read it beside `shortfall_to`, the same
        difference without the division.
        """
        denominator = abs(self.objective_value)
        if denominator < 1e-15:
            return 0.0
        gap = (self.objective_value - objective) / denominator
        return max(gap, 0.0)


    def shortfall_to(self, objective: float) -> float:
        """Absolute shortfall of `objective` against the optimum, in objective units.

        Zero means optimal. Unlike `gap_to` it does not depend on the size of the
        optimum, so it stays meaningful when the optimum is near zero or negative.
        """
        return max(self.objective_value - objective, 0.0)

def subset_count(n: int, k: int) -> int:
    return math.comb(n, k)


def exhaustive_best(problem: PortfolioProblem) -> GroundTruth:
    """Enumerate every admissible K-subset and return the best allocation.

    Raises:
        TooLargeToEnumerate: if the search space exceeds `MAX_SUBSETS`.
        ValueError: if the problem has no cardinality cap, making enumeration
            meaningless -- the continuous QP already solves that case exactly.
    """
    c = problem.constraints
    n = problem.n_assets
    if c.max_positions is None or c.max_positions >= n:
        raise ValueError(
            "exhaustive search needs a binding cardinality cap; without one the "
            "continuous QP is already the exact optimum"
        )

    k = c.max_positions
    total = subset_count(n, k)
    if total > MAX_SUBSETS:
        raise TooLargeToEnumerate(
            f"C({n},{k}) = {total:,} subsets exceeds the {MAX_SUBSETS:,} limit"
        )

    best_objective = -np.inf
    best_weights: np.ndarray | None = None
    best_subset: tuple[int, ...] = ()
    infeasible = 0

    for subset in combinations(range(n), k):
        idx = np.asarray(subset, dtype=int)
        try:
            sub_weights = solve_weights(
                problem.expected_returns[idx],
                problem.covariance[np.ix_(idx, idx)],
                c.risk_aversion,
                c.min_weight,
                c.max_weight,
                c.fully_invested,
            )
        except InfeasibleProblem:
            infeasible += 1
            continue

        weights = np.zeros(n)
        weights[idx] = sub_weights
        objective = objective_value(
            weights, problem.expected_returns, problem.covariance, c.risk_aversion
        )
        if objective > best_objective:
            best_objective = objective
            best_weights = weights
            best_subset = subset

    if best_weights is None:
        raise InfeasibleProblem(
            f"no feasible {k}-asset subset exists under the given constraints"
        )

    return GroundTruth(
        objective_value=best_objective,
        weights=best_weights,
        selected=best_subset,
        subsets_evaluated=total,
        subsets_infeasible=infeasible,
    )

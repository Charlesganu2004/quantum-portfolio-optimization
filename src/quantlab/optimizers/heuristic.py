"""Classical heuristic: backward elimination plus swap local search.

Mean-variance handles a cardinality cap by solving once and truncating to the
largest holdings. That throws away the interaction between the assets it keeps --
an asset that looked small in the full solution may be the best diversifier once
its neighbours are gone. This method instead removes one asset at a time and
re-optimizes after every removal, then attempts pairwise swaps to escape a local
optimum.

It is deterministic by construction: no random restarts, no seeds. Where the
problem is convex (no cardinality cap) it converges to the same optimum as
mean-variance, which is the correct result rather than a defect -- the methods
are expected to diverge only where the problem is combinatorial.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._qp import objective_value, required_positions, solve_weights, InfeasibleProblem
from .base import Optimizer, PortfolioProblem

#: Bounded so runtime stays comparable across universe sizes; every cap that
#: could hide a better solution is reported in `diagnostics`.
MAX_SWAP_ROUNDS = 3
SWAP_CANDIDATES = 5


class HeuristicOptimizer(Optimizer):
    method = "heuristic"

    def _solve(self, problem: PortfolioProblem) -> tuple[np.ndarray, dict[str, Any]]:
        c = problem.constraints
        n = problem.n_assets
        cap = c.max_positions if c.max_positions is not None else n

        needed = required_positions(c.max_weight) if c.fully_invested else 1
        if cap < needed:
            raise InfeasibleProblem(
                f"max_positions={cap} cannot reach a fully-invested portfolio "
                f"with max_weight={c.max_weight}; at least {needed} are required"
            )

        pool = list(range(n))
        weights = self._weights_for(pool, problem)
        eliminations = 0

        # Backward elimination: drop the least-committed holding, re-optimize.
        # Driven by the live count of non-zero holdings rather than the pool size,
        # so a solution that is already sparse enough costs no extra solves.
        while int(np.count_nonzero(weights > 1e-8)) > cap:
            local = weights[pool]
            drop_at = int(np.argmin(local))
            pool.pop(drop_at)
            weights = self._weights_for(pool, problem)
            eliminations += 1

        best_obj = self._objective(weights, problem)
        swaps = 0

        # Swap local search: trade the weakest holding for a promising outsider.
        for _ in range(MAX_SWAP_ROUNDS):
            if len(pool) >= n:
                break
            improved = False
            held = weights[pool]
            weakest_at = int(np.argmin(held))
            outsiders = [i for i in range(n) if i not in pool]
            for candidate in self._rank_outsiders(outsiders, problem)[:SWAP_CANDIDATES]:
                trial_pool = pool.copy()
                trial_pool[weakest_at] = candidate
                trial_pool.sort()
                try:
                    trial_weights = self._weights_for(trial_pool, problem)
                except InfeasibleProblem:
                    continue
                trial_obj = self._objective(trial_weights, problem)
                if trial_obj > best_obj + 1e-12:
                    pool, weights, best_obj = trial_pool, trial_weights, trial_obj
                    improved = True
                    swaps += 1
                    break
            if not improved:
                break

        return weights, {
            "eliminations": eliminations,
            "swaps_accepted": swaps,
            "swap_rounds_cap": MAX_SWAP_ROUNDS,
            "swap_candidates_cap": SWAP_CANDIDATES,
            "selected": [problem.symbols[i] for i in pool],
        }

    @staticmethod
    def _weights_for(pool: list[int], problem: PortfolioProblem) -> np.ndarray:
        """Optimal weights over `pool`, scattered back to full width."""
        c = problem.constraints
        idx = np.asarray(pool, dtype=int)
        sub = solve_weights(
            problem.expected_returns[idx],
            problem.covariance[np.ix_(idx, idx)],
            c.risk_aversion,
            c.min_weight,
            c.max_weight,
            c.fully_invested,
        )
        full = np.zeros(problem.n_assets)
        full[idx] = sub
        return full

    @staticmethod
    def _objective(weights: np.ndarray, problem: PortfolioProblem) -> float:
        return objective_value(
            weights,
            problem.expected_returns,
            problem.covariance,
            problem.constraints.risk_aversion,
        )

    @staticmethod
    def _rank_outsiders(
        outsiders: list[int], problem: PortfolioProblem
    ) -> list[int]:
        """Order excluded assets by standalone return-per-unit-risk.

        A cheap proxy -- it ignores correlation with current holdings -- used only
        to pick which swaps are worth the cost of a full re-solve. Ties break on
        index so the ordering is deterministic.
        """
        variances = np.diag(problem.covariance)
        scores = []
        for i in outsiders:
            risk = float(np.sqrt(max(variances[i], 1e-18)))
            scores.append((-float(problem.expected_returns[i]) / risk, i))
        scores.sort()
        return [i for _, i in scores]

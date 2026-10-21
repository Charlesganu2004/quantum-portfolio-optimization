"""Quantum-inspired QUBO selection solved by simulated annealing.

Formulation follows the standard cardinality-constrained portfolio QUBO (Lu et
al., "Quantum-Inspired Portfolio Optimization in the QUBO Framework"). Binary
``x_i`` marks asset i as held. Assuming equal weights ``w_i = x_i / K`` over the
K selected assets, maximizing ``mu'w - lambda w'Sigma w`` becomes minimization of

    E(x) = -(1/K) mu'x + (lambda/K^2) x'Sigma x + P (sum(x) - K)^2

Expanding the penalty and using ``x_i^2 = x_i`` for binaries gives a symmetric
QUBO matrix Q with

    Q_ii = -mu_i/K + (lambda/K^2) Sigma_ii + P(1 - 2K)
    Q_ij = (lambda/K^2) Sigma_ij + P                      (i != j)

so that ``E(x) = x'Qx`` up to the constant ``P K^2``.

The equal-weight assumption is a *selection* device only. Once annealing picks
the subset, continuous weights come from the same QP every other method uses --
so the comparison isolates selection strategy rather than confounding it with
weighting. This is the intended reading of the QUBO literature, where the binary
problem is the hard combinatorial core and weighting is a downstream convexity.

No quantum speedup is claimed. This runs on classical hardware and is benchmarked
against the classical methods on identical inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._qp import objective_value, required_positions, solve_weights, InfeasibleProblem
from .base import Optimizer, PortfolioProblem


class QuboOptimizer(Optimizer):
    """Cardinality-constrained selection by simulated annealing over a QUBO."""

    method = "qubo_annealing"

    def __init__(
        self,
        n_restarts: int = 3,
        n_sweeps: int = 120,
        seed: int = 20260803,
        initial_temp: float | None = None,
        final_temp_ratio: float = 1e-3,
    ) -> None:
        """
        Args:
            n_restarts: Independent annealing runs; the best energy wins. Guards
                against a single unlucky descent.
            n_sweeps: Cooling steps per restart. Each sweep proposes one flip per
                asset, so total proposals are ``n_restarts * n_sweeps * n``.
            seed: Fixed so results reproduce. An unreproducible benchmark is not
                a benchmark.
            initial_temp: Starting temperature. ``None`` calibrates it from the
                spread of single-flip energy changes, which keeps acceptance
                sensible across wildly different input scales.
            final_temp_ratio: Final temperature as a fraction of the initial.
        """
        if n_restarts < 1:
            raise ValueError(f"n_restarts must be >= 1, got {n_restarts}")
        if n_sweeps < 1:
            raise ValueError(f"n_sweeps must be >= 1, got {n_sweeps}")
        if not 0 < final_temp_ratio < 1:
            raise ValueError(
                f"final_temp_ratio must be in (0, 1), got {final_temp_ratio}"
            )
        self.n_restarts = n_restarts
        self.n_sweeps = n_sweeps
        self.seed = seed
        self.initial_temp = initial_temp
        self.final_temp_ratio = final_temp_ratio

    def _solve(self, problem: PortfolioProblem) -> tuple[np.ndarray, dict[str, Any]]:
        c = problem.constraints
        n = problem.n_assets
        target_k = c.max_positions if c.max_positions is not None else n

        needed = required_positions(c.max_weight) if c.fully_invested else 1
        if target_k < needed:
            raise InfeasibleProblem(
                f"max_positions={target_k} cannot reach a fully-invested portfolio "
                f"with max_weight={c.max_weight}; at least {needed} are required"
            )

        if target_k >= n:
            # Nothing to select: every asset is admissible. Annealing would burn
            # runtime to rediscover that, so go straight to the weighting QP.
            weights = self._weights_for(list(range(n)), problem)
            return weights, {
                "selection_skipped": True,
                "reason": "cardinality cap is not binding",
                "target_k": target_k,
            }

        matrix, penalty = self._build_qubo(problem, target_k)
        selection, energy, stats = self._anneal(matrix, target_k)
        selection = self._repair(selection, matrix, target_k)

        pool = sorted(int(i) for i in np.flatnonzero(selection))
        weights = self._weights_for(pool, problem)

        return weights, {
            "selection_skipped": False,
            "target_k": target_k,
            "penalty_weight": penalty,
            "best_energy": energy,
            "selected": [problem.symbols[i] for i in pool],
            "final_objective": objective_value(
                weights, problem.expected_returns, problem.covariance, c.risk_aversion
            ),
            **stats,
        }

    # ---------------------------------------------------------------- QUBO build

    def _build_qubo(
        self, problem: PortfolioProblem, target_k: int, penalty_multiplier: float = 1.1
    ) -> tuple[np.ndarray, float]:
        """Symmetric Q with ``E(x) = x'Qx`` (constant ``P K^2`` dropped).

        Args:
            penalty_multiplier: Scales the cardinality penalty above the largest
                single-flip objective change. The default 1.1 suits solvers whose
                moves preserve cardinality -- for those the penalty is provably
                inert, so anything above the margin is enough. Solvers that search
                the full unconstrained space (QAOA) need a substantially larger
                value, or the search concentrates on infeasible low-energy states.
        """
        mu = problem.expected_returns
        sigma = problem.covariance
        lam = problem.constraints.risk_aversion
        k = float(target_k)

        quadratic = (lam / (k * k)) * sigma
        linear = -mu / k + np.diag(quadratic)

        # The penalty must make any cardinality violation unprofitable, but no
        # larger: scaled off the objective's absolute magnitude it swamps the
        # landscape and annealing degenerates into picking a random feasible
        # subset. So bound it by the largest objective change a single flip can
        # produce -- the objective's *variation*, not its size.
        off_diagonal = quadratic.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        strongest_pairs = np.sort(np.abs(off_diagonal), axis=1)[:, ::-1]
        max_pair_sum = float(strongest_pairs[:, :target_k].sum(axis=1).max())
        max_flip_delta = float(np.abs(linear).max()) + 2.0 * max_pair_sum
        penalty = max(max_flip_delta * penalty_multiplier, 1e-12)

        matrix = quadratic + penalty
        np.fill_diagonal(matrix, linear + penalty * (1.0 - 2.0 * k))
        return (matrix + matrix.T) / 2.0, penalty

    # ------------------------------------------------------------------ annealing

    def _anneal(
        self, matrix: np.ndarray, target_k: int
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        """Metropolis annealing over cardinality-preserving swap moves.

        A move exchanges one held asset for one unheld asset, so the state never
        leaves the feasible set of exactly-K portfolios. Two consequences matter:

        * The penalty term contributes **exactly zero** to a swap's energy delta.
          (For any state with K ones, the penalty field is ``-PK`` on held assets
          and ``+PK`` on unheld ones; those cancel against the ``2P(1-2K) - 2P``
          diagonal contribution.) So the literature's penalty-based QUBO is
          preserved verbatim while the search sees the pure objective landscape --
          which is what single-bit flips destroyed by letting the penalty dominate.
        * Temperature calibrates against objective-scale deltas, so the schedule
          behaves identically whether inputs are daily decimals or annualized.

        Energy deltas use ``E(y) - E(x) = 2 d'Qx + d'Qd`` with ``d = e_j - e_i``,
        giving ``2(h_j - h_i) + Q_ii + Q_jj - 2 Q_ij``. The field ``h = Qx`` is
        maintained incrementally: O(1) to evaluate a move, O(n) to apply one.
        """
        n = matrix.shape[0]
        rng = np.random.default_rng(self.seed)
        diagonal = np.diag(matrix)
        proposals_per_sweep = max(n, 1)

        best_state: np.ndarray | None = None
        best_energy = np.inf
        total_accepted = 0

        for _restart in range(self.n_restarts):
            state = self._seed_state(rng, n, target_k)
            held = np.flatnonzero(state == 1.0)
            unheld = np.flatnonzero(state == 0.0)
            field = matrix @ state
            energy = float(state @ field)

            # Track the best state *visited*, not the state we happen to end on.
            # Without this the result depends on where the walk stops rather than
            # on the best solution found.
            run_best_state = state.copy()
            run_best_energy = energy

            temp = self.initial_temp or self._calibrate_temperature(
                matrix, diagonal, field, held, unheld, rng
            )
            cooling = self.final_temp_ratio ** (1.0 / max(self.n_sweeps - 1, 1))

            for _sweep in range(self.n_sweeps):
                for _ in range(proposals_per_sweep):
                    a = int(rng.integers(len(held)))
                    b = int(rng.integers(len(unheld)))
                    i, j = int(held[a]), int(unheld[b])

                    delta_e = (
                        2.0 * (field[j] - field[i])
                        + diagonal[i]
                        + diagonal[j]
                        - 2.0 * matrix[i, j]
                    )
                    if delta_e <= 0 or rng.random() < np.exp(
                        -delta_e / max(temp, 1e-18)
                    ):
                        state[i] = 0.0
                        state[j] = 1.0
                        field += matrix[:, j] - matrix[:, i]
                        energy += delta_e
                        held[a], unheld[b] = j, i
                        total_accepted += 1

                        if energy < run_best_energy:
                            run_best_energy = energy
                            run_best_state = state.copy()
                temp *= cooling

            # Recompute rather than trust an incremental sum over ~10^4 moves.
            run_best_energy = float(run_best_state @ (matrix @ run_best_state))
            if run_best_energy < best_energy:
                best_energy = run_best_energy
                best_state = run_best_state.copy()

        assert best_state is not None  # n_restarts >= 1 guarantees a solution
        return (
            best_state,
            best_energy,
            {
                "restarts": self.n_restarts,
                "sweeps_per_restart": self.n_sweeps,
                "moves_accepted": total_accepted,
                "move_type": "cardinality_preserving_swap",
            },
        )

    @staticmethod
    def _seed_state(
        rng: np.random.Generator, n: int, target_k: int
    ) -> np.ndarray:
        """Start from a random feasible-cardinality state."""
        state = np.zeros(n)
        state[rng.choice(n, size=target_k, replace=False)] = 1.0
        return state

    @staticmethod
    def _calibrate_temperature(
        matrix: np.ndarray,
        diagonal: np.ndarray,
        field: np.ndarray,
        held: np.ndarray,
        unheld: np.ndarray,
        rng: np.random.Generator,
        samples: int = 64,
    ) -> float:
        """Initial temperature from the typical magnitude of a swap's energy delta.

        Sampling the actual move distribution keeps the acceptance rate sensible
        regardless of input units, rather than hard-coding a temperature that
        happens to suit one dataset's scale.
        """
        deltas = []
        for _ in range(samples):
            i = int(held[rng.integers(len(held))])
            j = int(unheld[rng.integers(len(unheld))])
            deltas.append(
                abs(
                    2.0 * (field[j] - field[i])
                    + diagonal[i]
                    + diagonal[j]
                    - 2.0 * matrix[i, j]
                )
            )
        magnitude = float(np.mean(deltas)) if deltas else 0.0
        return max(magnitude, 1e-12)

    # -------------------------------------------------------------------- repair

    @staticmethod
    def _repair(
        state: np.ndarray, matrix: np.ndarray, target_k: int
    ) -> np.ndarray:
        """Force exact cardinality regardless of how well the penalty held.

        Feasibility is guaranteed here rather than hoped for from the penalty
        weight, so the base class never has to reject a QUBO solution.
        """
        state = state.copy()
        diagonal = np.diag(matrix)

        while int(state.sum()) > target_k:
            held = np.flatnonzero(state)
            field = matrix @ state
            # Removing asset i changes energy by -(Q_ii + 2(h_i - Q_ii)); drop
            # whichever removal helps most (or hurts least).
            deltas = -(diagonal[held] + 2.0 * (field[held] - diagonal[held]))
            state[held[int(np.argmin(deltas))]] = 0.0

        while int(state.sum()) < target_k:
            free = np.flatnonzero(state == 0)
            field = matrix @ state
            deltas = diagonal[free] + 2.0 * field[free]
            state[free[int(np.argmin(deltas))]] = 1.0

        return state

    # --------------------------------------------------------------------- shared

    @staticmethod
    def _weights_for(pool: list[int], problem: PortfolioProblem) -> np.ndarray:
        """Continuous weights over `pool` -- the same QP every method uses."""
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

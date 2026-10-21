"""Optional simulated-quantum path: QAOA over the QUBO objective.

This is the fourth method in the Project II comparison and the only one that
touches quantum software. It builds the QUBO with `QuboOptimizer._build_qubo`,
so the objective and the form of the cardinality penalty are the same as the
annealer's, then hands the selected subset to the *identical* weighting QP.

The penalty strength is not the same. QAOA searches the whole unconstrained
space and uses `penalty_multiplier=5.0`, while the annealer's swap moves keep
the cardinality fixed and use 1.1. The two matrices differ by that penalty term
(`tests/test_qaoa.py` checks exactly this), so a difference in result is
attributable to the solver and the penalty strength together, not to the solver
alone.

Three honest statements the report should carry:

1. **No speedup is claimed or possible here.** QAOA runs on a statevector
   simulator on classical hardware. Simulating N qubits costs O(2^N) memory and
   time, so this method is *exponentially slower* than the classical solvers and
   gets slower the moment the problem gets interesting. Measured runtime is
   reported as-is.
2. **It is capped at `MAX_QUBITS` assets.** Beyond that the run time is
   impractical: each circuit evaluation costs O(2^N) and a variational run needs
   hundreds of them. Memory is not the limit (a 20-qubit statevector is 16 MiB).
   The cap is a stated experimental limit, not a bug, and `QaoaNotApplicable` is
   raised rather than silently degrading.
3. **Real quantum hardware would not fix this** at present scale -- Project I's
   literature review already established that, and cloud QPU access latency is
   measured in tens of milliseconds before queueing.

The value of including it is scientific: it shows what the quantum-inspired
QUBO formulation costs when solved by an actual quantum algorithm rather than by
classical annealing, on identical inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._qp import required_positions, solve_weights, InfeasibleProblem
from .base import Optimizer, PortfolioProblem
from .qubo import QuboOptimizer

#: One simulated qubit per asset. A runtime limit, not a memory limit: at 14
#: qubits a run already takes seconds, and each added qubit doubles the cost of
#: every circuit evaluation.
MAX_QUBITS = 14


class QaoaNotApplicable(RuntimeError):
    """Raised when the problem exceeds what a simulator can handle."""


def qiskit_available() -> bool:
    """Whether the optional quantum extra is installed.

    Only `qiskit` itself is needed: the circuit is built with `QuantumCircuit`
    and simulated with `qiskit.quantum_info.Statevector`, and the variational
    loop uses SciPy.
    """
    try:
        import qiskit  # noqa: F401
    except ImportError:
        return False
    return True


class QaoaOptimizer(Optimizer):
    """Quantum Approximate Optimization Algorithm on a local simulator."""

    method = "qaoa_simulated"

    def __init__(
        self,
        reps: int = 2,
        max_iterations: int = 120,
        seed: int = 20260803,
        max_qubits: int = MAX_QUBITS,
        n_candidates: int = 32,
        penalty_multiplier: float = 5.0,
    ) -> None:
        """
        Args:
            reps: QAOA circuit depth (p). Higher p can improve solution quality
                at steeply rising simulation cost.
            max_iterations: Classical optimizer budget for the variational loop.
            seed: Fixed so the run reproduces.
            max_qubits: Refuse problems larger than this.
            n_candidates: How many of the highest-probability bitstrings to read
                out of the final distribution. Standard QAOA practice is to take
                the best of a modest sample rather than the single most likely
                state; reading all 2^n would be brute force wearing a costume.
            penalty_multiplier: Cardinality-penalty strength passed through to the
                QUBO build. QAOA searches the full unconstrained space, so it needs
                a far stronger penalty than the swap-move annealer, for which the
                penalty is provably inert.

        Defaults (reps=2, penalty_multiplier=5.0) come from a grid search
        (`scripts/tune_qaoa.py`) validated on held-out problems
        (`scripts/validate_qaoa_holdout.py`). Both scripts used
        `max_iterations=250`. The default here is 120, which is faster and is
        what the scalability sweep, the tests and `scripts/run_session.py` use;
        it has not been validated on the holdout set. Be aware of what the
        validation found: the configuration scored 0.000% mean gap on the six tuning cases
        and **53.256% on twelve holdout cases** -- the grid search overfit. These
        defaults are the best available, not good. Neighbouring grid points score
        13-62%, so QAOA's quality here is highly sensitive to configuration, which
        is itself a reportable property of the method rather than a bug in it.
        """
        if reps < 1:
            raise ValueError(f"reps must be >= 1, got {reps}")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if n_candidates < 1:
            raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
        if penalty_multiplier <= 0:
            raise ValueError(
                f"penalty_multiplier must be positive, got {penalty_multiplier}"
            )
        self.reps = reps
        self.max_iterations = max_iterations
        self.seed = seed
        self.max_qubits = max_qubits
        self.n_candidates = n_candidates
        self.penalty_multiplier = penalty_multiplier

    def _solve(self, problem: PortfolioProblem) -> tuple[np.ndarray, dict[str, Any]]:
        if not qiskit_available():
            raise QaoaNotApplicable(
                "qiskit is required for the QAOA path. "
                "Install with: pip install 'quantlab[quantum]'"
            )

        c = problem.constraints
        n = problem.n_assets
        target_k = c.max_positions if c.max_positions is not None else n

        needed = required_positions(c.max_weight) if c.fully_invested else 1
        if target_k < needed:
            raise InfeasibleProblem(
                f"max_positions={target_k} cannot reach a fully-invested portfolio "
                f"with max_weight={c.max_weight}; at least {needed} are required"
            )

        if n > self.max_qubits:
            raise QaoaNotApplicable(
                f"{n} assets needs {n} simulated qubits, above the {self.max_qubits} "
                f"limit. Simulation cost is O(2^n); this is a stated experimental "
                f"boundary, not a solver failure."
            )

        if target_k >= n:
            weights = self._weights_for(list(range(n)), problem)
            return weights, {
                "selection_skipped": True,
                "reason": "cardinality cap is not binding",
                "qubits": n,
            }

        # Same objective and penalty form as the classical annealer, with a
        # stronger penalty multiplier (see the module docstring).
        helper = QuboOptimizer()
        matrix, penalty = helper._build_qubo(
            problem, target_k, penalty_multiplier=self.penalty_multiplier
        )
        selection, stats = self._run_qaoa(matrix, n, target_k)
        selection = helper._repair(selection, matrix, target_k)

        pool = sorted(int(i) for i in np.flatnonzero(selection))
        weights = self._weights_for(pool, problem)

        return weights, {
            "selection_skipped": False,
            "target_k": target_k,
            "penalty_weight": penalty,
            "qubits": n,
            "reps": self.reps,
            "selected": [problem.symbols[i] for i in pool],
            "simulated_quantum": True,
            "speedup_claimed": False,
            **stats,
        }

    @staticmethod
    def _to_ising(matrix: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, float]:
        """Map the QUBO to an Ising Hamiltonian.

        Substituting ``x_i = (1 - z_i)/2`` with ``z_i`` in {-1,+1} into
        ``E(x) = x'Qx`` gives ``E = c + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j``
        with

            h_i  = -Q_ii/2 - sum_{j!=i} Q_ij/2
            J_ij = Q_ij/2

        Written out rather than delegated so the encoding is inspectable in the
        methods section -- this substitution is where a sign error would silently
        make QAOA optimize the wrong problem.
        """
        diagonal = np.diag(matrix)
        off_diagonal = matrix - np.diag(diagonal)
        fields = -diagonal / 2.0 - off_diagonal.sum(axis=1) / 2.0
        couplings = np.triu(off_diagonal, k=1) / 2.0
        constant = float(diagonal.sum() / 2.0 + np.triu(off_diagonal, k=1).sum() / 2.0)
        return fields, couplings, constant

    @staticmethod
    def _all_energies(matrix: np.ndarray, n: int) -> np.ndarray:
        """Energy of every one of the 2^n basis states.

        Feasible only because `max_qubits` bounds n -- which is precisely the
        point being demonstrated: at any size a simulator can handle, the problem
        is already classically enumerable.

        Bit i of the state index is qubit i (Qiskit little-endian ordering).
        """
        size = 1 << n
        bits = ((np.arange(size)[:, None] >> np.arange(n)) & 1).astype(float)
        return np.einsum("si,ij,sj->s", bits, matrix, bits)

    def _build_circuit(
        self,
        params: np.ndarray,
        fields: np.ndarray,
        couplings: np.ndarray,
        n: int,
    ):
        """QAOA ansatz: uniform superposition, then `reps` cost/mixer layers."""
        from qiskit import QuantumCircuit

        circuit = QuantumCircuit(n)
        circuit.h(range(n))

        for layer in range(self.reps):
            gamma = float(params[layer])
            beta = float(params[self.reps + layer])

            # exp(-i * gamma * H_cost): RZ for fields, RZZ for couplings.
            for i in range(n):
                if abs(fields[i]) > 1e-15:
                    circuit.rz(2.0 * gamma * float(fields[i]), i)
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(couplings[i, j]) > 1e-15:
                        circuit.rzz(2.0 * gamma * float(couplings[i, j]), i, j)

            # exp(-i * beta * H_mixer) with H_mixer = sum_i X_i.
            circuit.rx(2.0 * beta, range(n))

        return circuit

    def _run_qaoa(
        self, matrix: np.ndarray, n: int, target_k: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Solve ``min x'Qx`` over binary x with QAOA on a statevector simulator.

        The circuit is built directly rather than through `qiskit_algorithms`,
        whose QAOA ansatz is incompatible with the Sampler primitive in current
        Qiskit. Building it here also makes the Ising encoding and the ansatz
        structure explicit for the write-up.

        **Readout is post-selected on the feasible subspace** -- only bitstrings
        with exactly `target_k` ones are considered. QAOA optimizes over all 2^n
        states and relies on the penalty term to discourage infeasible ones, but
        the penalty is calibrated for the classical annealer's swap moves, where
        it is provably inert. Without post-selection QAOA returns low-energy
        *infeasible* states and the downstream repair rebuilds from a bad start.
        Post-selecting on the constraint is the standard treatment for
        cardinality-constrained QAOA and is honest about what the algorithm
        produces: a distribution, from which the feasible portion is read.
        """
        from qiskit.quantum_info import Statevector
        from scipy.optimize import minimize

        fields, couplings, _constant = self._to_ising(matrix, n)
        energies = self._all_energies(matrix, n)

        popcount = np.array(
            [bin(index).count("1") for index in range(1 << n)], dtype=int
        )
        feasible = np.flatnonzero(popcount == target_k)
        if feasible.size == 0:
            raise QaoaNotApplicable(
                f"no {target_k}-of-{n} basis state exists to post-select on"
            )

        evaluations = 0

        def expectation(params: np.ndarray) -> float:
            """Exact <psi|H|psi> from the statevector -- no shot noise."""
            nonlocal evaluations
            evaluations += 1
            probabilities = np.abs(
                Statevector(self._build_circuit(params, fields, couplings, n)).data
            ) ** 2
            return float(probabilities @ energies)

        # Deterministic start; a fixed seed keeps the reported benchmark reproducible.
        rng = np.random.default_rng(self.seed)
        initial = np.concatenate(
            [rng.uniform(0, np.pi, self.reps), rng.uniform(0, np.pi / 2, self.reps)]
        )
        outcome = minimize(
            expectation,
            initial,
            method="COBYLA",
            options={"maxiter": self.max_iterations},
        )

        final = np.abs(
            Statevector(self._build_circuit(outcome.x, fields, couplings, n)).data
        ) ** 2

        # Read the answer out of QAOA's own output distribution, restricted to
        # feasible states -- deliberately not argmin over all `energies`, which
        # would be brute force reported as QAOA.
        feasible_probability = final[feasible]
        ranked = feasible[np.argsort(feasible_probability)[::-1]]
        candidates = ranked[: self.n_candidates]
        best_index = int(candidates[int(np.argmin(energies[candidates]))])

        state = np.array([(best_index >> i) & 1 for i in range(n)], dtype=float)
        return state, {
            "qaoa_expectation": float(outcome.fun),
            "sampled_energy": float(energies[best_index]),
            "circuit_evaluations": evaluations,
            "classical_optimizer": "COBYLA",
            "max_iterations": self.max_iterations,
            "candidates_read": int(min(self.n_candidates, feasible.size)),
            "feasible_states": int(feasible.size),
            "feasible_probability_mass": float(feasible_probability.sum()),
            "post_selected": True,
        }

    @staticmethod
    def _weights_for(pool: list[int], problem: PortfolioProblem) -> np.ndarray:
        """The same weighting QP every other method uses."""
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

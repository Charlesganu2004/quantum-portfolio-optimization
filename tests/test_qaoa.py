"""Simulated-quantum path: contract, limits, and parity with the classical solver."""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.evaluation.ground_truth import exhaustive_best
from quantlab.optimizers import (
    Constraints,
    PortfolioProblem,
    QaoaNotApplicable,
    QaoaOptimizer,
    QuboOptimizer,
    qiskit_available,
)

pytestmark = pytest.mark.skipif(
    not qiskit_available(), reason="optional quantum extra not installed"
)

TRADING_DAYS = 252


def problem(n: int, seed: int, cap: int | None) -> PortfolioProblem:
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


class TestContract:
    def test_respects_the_cardinality_cap(self) -> None:
        result = QaoaOptimizer().solve(problem(8, seed=1, cap=3))
        assert result.n_positions <= 3

    def test_weights_sum_to_one(self) -> None:
        result = QaoaOptimizer().solve(problem(8, seed=1, cap=3))
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-5)

    def test_is_long_only(self) -> None:
        result = QaoaOptimizer().solve(problem(8, seed=2, cap=3))
        assert (result.weights >= -1e-6).all()

    def test_is_deterministic(self) -> None:
        """Seeded, so the reported benchmark reproduces."""
        p = problem(8, seed=3, cap=3)
        first = QaoaOptimizer().solve(p).weights
        second = QaoaOptimizer().solve(p).weights
        np.testing.assert_allclose(first, second, atol=1e-9)

    def test_records_runtime(self) -> None:
        assert QaoaOptimizer().solve(problem(8, 1, 3)).runtime_seconds > 0


class TestLimits:
    def test_refuses_beyond_the_qubit_cap(self) -> None:
        """A stated experimental boundary, surfaced rather than degraded."""
        with pytest.raises(QaoaNotApplicable, match="simulated qubits"):
            QaoaOptimizer(max_qubits=8).solve(problem(12, seed=1, cap=4))

    def test_error_explains_the_exponential_cost(self) -> None:
        with pytest.raises(QaoaNotApplicable, match=r"O\(2\^n\)"):
            QaoaOptimizer(max_qubits=6).solve(problem(10, seed=1, cap=3))

    def test_skips_annealing_when_cap_is_not_binding(self) -> None:
        result = QaoaOptimizer().solve(problem(6, seed=1, cap=None))
        assert result.diagnostics["selection_skipped"] is True

    def test_rejects_invalid_depth(self) -> None:
        with pytest.raises(ValueError, match="reps"):
            QaoaOptimizer(reps=0)


class TestParity:
    def test_qubo_matrices_differ_only_in_the_penalty_term(self) -> None:
        """QAOA and the annealer share an objective but not a penalty strength.

        Each matrix is built the way its solver builds it: the annealer calls
        `_build_qubo` with the default multiplier, QAOA passes its own
        `penalty_multiplier`. The penalty enters as ``P * R`` with R all ones and
        a diagonal of ``1 - 2K``, so the two matrices must differ by exactly
        ``(P_qaoa - P_annealer) * R`` and agree once that term is removed.
        """
        n, k = 8, 3
        p = problem(n, seed=4, cap=k)
        annealer, penalty_annealer = QuboOptimizer()._build_qubo(p, k)
        quantum, penalty_qaoa = QuboOptimizer()._build_qubo(
            p, k, penalty_multiplier=QaoaOptimizer().penalty_multiplier
        )

        # The penalty strengths are genuinely different, so the matrices are too.
        assert penalty_qaoa > penalty_annealer
        assert not np.allclose(annealer, quantum)

        penalty_pattern = np.ones((n, n))
        np.fill_diagonal(penalty_pattern, 1.0 - 2.0 * k)
        np.testing.assert_allclose(
            quantum - annealer,
            (penalty_qaoa - penalty_annealer) * penalty_pattern,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            quantum - penalty_qaoa * penalty_pattern,
            annealer - penalty_annealer * penalty_pattern,
            atol=1e-12,
        )

    def test_solvers_report_the_penalty_they_actually_used(self) -> None:
        p = problem(8, seed=4, cap=3)
        _, expected_annealer = QuboOptimizer()._build_qubo(p, 3)
        _, expected_qaoa = QuboOptimizer()._build_qubo(
            p, 3, penalty_multiplier=QaoaOptimizer().penalty_multiplier
        )
        assert QuboOptimizer().solve(p).diagnostics["penalty_weight"] == pytest.approx(
            expected_annealer
        )
        assert QaoaOptimizer().solve(p).diagnostics["penalty_weight"] == pytest.approx(
            expected_qaoa
        )

    def test_never_claims_a_speedup(self) -> None:
        diagnostics = QaoaOptimizer().solve(problem(8, 1, 3)).diagnostics
        assert diagnostics["speedup_claimed"] is False
        assert diagnostics["simulated_quantum"] is True

    def test_is_slower_than_classical_annealing(self) -> None:
        """The expected, reportable result -- simulation costs O(2^n)."""
        p = problem(10, seed=5, cap=4)
        quantum = QaoaOptimizer().solve(p).runtime_seconds
        classical = QuboOptimizer().solve(p).runtime_seconds
        assert quantum > classical

    @pytest.mark.parametrize(
        ("n", "cap", "seed"),
        [(8, 3, 101), (9, 3, 103), (10, 4, 106), (11, 4, 108), (12, 4, 110)],
    )
    def test_produces_a_feasible_solution_without_diverging(
        self, n: int, cap: int, seed: int
    ) -> None:
        """A breakage guard, NOT a quality bar.

        The bound is deliberately loose because QAOA's measured quality on this
        problem is poor and highly variable: 53.3% mean gap and 257.6% worst
        across a 12-case holdout (`scripts/validate_qaoa_holdout.py`). A tight
        threshold here would only be met by tuning against the test set -- which
        is exactly the mistake this suite already caught once. A grid search
        found reps=2/penalty=5.0 scoring 0.000% on six tuning cases; on holdout
        the same configuration scored 53.256%.

        So this asserts only that the solver runs, stays feasible, and does not
        diverge catastrophically. Quality is *reported*, not asserted.
        """
        p = problem(n, seed=seed, cap=cap)
        truth = exhaustive_best(p)
        result = QaoaOptimizer().solve(p)
        gap = truth.gap_to(result.objective_value)
        assert result.n_positions <= cap
        assert np.isfinite(result.objective_value)
        assert gap < 5.0, (
            f"QAOA gap {gap:.1%} at n={n} cap={cap} seed={seed} exceeds even the "
            f"loose divergence guard -- the encoding or readout is likely broken"
        )

    def test_underperforms_classical_annealing(self) -> None:
        """The headline Project II finding, locked in as a regression.

        If this ever fails, either QAOA improved dramatically or the classical
        annealer regressed. Both need investigating before the report changes.
        """
        quantum_gaps, classical_gaps = [], []
        for n, cap, seed in [(8, 3, 101), (9, 3, 103), (11, 4, 108), (12, 4, 110)]:
            p = problem(n, seed=seed, cap=cap)
            truth = exhaustive_best(p)
            quantum_gaps.append(truth.gap_to(QaoaOptimizer().solve(p).objective_value))
            classical_gaps.append(truth.gap_to(QuboOptimizer().solve(p).objective_value))
        assert np.mean(quantum_gaps) > np.mean(classical_gaps)

    def test_readout_is_post_selected_on_feasible_states(self) -> None:
        diagnostics = QaoaOptimizer().solve(problem(8, 1, 3)).diagnostics
        assert diagnostics["post_selected"] is True
        assert diagnostics["feasible_states"] == 56  # C(8,3)

    def test_never_returns_an_infeasible_selection(self) -> None:
        """Post-selection must make the repair step a no-op, not a rescue."""
        for seed in (1, 2, 3, 4, 5):
            result = QaoaOptimizer().solve(problem(9, seed=seed, cap=3))
            assert result.n_positions <= 3

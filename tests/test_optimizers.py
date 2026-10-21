"""Tests every optimizer must pass.

`ALL_OPTIMIZERS` is parametrized across the suite, so a new method is held to the
same contract as the existing ones the moment it is registered. That contract is
what makes the runtime and quality comparison in the report defensible.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.optimizers import ALL_OPTIMIZERS
from quantlab.optimizers.base import Constraints, Optimizer, PortfolioProblem


def make_problem(
    n: int = 8,
    seed: int = 0,
    constraints: Constraints | None = None,
    transaction_cost_bps: float = 0.0,
) -> PortfolioProblem:
    """Deterministic well-conditioned problem with a genuine risk/return spread."""
    rng = np.random.default_rng(seed)
    expected_returns = rng.normal(0.0008, 0.0006, size=n)
    # Factor structure + diagonal floor keeps the covariance positive definite.
    loadings = rng.normal(0, 1, size=(n, 3))
    cov = loadings @ loadings.T * 1e-5 + np.eye(n) * 1e-4
    cov = (cov + cov.T) / 2
    return PortfolioProblem(
        expected_returns=expected_returns,
        covariance=cov,
        symbols=[f"A{i:02d}" for i in range(n)],
        constraints=constraints or Constraints(),
        transaction_cost_bps=transaction_cost_bps,
    )


@pytest.fixture(params=ALL_OPTIMIZERS, ids=lambda o: o.method)
def optimizer(request: pytest.FixtureRequest) -> Optimizer:
    return request.param()


class TestContract:
    """Behaviour every method must exhibit."""

    def test_weights_sum_to_one_when_fully_invested(self, optimizer: Optimizer) -> None:
        result = optimizer.solve(make_problem())
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-5)

    def test_long_only_by_default(self, optimizer: Optimizer) -> None:
        result = optimizer.solve(make_problem())
        assert (result.weights >= -1e-6).all()

    def test_respects_max_weight(self, optimizer: Optimizer) -> None:
        problem = make_problem(constraints=Constraints(max_weight=0.25))
        result = optimizer.solve(problem)
        assert result.weights.max() <= 0.25 + 1e-6

    def test_respects_cardinality_cap(self, optimizer: Optimizer) -> None:
        problem = make_problem(n=16, constraints=Constraints(max_positions=4))
        result = optimizer.solve(problem)
        assert result.n_positions <= 4

    def test_records_positive_runtime(self, optimizer: Optimizer) -> None:
        result = optimizer.solve(make_problem())
        assert result.runtime_seconds > 0

    def test_reports_own_method_name(self, optimizer: Optimizer) -> None:
        result = optimizer.solve(make_problem())
        assert result.method == optimizer.method
        assert result.method != "unnamed"

    def test_deterministic_across_runs(self, optimizer: Optimizer) -> None:
        """Same input twice must give the same weights.

        Stochastic methods must seed themselves; an unreproducible benchmark is
        not a benchmark.
        """
        problem = make_problem(seed=7)
        first = optimizer.solve(problem).weights
        second = optimizer.solve(problem).weights
        np.testing.assert_allclose(first, second, atol=1e-9)

    def test_risk_and_return_are_consistent(self, optimizer: Optimizer) -> None:
        problem = make_problem()
        result = optimizer.solve(problem)
        expected_var = result.weights @ problem.covariance @ result.weights
        assert result.expected_risk == pytest.approx(np.sqrt(expected_var), rel=1e-6)
        assert result.expected_return == pytest.approx(
            problem.expected_returns @ result.weights, rel=1e-6
        )

    def test_transaction_cost_reduces_expected_return(
        self, optimizer: Optimizer
    ) -> None:
        free = optimizer.solve(make_problem(transaction_cost_bps=0.0))
        costly = optimizer.solve(make_problem(transaction_cost_bps=50.0))
        assert costly.diagnostics["transaction_cost"] > 0
        assert costly.expected_return < free.expected_return

    @pytest.mark.parametrize("n", [4, 8, 16, 32])
    def test_scales_across_universe_sizes(self, optimizer: Optimizer, n: int) -> None:
        result = optimizer.solve(make_problem(n=n))
        assert result.weights.shape == (n,)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-5)


class TestQuality:
    """Solutions must be economically sensible, not merely feasible."""

    def test_beats_equal_weight_on_objective(self, optimizer: Optimizer) -> None:
        """An optimizer that cannot beat 1/N is not doing its job."""
        problem = make_problem(n=12, seed=3)
        result = optimizer.solve(problem)

        n = problem.n_assets
        equal = np.full(n, 1.0 / n)
        equal_obj = float(
            problem.expected_returns @ equal
            - problem.constraints.risk_aversion * (equal @ problem.covariance @ equal)
        )
        assert result.objective_value >= equal_obj - 1e-9

    def test_higher_risk_aversion_lowers_risk(self, optimizer: Optimizer) -> None:
        timid = optimizer.solve(
            make_problem(n=12, seed=5, constraints=Constraints(risk_aversion=50.0))
        )
        bold = optimizer.solve(
            make_problem(n=12, seed=5, constraints=Constraints(risk_aversion=0.1))
        )
        assert timid.expected_risk <= bold.expected_risk + 1e-9


class TestValidation:
    """Bad input must be rejected at construction, not silently absorbed."""

    def test_rejects_asymmetric_covariance(self) -> None:
        cov = np.array([[1.0, 0.5], [0.2, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            PortfolioProblem(
                expected_returns=np.array([0.1, 0.1]),
                covariance=cov,
                symbols=["A", "B"],
            )

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            PortfolioProblem(
                expected_returns=np.array([0.1, 0.1, 0.1]),
                covariance=np.eye(2),
                symbols=["A", "B"],
            )

    def test_rejects_min_above_max_weight(self) -> None:
        with pytest.raises(ValueError, match="exceeds max_weight"):
            Constraints(min_weight=0.5, max_weight=0.2)

    def test_rejects_zero_max_positions(self) -> None:
        with pytest.raises(ValueError, match="max_positions"):
            Constraints(max_positions=0)

    def test_rejects_negative_transaction_cost(self) -> None:
        with pytest.raises(ValueError, match="transaction_cost_bps"):
            PortfolioProblem(
                expected_returns=np.array([0.1]),
                covariance=np.eye(1),
                symbols=["A"],
                transaction_cost_bps=-1.0,
            )


class TestBaseGuards:
    """The base class must catch a method that returns an invalid portfolio."""

    def test_rejects_weights_that_break_cardinality(self) -> None:
        class Cheater(Optimizer):
            method = "cheater"

            def _solve(self, problem: PortfolioProblem):
                return np.full(problem.n_assets, 1.0 / problem.n_assets), {}

        problem = make_problem(n=10, constraints=Constraints(max_positions=2))
        with pytest.raises(ValueError, match="exceeds max_positions"):
            Cheater().solve(problem)

    def test_rejects_weights_that_do_not_sum_to_one(self) -> None:
        class Cheater(Optimizer):
            method = "cheater"

            def _solve(self, problem: PortfolioProblem):
                return np.zeros(problem.n_assets), {}

        with pytest.raises(ValueError, match="sum to 1"):
            Cheater().solve(make_problem())

"""Shared optimizer interface.

Every optimization method in this project implements `Optimizer` and is measured
through the same `solve()` entry point, so runtime comparisons are fair: timing,
validation, and diagnostics are handled identically for all methods and cannot
drift between implementations.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Constraints:
    """Portfolio constraints applied by every optimizer.

    Attributes:
        max_weight: Upper bound on any single asset's weight.
        min_weight: Lower bound on any single asset's weight. Zero means long-only.
        max_positions: Cardinality cap -- at most this many non-zero holdings.
            ``None`` disables the cap. Project I found mean-variance and QUBO
            differ sharply here, so it is a first-class reported result.
        fully_invested: If True, weights must sum to 1. If False, they may sum to
            less, with the remainder held as cash.
        risk_aversion: Lambda in ``mu'w - lambda * w'Sigma w``. Higher is more
            risk-averse.
    """

    max_weight: float = 1.0
    min_weight: float = 0.0
    max_positions: int | None = None
    fully_invested: bool = True
    risk_aversion: float = 1.0

    def __post_init__(self) -> None:
        if self.max_weight <= 0:
            raise ValueError(f"max_weight must be positive, got {self.max_weight}")
        if self.min_weight > self.max_weight:
            raise ValueError(
                f"min_weight ({self.min_weight}) exceeds max_weight ({self.max_weight})"
            )
        if self.max_positions is not None and self.max_positions < 1:
            raise ValueError(f"max_positions must be >= 1, got {self.max_positions}")
        if self.risk_aversion < 0:
            raise ValueError(f"risk_aversion must be >= 0, got {self.risk_aversion}")


@dataclass(frozen=True)
class PortfolioProblem:
    """One optimization instance. Identical input for every method.

    Attributes:
        expected_returns: Per-asset expected return, shape (n,).
        covariance: Return covariance, shape (n, n). Must be symmetric.
        symbols: Asset identifiers, length n.
        constraints: Constraints to honour.
        current_weights: Existing allocation, shape (n,). Used for turnover and
            transaction-cost terms. Defaults to all-cash.
        transaction_cost_bps: Round-trip cost in basis points, charged on turnover.
    """

    expected_returns: np.ndarray
    covariance: np.ndarray
    symbols: list[str]
    constraints: Constraints = field(default_factory=Constraints)
    current_weights: np.ndarray | None = None
    transaction_cost_bps: float = 0.0

    def __post_init__(self) -> None:
        n = len(self.symbols)
        if self.expected_returns.shape != (n,):
            raise ValueError(
                f"expected_returns shape {self.expected_returns.shape} "
                f"does not match {n} symbols"
            )
        if self.covariance.shape != (n, n):
            raise ValueError(
                f"covariance shape {self.covariance.shape} does not match {n} symbols"
            )
        if not np.allclose(self.covariance, self.covariance.T, atol=1e-8):
            raise ValueError("covariance matrix must be symmetric")
        if self.current_weights is not None and self.current_weights.shape != (n,):
            raise ValueError(
                f"current_weights shape {self.current_weights.shape} "
                f"does not match {n} symbols"
            )
        if self.transaction_cost_bps < 0:
            raise ValueError(
                f"transaction_cost_bps must be >= 0, got {self.transaction_cost_bps}"
            )

    @property
    def n_assets(self) -> int:
        return len(self.symbols)

    def starting_weights(self) -> np.ndarray:
        """Current allocation, or all-cash when none is set."""
        if self.current_weights is None:
            return np.zeros(self.n_assets)
        return self.current_weights


@dataclass
class OptimizerResult:
    """Output of one optimizer run.

    `runtime_seconds` is set by `Optimizer.solve` rather than by the concrete
    method, so no implementation can accidentally under-report its own cost.
    """

    candidate_id: str
    method: str
    weights: np.ndarray
    objective_value: float
    expected_return: float
    expected_risk: float
    n_positions: int
    runtime_seconds: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def sharpe_proxy(self) -> float:
        """Expected return over expected risk. Not an annualized Sharpe ratio --
        that is computed from realized returns in `evaluation.metrics`."""
        if self.expected_risk <= 0:
            return 0.0
        return self.expected_return / self.expected_risk


class Optimizer(ABC):
    """Base class for all portfolio optimization methods."""

    #: Short stable name used in reports, tables, and figure legends.
    method: str = "unnamed"

    @abstractmethod
    def _solve(self, problem: PortfolioProblem) -> tuple[np.ndarray, dict[str, Any]]:
        """Return (weights, diagnostics). Implemented by each method.

        Implementations must not time themselves; `solve` does that.
        """

    def solve(self, problem: PortfolioProblem) -> OptimizerResult:
        """Run the method and package a timed, validated result."""
        start = time.perf_counter()
        weights, diagnostics = self._solve(problem)
        runtime = time.perf_counter() - start

        weights = np.asarray(weights, dtype=float)
        self._validate(weights, problem)

        gross_return = float(problem.expected_returns @ weights)
        cost = self._transaction_cost(weights, problem)
        expected_return = gross_return - cost
        variance = float(weights @ problem.covariance @ weights)
        expected_risk = float(np.sqrt(max(variance, 0.0)))
        objective = expected_return - problem.constraints.risk_aversion * variance

        return OptimizerResult(
            candidate_id=f"{self.method}-{problem.n_assets}a",
            method=self.method,
            weights=weights,
            objective_value=objective,
            expected_return=expected_return,
            expected_risk=expected_risk,
            n_positions=int(np.count_nonzero(np.abs(weights) > 1e-8)),
            runtime_seconds=runtime,
            diagnostics={**diagnostics, "transaction_cost": cost},
        )

    @staticmethod
    def _transaction_cost(weights: np.ndarray, problem: PortfolioProblem) -> float:
        """Cost charged on turnover from the current allocation."""
        if problem.transaction_cost_bps <= 0:
            return 0.0
        turnover = float(np.abs(weights - problem.starting_weights()).sum())
        return turnover * problem.transaction_cost_bps / 10_000.0

    @staticmethod
    def _validate(weights: np.ndarray, problem: PortfolioProblem) -> None:
        """Reject any solution that violates the stated constraints.

        A method that cannot satisfy the constraints must fail loudly here rather
        than return a quietly invalid portfolio into the comparison.
        """
        c = problem.constraints
        if weights.shape != (problem.n_assets,):
            raise ValueError(
                f"solver returned {weights.shape} weights for "
                f"{problem.n_assets} assets"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("solver returned non-finite weights")
        tol = 1e-6
        if np.any(weights < c.min_weight - tol):
            raise ValueError(
                f"weight below min_weight {c.min_weight}: {weights.min()}"
            )
        if np.any(weights > c.max_weight + tol):
            raise ValueError(
                f"weight above max_weight {c.max_weight}: {weights.max()}"
            )
        if c.fully_invested and not np.isclose(weights.sum(), 1.0, atol=1e-5):
            raise ValueError(f"weights must sum to 1, got {weights.sum()}")
        if not c.fully_invested and weights.sum() > 1.0 + tol:
            raise ValueError(f"weights sum to more than 1: {weights.sum()}")
        if c.max_positions is not None:
            held = int(np.count_nonzero(np.abs(weights) > 1e-8))
            if held > c.max_positions:
                raise ValueError(
                    f"{held} positions exceeds max_positions {c.max_positions}"
                )

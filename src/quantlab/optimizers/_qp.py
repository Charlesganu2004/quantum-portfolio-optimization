"""Shared quadratic-program solver for weights over a fixed asset subset.

Mean-variance, the greedy heuristic, and the QUBO method all need the same
question answered -- "given that I hold exactly these assets, what are the best
weights?" -- so they share one implementation. Only the *selection* of assets
differs between methods, which is precisely the axis the report compares.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize


class InfeasibleProblem(ValueError):
    """Raised when no allocation can satisfy the constraints."""


def required_positions(max_weight: float) -> int:
    """Minimum holdings needed to reach a fully-invested portfolio."""
    if max_weight >= 1.0:
        return 1
    return math.ceil(1.0 / max_weight)


def solve_weights(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float,
    min_weight: float,
    max_weight: float,
    fully_invested: bool,
) -> np.ndarray:
    """Maximize ``mu'w - lambda * w'Sigma w`` over the given assets.

    Returns weights for every asset passed in. Callers wanting a cardinality cap
    should pass only the selected subset and scatter the result back.
    """
    n = len(expected_returns)
    if n == 0:
        return np.zeros(0)

    if fully_invested and n * max_weight < 1.0 - 1e-9:
        raise InfeasibleProblem(
            f"{n} assets capped at {max_weight} cannot sum to 1; "
            f"need at least {required_positions(max_weight)} positions"
        )

    def objective(w: np.ndarray) -> float:
        return float(-expected_returns @ w + risk_aversion * (w @ covariance @ w))

    def gradient(w: np.ndarray) -> np.ndarray:
        return -expected_returns + 2.0 * risk_aversion * (covariance @ w)

    bounds = [(min_weight, max_weight)] * n
    constraints = []
    if fully_invested:
        constraints.append(
            {
                "type": "eq",
                "fun": lambda w: w.sum() - 1.0,
                "jac": lambda w: np.ones_like(w),
            }
        )
    else:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: 1.0 - w.sum(),
                "jac": lambda w: -np.ones_like(w),
            }
        )

    # Feasible, deterministic start: equal weight clipped into the box.
    start = np.clip(np.full(n, 1.0 / n), min_weight, max_weight)
    if fully_invested and start.sum() > 0:
        start = start / start.sum()
        start = np.clip(start, min_weight, max_weight)

    result = minimize(
        objective,
        start,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-12},
    )

    weights = np.clip(result.x, min_weight, max_weight)
    if fully_invested:
        total = weights.sum()
        if total <= 0:
            raise InfeasibleProblem("solver produced an all-zero allocation")
        weights = weights / total
        # Renormalizing can push a weight fractionally over the cap; a couple of
        # water-filling passes pull it back without breaking the sum.
        for _ in range(50):
            excess = np.maximum(weights - max_weight, 0.0).sum()
            if excess <= 1e-12:
                break
            weights = np.minimum(weights, max_weight)
            headroom = max_weight - weights
            room_total = headroom.sum()
            if room_total <= 1e-12:
                break
            weights = weights + headroom * (excess / room_total)
    return weights


def objective_value(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float,
) -> float:
    """``mu'w - lambda * w'Sigma w`` -- the quantity every method maximizes."""
    return float(
        expected_returns @ weights - risk_aversion * (weights @ covariance @ weights)
    )

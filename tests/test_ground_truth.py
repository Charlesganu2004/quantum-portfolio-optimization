"""Exhaustive-optimum checks and per-method quality regression bounds.

The gap-to-optimal numbers are the strongest claim this project makes, so they
are asserted rather than eyeballed. If an optimizer change degrades solution
quality, these fail -- which is how the QUBO annealer's two bugs would have been
caught automatically instead of by inspection.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.evaluation.ground_truth import (
    MAX_SUBSETS,
    GroundTruth,
    TooLargeToEnumerate,
    exhaustive_best,
    subset_count,
)
from quantlab.optimizers import (
    ALL_OPTIMIZERS,
    Constraints,
    HeuristicOptimizer,
    MeanVarianceOptimizer,
    PortfolioProblem,
    QuboOptimizer,
)

TRADING_DAYS = 252

#: Mean gap-to-optimal each method must stay within, as a fraction. Set with
#: headroom over measured performance so ordinary solver jitter does not fail the
#: build, but tight enough to catch a real regression. Measured 2026-08-03:
#: mean_variance 1.127%, heuristic 0.007%, qubo_annealing 0.184%.
MEAN_GAP_BUDGET = {
    "mean_variance": 0.030,
    "heuristic": 0.005,
    "qubo_annealing": 0.015,
}


def problem(n: int, seed: int, cap: int, risk_aversion: float = 25.0) -> PortfolioProblem:
    rng = np.random.default_rng(seed)
    daily_mu = rng.normal(0.0008, 0.0006, n)
    loadings = rng.normal(0, 1, (n, 3))
    daily_cov = loadings @ loadings.T * 1e-5 + np.eye(n) * 1e-4
    daily_cov = (daily_cov + daily_cov.T) / 2
    return PortfolioProblem(
        expected_returns=daily_mu * TRADING_DAYS,
        covariance=daily_cov * TRADING_DAYS,
        symbols=[f"A{i:02d}" for i in range(n)],
        constraints=Constraints(
            max_positions=cap, max_weight=0.4, risk_aversion=risk_aversion
        ),
    )


class TestExhaustiveSearch:
    @staticmethod
    def _diagonal_problem(n: int, seed: int, cap: int) -> PortfolioProblem:
        """Uncorrelated assets and no weight cap, so the optimum has a closed form."""
        rng = np.random.default_rng(seed)
        return PortfolioProblem(
            expected_returns=rng.normal(0.10, 0.08, n),
            covariance=np.diag(rng.uniform(0.02, 0.20, n)),
            symbols=[f"A{i:02d}" for i in range(n)],
            constraints=Constraints(
                max_positions=cap, max_weight=1.0, risk_aversion=3.0
            ),
        )

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_single_asset_optimum_matches_the_closed_form(self, seed: int) -> None:
        """With one position the portfolio is 100% one asset: no solver involved.

        The objective is ``mu_i - lambda * sigma_ii``, so the optimum is the
        largest of those, computed here directly from the inputs.
        """
        p = self._diagonal_problem(n=9, seed=seed, cap=1)
        lam = p.constraints.risk_aversion
        scores = p.expected_returns - lam * np.diag(p.covariance)

        truth = exhaustive_best(p)

        assert truth.selected == (int(np.argmax(scores)),)
        assert truth.objective_value == pytest.approx(float(scores.max()), abs=1e-7)

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_two_asset_optimum_matches_the_closed_form(self, seed: int) -> None:
        """Two uncorrelated assets maximize a one-variable quadratic in w.

        ``f(w) = mu1 w + mu2 (1-w) - lambda (s1 w^2 + s2 (1-w)^2)`` has its peak
        at ``w = (mu1 - mu2 + 2 lambda s2) / (2 lambda (s1 + s2))``, clipped to
        [0, 1]. Scanning every pair with that formula does not touch the QP
        solver that `exhaustive_best` uses, so the two can disagree.
        """
        from itertools import combinations

        p = self._diagonal_problem(n=8, seed=seed, cap=2)
        lam = p.constraints.risk_aversion
        mu = p.expected_returns
        var = np.diag(p.covariance)

        best_value, best_pair = -np.inf, ()
        for i, j in combinations(range(8), 2):
            w = (mu[i] - mu[j] + 2 * lam * var[j]) / (2 * lam * (var[i] + var[j]))
            w = float(np.clip(w, 0.0, 1.0))
            value = (
                mu[i] * w + mu[j] * (1 - w)
                - lam * (var[i] * w**2 + var[j] * (1 - w) ** 2)
            )
            if value > best_value:
                best_value, best_pair = value, (i, j)

        truth = exhaustive_best(p)

        assert truth.selected == best_pair
        assert truth.objective_value == pytest.approx(best_value, abs=1e-6)

    def test_reports_the_full_search_space(self) -> None:
        truth = exhaustive_best(problem(n=10, seed=1, cap=3))
        assert truth.subsets_evaluated == subset_count(10, 3) == 120
        assert len(truth.selected) == 3

    def test_refuses_an_intractable_search(self) -> None:
        with pytest.raises(TooLargeToEnumerate, match="exceeds"):
            exhaustive_best(problem(n=40, seed=1, cap=12))

    def test_requires_a_binding_cap(self) -> None:
        p = PortfolioProblem(
            expected_returns=np.array([0.1, 0.2]),
            covariance=np.eye(2) * 0.01,
            symbols=["A", "B"],
            constraints=Constraints(max_positions=None),
        )
        with pytest.raises(ValueError, match="binding cardinality cap"):
            exhaustive_best(p)

    def test_gap_never_reports_beating_the_optimum(self) -> None:
        """Solver tolerance must not manufacture a negative gap."""
        truth = exhaustive_best(problem(n=8, seed=2, cap=3))
        assert truth.gap_to(truth.objective_value + 1e-12) == 0.0
        assert truth.gap_to(truth.objective_value) == 0.0
        assert truth.gap_to(truth.objective_value * 0.5) > 0

    def test_shortfall_is_the_unnormalized_gap(self) -> None:
        truth = exhaustive_best(problem(n=8, seed=2, cap=3))
        worse = truth.objective_value - 0.05
        assert truth.shortfall_to(worse) == pytest.approx(0.05)
        assert truth.shortfall_to(truth.objective_value + 1e-12) == 0.0

    def test_gap_blows_up_near_a_zero_optimum_but_shortfall_does_not(self) -> None:
        """The reason both are reported: a tiny optimum inflates the percentage."""
        truth = GroundTruth(
            objective_value=-0.002,
            weights=np.zeros(4),
            selected=(0, 1),
            subsets_evaluated=6,
            subsets_infeasible=0,
        )
        assert truth.gap_to(-0.12) == pytest.approx(59.0)
        assert truth.shortfall_to(-0.12) == pytest.approx(0.118)

    def test_subset_limit_is_enforced_at_the_documented_value(self) -> None:
        """C(26,6) is the largest 6-subset search accepted; C(27,6) is refused.

        Both ends go through `exhaustive_best` itself, with the subset
        enumeration patched out so the test checks the guard and not the speed.
        """
        assert subset_count(26, 6) == 230_230 <= MAX_SUBSETS
        assert subset_count(27, 6) == 296_010 > MAX_SUBSETS

        with pytest.raises(TooLargeToEnumerate, match="exceeds"):
            exhaustive_best(problem(n=27, seed=1, cap=6))

    def test_search_at_the_limit_is_accepted(self, monkeypatch) -> None:
        """C(26,6) passes the size guard; the enumeration itself is stubbed."""
        import quantlab.evaluation.ground_truth as ground_truth

        seen: list[int] = []

        def fake_combinations(iterable, r):
            seen.append(r)
            return iter(())

        monkeypatch.setattr(ground_truth, "combinations", fake_combinations)
        with pytest.raises(Exception) as excinfo:
            exhaustive_best(problem(n=26, seed=1, cap=6))
        assert not isinstance(excinfo.value, TooLargeToEnumerate)
        assert seen == [6]


@pytest.mark.slow
class TestQualityRegression:
    """Per-method gap budgets over a fixed battery of problems."""

    CASES = [(12, 4), (16, 4), (16, 6)]
    SEEDS = [1, 2, 3, 4, 5]

    @pytest.fixture(scope="class")
    def gaps(self) -> dict[str, list[float]]:
        collected: dict[str, list[float]] = {c.method: [] for c in ALL_OPTIMIZERS}
        for n, cap in self.CASES:
            for seed in self.SEEDS:
                p = problem(n, seed, cap)
                truth = exhaustive_best(p)
                for cls in ALL_OPTIMIZERS:
                    collected[cls.method].append(
                        truth.gap_to(cls().solve(p).objective_value)
                    )
        return collected

    @pytest.mark.parametrize("cls", ALL_OPTIMIZERS, ids=lambda c: c.method)
    def test_mean_gap_within_budget(self, gaps, cls) -> None:
        budget = MEAN_GAP_BUDGET.get(cls.method)
        if budget is None:
            pytest.skip(f"no gap budget recorded for {cls.method}")
        mean_gap = float(np.mean(gaps[cls.method]))
        assert mean_gap <= budget, (
            f"{cls.method} mean gap {mean_gap:.4%} exceeds budget {budget:.4%}"
        )

    def test_qubo_is_competitive_with_truncation(self, gaps) -> None:
        """The quantum-inspired method must justify its runtime.

        Its claimed contribution is a smaller worst case than truncation on
        this battery, so it is held to a worst-case bound that mean-variance
        truncation does not meet. The bound is specific to these problems:
        `scripts/scalability_sweep.py` finds larger QUBO gaps on other ones.
        """
        qubo_worst = float(np.max(gaps[QuboOptimizer.method]))
        mv_worst = float(np.max(gaps[MeanVarianceOptimizer.method]))
        assert qubo_worst < mv_worst, (
            f"QUBO worst-case gap {qubo_worst:.4%} no longer beats mean-variance "
            f"truncation {mv_worst:.4%} -- the headline finding has regressed"
        )

    def test_heuristic_is_the_quality_leader(self, gaps) -> None:
        means = {m: float(np.mean(g)) for m, g in gaps.items()}
        assert means[HeuristicOptimizer.method] == min(means.values())

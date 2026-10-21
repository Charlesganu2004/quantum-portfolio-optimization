"""Expected-return estimator contract and statistical properties.

Mean-variance optimization amplifies error in `mu`, so an estimator that quietly
returns the wrong scale or the wrong sign corrupts every backtest downstream.
Each estimator is checked for the shared contract plus the specific property it
claims to provide.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantlab.data.estimators import (
    ALL_ESTIMATORS,
    GrandMeanEstimator,
    JamesSteinEstimator,
    MomentumEstimator,
    ReturnEstimator,
    SampleMeanEstimator,
    TRADING_DAYS,
    TrendQualityEstimator,
)
from quantlab.data.features import to_returns
from quantlab.data.sources import BarRequest, SyntheticBarSource


def history(assets: int = 12, sessions: int = 400, seed: int = 0):
    frame = SyntheticBarSource(seed=seed).fetch(
        BarRequest(
            tuple(f"S{i:02d}" for i in range(assets)), date(2023, 1, 1), date(2026, 6, 1)
        )
    ).tail(sessions)
    return frame, to_returns(frame)


@pytest.fixture(params=ALL_ESTIMATORS, ids=lambda e: e.name)
def estimator(request: pytest.FixtureRequest) -> ReturnEstimator:
    return request.param()


class TestContract:
    def test_returns_one_value_per_asset(self, estimator: ReturnEstimator) -> None:
        prices, returns = history()
        result = estimator.estimate(prices, returns)
        assert list(result.index) == list(returns.columns)

    def test_output_is_finite(self, estimator: ReturnEstimator) -> None:
        prices, returns = history()
        assert np.all(np.isfinite(estimator.estimate(prices, returns).to_numpy()))

    def test_is_deterministic(self, estimator: ReturnEstimator) -> None:
        prices, returns = history()
        first = estimator.estimate(prices, returns)
        second = estimator.estimate(prices, returns)
        pd.testing.assert_series_equal(first, second)

    def test_output_is_on_a_plausible_annual_scale(
        self, estimator: ReturnEstimator
    ) -> None:
        """Guards against a missing or doubled annualization factor."""
        prices, returns = history()
        values = estimator.estimate(prices, returns).to_numpy()
        assert np.abs(values).max() < 5.0, "expected returns above 500% signal a scale bug"

    def test_rejects_empty_input(self, estimator: ReturnEstimator) -> None:
        with pytest.raises(ValueError, match="empty"):
            estimator.estimate(pd.DataFrame(), pd.DataFrame())

    def test_reports_its_own_name(self, estimator: ReturnEstimator) -> None:
        assert estimator.name != "unnamed"

    def test_falls_back_rather_than_fabricating_on_short_history(
        self, estimator: ReturnEstimator
    ) -> None:
        """A window too short for the configured lookback must not invent a signal."""
        prices, returns = history(assets=6, sessions=20)
        result = estimator.estimate(prices, returns)
        assert np.all(np.isfinite(result.to_numpy()))
        assert len(result) == 6


class TestSampleMean:
    def test_is_the_annualized_trailing_mean(self) -> None:
        prices, returns = history()
        result = SampleMeanEstimator().estimate(prices, returns)
        np.testing.assert_allclose(
            result.to_numpy(), returns.mean().to_numpy() * TRADING_DAYS, rtol=1e-12
        )


class TestGrandMean:
    def test_assigns_every_asset_the_same_value(self) -> None:
        """The null hypothesis: no view on relative performance."""
        prices, returns = history()
        result = GrandMeanEstimator().estimate(prices, returns)
        assert result.nunique() == 1

    def test_equals_the_pooled_average(self) -> None:
        prices, returns = history()
        result = GrandMeanEstimator().estimate(prices, returns)
        expected = float(returns.to_numpy().mean()) * TRADING_DAYS
        assert float(result.iloc[0]) == pytest.approx(expected)


class TestJamesStein:
    def test_pulls_estimates_toward_the_grand_mean(self) -> None:
        prices, returns = history()
        sample = SampleMeanEstimator().estimate(prices, returns)
        shrunk = JamesSteinEstimator().estimate(prices, returns)
        grand = float(sample.mean())
        assert (shrunk - grand).abs().sum() <= (sample - grand).abs().sum() + 1e-12

    def test_preserves_the_cross_sectional_average(self) -> None:
        """Shrinkage redistributes around the mean; it must not shift it."""
        prices, returns = history()
        sample = SampleMeanEstimator().estimate(prices, returns)
        shrunk = JamesSteinEstimator().estimate(prices, returns)
        assert float(shrunk.mean()) == pytest.approx(float(sample.mean()), rel=1e-9)

    def test_preserves_ordering(self) -> None:
        prices, returns = history()
        sample = SampleMeanEstimator().estimate(prices, returns)
        shrunk = JamesSteinEstimator().estimate(prices, returns)
        assert list(sample.sort_values().index) == list(shrunk.sort_values().index)

    def test_passes_through_below_three_assets(self) -> None:
        """The James-Stein result requires three or more estimates."""
        prices, returns = history(assets=2)
        sample = SampleMeanEstimator().estimate(prices, returns)
        shrunk = JamesSteinEstimator().estimate(prices, returns)
        pd.testing.assert_series_equal(sample, shrunk)

    def test_shrinks_harder_on_noisier_data(self) -> None:
        prices, returns = history(assets=10, sessions=400)
        noisy = returns * 5.0
        grand_clean = float((returns.mean() * TRADING_DAYS).mean())
        grand_noisy = float((noisy.mean() * TRADING_DAYS).mean())

        clean_spread = float(
            (JamesSteinEstimator().estimate(prices, returns) - grand_clean).abs().mean()
        )
        sample_clean = float((returns.mean() * TRADING_DAYS - grand_clean).abs().mean())
        noisy_spread = float(
            (JamesSteinEstimator().estimate(prices, noisy) - grand_noisy).abs().mean()
        )
        sample_noisy = float((noisy.mean() * TRADING_DAYS - grand_noisy).abs().mean())

        # Retained fraction of the original dispersion falls as noise rises.
        assert noisy_spread / sample_noisy <= clean_spread / sample_clean + 1e-9


class TestMomentum:
    def test_ranks_the_strongest_riser_highest(self) -> None:
        index = pd.bdate_range("2024-01-01", periods=200)
        prices = pd.DataFrame(
            {
                "WINNER": 100 * np.exp(0.004 * np.arange(200)),
                "FLAT": np.full(200, 100.0),
                "LOSER": 100 * np.exp(-0.004 * np.arange(200)),
            },
            index=index,
        )
        result = MomentumEstimator(lookback=60, skip=5).estimate(
            prices, to_returns(prices)
        )
        assert result["WINNER"] > result["FLAT"] > result["LOSER"]

    def test_falls_back_when_lookback_exceeds_history(self) -> None:
        prices, returns = history(assets=5, sessions=30)
        result = MomentumEstimator(lookback=200, skip=5).estimate(prices, returns)
        pd.testing.assert_series_equal(
            result, SampleMeanEstimator().estimate(prices, returns)
        )

    def test_rejects_negative_spread(self) -> None:
        with pytest.raises(ValueError, match="spread"):
            MomentumEstimator(spread=-0.1)


class TestTrendQuality:
    def test_prefers_a_steady_riser_over_an_equal_sized_jump(self) -> None:
        index = pd.bdate_range("2024-01-01", periods=200)
        steady = 100 * np.exp(0.003 * np.arange(200))
        jump = np.concatenate([np.full(100, 100.0), np.full(100, steady[-1])])
        prices = pd.DataFrame({"STEADY": steady, "JUMP": jump, "FLAT": 100.0}, index=index)
        result = TrendQualityEstimator(lookback=60, skip=5, window=60).estimate(
            prices, to_returns(prices)
        )
        assert result["STEADY"] > result["JUMP"]

    def test_falls_back_when_window_exceeds_history(self) -> None:
        prices, returns = history(assets=5, sessions=30)
        result = TrendQualityEstimator(lookback=200, window=200).estimate(prices, returns)
        pd.testing.assert_series_equal(
            result, SampleMeanEstimator().estimate(prices, returns)
        )


class TestIntegration:
    @pytest.mark.parametrize("cls", ALL_ESTIMATORS, ids=lambda c: c.name)
    def test_output_drives_an_optimizer(self, cls) -> None:
        from quantlab.data.features import annualized_moments
        from quantlab.optimizers import Constraints, PortfolioProblem, QuboOptimizer

        prices, returns = history(assets=10, sessions=300)
        _, covariance, symbols, _ = annualized_moments(returns)
        expected = cls().estimate(prices, returns).reindex(symbols).to_numpy()

        problem = PortfolioProblem(
            expected_returns=expected,
            covariance=covariance,
            symbols=symbols,
            constraints=Constraints(max_positions=4, max_weight=0.4, risk_aversion=25.0),
        )
        assert QuboOptimizer().solve(problem).n_positions <= 4

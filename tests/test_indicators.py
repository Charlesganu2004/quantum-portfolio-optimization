"""Indicator correctness.

These feed the expected-return estimators, so a silently wrong RSI or momentum
convention would propagate into every backtest number without failing anything.
Definitions are pinned against hand-computed values, not just self-consistency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.data.indicators import (
    TRADING_DAYS,
    average_true_range,
    cross_sectional_zscore,
    ema_crossover_signal,
    exponential_moving_average,
    momentum,
    realized_volatility,
    relative_strength_index,
    trend_strength,
)


def price_frame(values: dict[str, list[float]]) -> pd.DataFrame:
    length = len(next(iter(values.values())))
    return pd.DataFrame(values, index=pd.bdate_range("2026-01-01", periods=length))


class TestEma:
    def test_matches_hand_computed_smoothing(self) -> None:
        """alpha = 2/(span+1); first value seeds the series."""
        frame = price_frame({"A": [10.0, 20.0, 30.0]})
        result = exponential_moving_average(frame, span=2)["A"].tolist()
        alpha = 2.0 / 3.0
        expected = [10.0]
        for price in (20.0, 30.0):
            expected.append(alpha * price + (1 - alpha) * expected[-1])
        np.testing.assert_allclose(result, expected)

    def test_constant_prices_give_constant_ema(self) -> None:
        frame = price_frame({"A": [42.0] * 20})
        assert exponential_moving_average(frame, span=5)["A"].nunique() == 1

    def test_rejects_invalid_span(self) -> None:
        with pytest.raises(ValueError, match="span"):
            exponential_moving_average(price_frame({"A": [1.0, 2.0]}), span=0)


class TestEmaCrossover:
    def test_uptrend_is_positive(self) -> None:
        frame = price_frame({"A": list(np.linspace(100, 200, 80))})
        assert ema_crossover_signal(frame, 5, 20)["A"].iloc[-1] > 0

    def test_downtrend_is_negative(self) -> None:
        frame = price_frame({"A": list(np.linspace(200, 100, 80))})
        assert ema_crossover_signal(frame, 5, 20)["A"].iloc[-1] < 0

    def test_is_scale_invariant(self) -> None:
        """A $20 stock and a $2000 stock must yield the same signal."""
        path = list(np.linspace(100, 150, 60))
        cheap = ema_crossover_signal(price_frame({"A": path}), 5, 20)["A"].iloc[-1]
        dear = ema_crossover_signal(
            price_frame({"A": [p * 100 for p in path]}), 5, 20
        )["A"].iloc[-1]
        assert cheap == pytest.approx(dear, rel=1e-9)

    def test_rejects_fast_span_at_or_above_slow(self) -> None:
        with pytest.raises(ValueError, match="fast_span"):
            ema_crossover_signal(price_frame({"A": [1.0, 2.0]}), 20, 20)


class TestRsi:
    def test_stays_within_bounds(self) -> None:
        rng = np.random.default_rng(0)
        walk = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 200)))
        rsi = relative_strength_index(price_frame({"A": list(walk)}))["A"].dropna()
        assert rsi.min() >= 0.0
        assert rsi.max() <= 100.0

    def test_monotonic_rise_pins_at_hundred(self) -> None:
        frame = price_frame({"A": list(np.linspace(100, 200, 60))})
        assert relative_strength_index(frame)["A"].iloc[-1] == pytest.approx(100.0)

    def test_monotonic_fall_pins_at_zero(self) -> None:
        frame = price_frame({"A": list(np.linspace(200, 100, 60))})
        assert relative_strength_index(frame)["A"].iloc[-1] == pytest.approx(0.0)

    def test_uses_wilder_smoothing_not_simple_average(self) -> None:
        """Wilder's alpha = 1/window differs materially from an SMA variant."""
        rng = np.random.default_rng(7)
        walk = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, 120)))
        frame = price_frame({"A": list(walk)})
        wilder = relative_strength_index(frame, window=14)["A"].iloc[-1]

        delta = frame["A"].diff()
        sma_gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        sma_loss = (-delta).clip(lower=0).rolling(14).mean().iloc[-1]
        sma_rsi = 100 - 100 / (1 + sma_gain / sma_loss)
        assert abs(wilder - sma_rsi) > 1e-6

    def test_rejects_tiny_window(self) -> None:
        with pytest.raises(ValueError, match="window"):
            relative_strength_index(price_frame({"A": [1.0, 2.0]}), window=1)


class TestRealizedVolatility:
    def test_annualization_factor(self) -> None:
        rng = np.random.default_rng(1)
        returns = pd.DataFrame(
            {"A": rng.normal(0, 0.01, 100)}, index=pd.bdate_range("2026-01-01", periods=100)
        )
        daily = realized_volatility(returns, 20, annualize=False)["A"].iloc[-1]
        annual = realized_volatility(returns, 20, annualize=True)["A"].iloc[-1]
        assert annual == pytest.approx(daily * np.sqrt(TRADING_DAYS))

    def test_higher_dispersion_gives_higher_volatility(self) -> None:
        rng = np.random.default_rng(2)
        index = pd.bdate_range("2026-01-01", periods=100)
        calm = pd.DataFrame({"A": rng.normal(0, 0.005, 100)}, index=index)
        wild = pd.DataFrame({"A": rng.normal(0, 0.05, 100)}, index=index)
        assert (
            realized_volatility(wild, 20)["A"].iloc[-1]
            > realized_volatility(calm, 20)["A"].iloc[-1]
        )


class TestAtr:
    def test_tracks_the_daily_range(self) -> None:
        n = 40
        index = pd.bdate_range("2026-01-01", periods=n)
        close = pd.DataFrame({"A": [100.0] * n}, index=index)
        high = pd.DataFrame({"A": [102.0] * n}, index=index)
        low = pd.DataFrame({"A": [98.0] * n}, index=index)
        atr = average_true_range(high, low, close)["A"].iloc[-1]
        assert atr == pytest.approx(4.0, rel=1e-6)


class TestMomentum:
    def test_measures_total_return(self) -> None:
        frame = price_frame({"A": [100.0] * 10 + [150.0]})
        assert momentum(frame, lookback=10)["A"] == pytest.approx(0.5)

    def test_skip_excludes_the_most_recent_move(self) -> None:
        """The last period is dominated by reversal, so `skip` must drop it."""
        frame = price_frame({"A": [100.0, 100.0, 100.0, 120.0, 60.0]})
        with_skip = momentum(frame, lookback=2, skip=1)["A"]
        without_skip = momentum(frame, lookback=2, skip=0)["A"]
        assert with_skip == pytest.approx(0.2)
        assert without_skip < 0

    def test_rejects_insufficient_history(self) -> None:
        with pytest.raises(ValueError, match="need at least"):
            momentum(price_frame({"A": [1.0, 2.0]}), lookback=10)

    def test_rejects_negative_skip(self) -> None:
        with pytest.raises(ValueError, match="skip"):
            momentum(price_frame({"A": [1.0] * 20}), lookback=5, skip=-1)


class TestTrendStrength:
    def test_perfect_exponential_trend_scores_one(self) -> None:
        frame = price_frame({"A": list(100 * np.exp(0.01 * np.arange(60)))})
        assert trend_strength(frame, 60)["A"] == pytest.approx(1.0, abs=1e-9)

    def test_stays_within_zero_and_one(self) -> None:
        rng = np.random.default_rng(3)
        walk = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 120)))
        score = trend_strength(price_frame({"A": list(walk)}), 60)["A"]
        assert 0.0 <= score <= 1.0

    def test_separates_steady_drift_from_a_single_jump(self) -> None:
        """The distinction momentum alone cannot make."""
        steady = list(100 * np.exp(0.004 * np.arange(60)))
        jump = [100.0] * 30 + [127.0] * 30
        assert (
            trend_strength(price_frame({"A": steady}), 60)["A"]
            > trend_strength(price_frame({"A": jump}), 60)["A"]
        )

    def test_rejects_tiny_window(self) -> None:
        with pytest.raises(ValueError, match="window"):
            trend_strength(price_frame({"A": [1.0] * 10}), window=2)


class TestZscore:
    def test_standardizes_to_zero_mean_unit_spread(self) -> None:
        result = cross_sectional_zscore(pd.Series([1.0, 2.0, 3.0, 4.0]))
        assert float(result.mean()) == pytest.approx(0.0, abs=1e-12)
        assert float(result.std(ddof=0)) == pytest.approx(1.0)

    def test_identical_values_give_zero_not_nan(self) -> None:
        result = cross_sectional_zscore(pd.Series([5.0, 5.0, 5.0]))
        assert (result == 0.0).all()

    def test_all_nan_input_passes_through(self) -> None:
        result = cross_sectional_zscore(pd.Series([np.nan, np.nan]))
        assert result.isna().all()

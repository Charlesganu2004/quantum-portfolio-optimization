"""Expected-return estimators -- the input that dominates portfolio outcomes.

Mean-variance optimization is far more sensitive to errors in expected returns
than to errors in the covariance matrix, and it actively concentrates into
whichever assets have the largest estimation error. Michaud named this "error
maximization": the optimizer treats a noisy high estimate as an opportunity.

The sample mean of trailing returns -- the obvious estimator, and what the
pipeline used first -- is an especially poor predictor over short windows. This
module makes the estimator a swappable, benchmarked component so the report can
answer a question the four-optimizer comparison alone cannot: does the choice of
`mu` estimator matter more than the choice of optimizer?

Every estimator returns annualized expected returns aligned to the input columns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .indicators import cross_sectional_zscore, momentum, trend_strength

TRADING_DAYS = 252


class ReturnEstimator(ABC):
    """Estimates annualized expected returns from price history."""

    name: str = "unnamed"

    @abstractmethod
    def estimate(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        ...

    def _check(self, prices: pd.DataFrame) -> None:
        if prices.empty:
            raise ValueError(f"{self.name}: empty price frame")


class SampleMeanEstimator(ReturnEstimator):
    """Trailing mean return, annualized. The naive baseline."""

    name = "sample_mean"

    def estimate(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        self._check(prices)
        return returns.mean() * TRADING_DAYS


class GrandMeanEstimator(ReturnEstimator):
    """One number for every asset: the cross-sectional average.

    Deliberately carries no view on which asset outperforms. Any estimator that
    cannot beat this is contributing noise, not information, so it is the honest
    null hypothesis for the comparison.
    """

    name = "grand_mean"

    def estimate(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        self._check(prices)
        grand = float(returns.to_numpy().mean()) * TRADING_DAYS
        return pd.Series(grand, index=returns.columns)


class JamesSteinEstimator(ReturnEstimator):
    """Shrink per-asset means toward the grand mean.

    The James-Stein result is that shrinking individual estimates toward a common
    target dominates the sample mean in expected squared error for three or more
    assets. Shrinkage intensity rises with the noise-to-dispersion ratio, so
    scarce or noisy data pulls harder toward the grand mean.
    """

    name = "james_stein"

    def estimate(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        self._check(prices)
        observations, n_assets = returns.shape
        sample = returns.mean() * TRADING_DAYS
        if n_assets < 3 or observations < 2:
            return sample

        grand = float(sample.mean())
        dispersion = float(((sample - grand) ** 2).sum())
        if dispersion <= 0:
            return sample

        # Average sampling variance of each asset's annualized mean.
        noise = float(
            (returns.var(ddof=1) * TRADING_DAYS**2 / observations).mean()
        )
        intensity = float(np.clip((n_assets - 2) * noise / dispersion, 0.0, 1.0))
        return grand + (1.0 - intensity) * (sample - grand)


class MomentumEstimator(ReturnEstimator):
    """Rank assets by trailing momentum, skipping the most recent sessions.

    Converts a cross-sectional momentum score into expected returns by scaling a
    standardized score around the grand mean, so the output stays on a plausible
    return scale instead of asserting implausible magnitudes.
    """

    name = "momentum"

    def __init__(self, lookback: int = 60, skip: int = 5, spread: float = 0.10) -> None:
        if spread < 0:
            raise ValueError(f"spread must be >= 0, got {spread}")
        self.lookback = lookback
        self.skip = skip
        self.spread = spread

    def estimate(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        self._check(prices)
        required = self.lookback + self.skip + 1
        if len(prices) < required:
            # Not enough history for the configured window: fall back rather than
            # fabricate a signal from a truncated lookback.
            return SampleMeanEstimator().estimate(prices, returns)

        scores = cross_sectional_zscore(
            momentum(prices, lookback=self.lookback, skip=self.skip)
        )
        grand = float(returns.to_numpy().mean()) * TRADING_DAYS
        return grand + scores.fillna(0.0) * self.spread


class TrendQualityEstimator(ReturnEstimator):
    """Momentum weighted by how consistent the trend is.

    Scales each momentum score by the R-squared of a linear fit to log price, so
    a steady drift counts for more than an equal move produced by a single jump.
    """

    name = "trend_quality"

    def __init__(
        self, lookback: int = 60, skip: int = 5, window: int = 60, spread: float = 0.10
    ) -> None:
        self.lookback = lookback
        self.skip = skip
        self.window = window
        self.spread = spread

    def estimate(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        self._check(prices)
        required = max(self.lookback + self.skip + 1, self.window)
        if len(prices) < required:
            return SampleMeanEstimator().estimate(prices, returns)

        raw = momentum(prices, lookback=self.lookback, skip=self.skip)
        quality = trend_strength(prices, window=self.window).reindex(raw.index)
        scores = cross_sectional_zscore(raw * quality.fillna(0.0))
        grand = float(returns.to_numpy().mean()) * TRADING_DAYS
        return grand + scores.fillna(0.0) * self.spread


#: Every estimator in the comparison, naive baseline first.
ALL_ESTIMATORS: list[type[ReturnEstimator]] = [
    SampleMeanEstimator,
    GrandMeanEstimator,
    JamesSteinEstimator,
    MomentumEstimator,
    TrendQualityEstimator,
]

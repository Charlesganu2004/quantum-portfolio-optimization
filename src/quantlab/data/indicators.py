"""Technical indicators and trend features.

Each function is a pure transform over a price or return frame, so indicators can
be composed into expected-return estimators (`quantlab.data.estimators`) and
cited against their entry in the formula knowledge base
(`quantlab.knowledge.formulas`).

Every definition here matches a source-backed record in that knowledge base, so
the LLM reviewer can explain a signal by citing the formula rather than
paraphrasing it from memory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def exponential_moving_average(prices: pd.DataFrame, span: int) -> pd.DataFrame:
    """EMA with the conventional ``alpha = 2 / (span + 1)`` smoothing."""
    if span < 1:
        raise ValueError(f"span must be >= 1, got {span}")
    return prices.ewm(span=span, adjust=False).mean()


def ema_crossover_signal(
    prices: pd.DataFrame, fast_span: int = 12, slow_span: int = 26
) -> pd.DataFrame:
    """Normalized gap between a fast and slow EMA.

    Positive means the fast average leads -- conventionally read as an uptrend.
    Dividing by the slow EMA makes the value comparable across price levels, so
    a $500 stock and a $20 stock produce signals on the same scale.
    """
    if fast_span >= slow_span:
        raise ValueError(
            f"fast_span ({fast_span}) must be below slow_span ({slow_span})"
        )
    fast = exponential_moving_average(prices, fast_span)
    slow = exponential_moving_average(prices, slow_span)
    return (fast - slow) / slow.replace(0.0, np.nan)


def relative_strength_index(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder's RSI, bounded 0-100.

    Uses Wilder's smoothing (``alpha = 1/window``) rather than a simple moving
    average, which is the original definition and differs materially from the
    SMA variant many libraries ship.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()

    # No losses in the window means maximum strength; guard the division rather
    # than emitting inf.
    relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    return rsi.where(avg_loss != 0.0, 100.0).where(avg_gain != 0.0, 0.0)


def realized_volatility(
    returns: pd.DataFrame, window: int = 20, annualize: bool = True
) -> pd.DataFrame:
    """Rolling standard deviation of returns."""
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    vol = returns.rolling(window).std(ddof=1)
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol


def average_true_range(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14
) -> pd.DataFrame:
    """Wilder's ATR -- average of the true range over `window` sessions."""
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).stack(),
            (high - previous_close).abs().stack(),
            (low - previous_close).abs().stack(),
        ],
        axis=1,
    ).max(axis=1).unstack()
    return true_range.ewm(alpha=1.0 / window, adjust=False).mean()


def momentum(prices: pd.DataFrame, lookback: int, skip: int = 0) -> pd.Series:
    """Total return over `lookback` sessions, optionally skipping the most recent.

    `skip` implements the standard momentum convention of excluding the latest
    period, which is dominated by short-term reversal and otherwise contaminates
    the signal.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if skip < 0:
        raise ValueError(f"skip must be >= 0, got {skip}")
    if len(prices) < lookback + skip + 1:
        raise ValueError(
            f"need at least {lookback + skip + 1} rows for lookback={lookback} "
            f"skip={skip}, got {len(prices)}"
        )
    end = prices.iloc[-1 - skip]
    start = prices.iloc[-1 - skip - lookback]
    return (end / start) - 1.0


def trend_strength(prices: pd.DataFrame, window: int = 60) -> pd.Series:
    """R-squared of a linear fit to log price -- how *consistent* a trend is.

    Distinct from momentum, which measures size and direction. A name can have
    large momentum from one jump (low trend strength) or a modest but steady
    drift (high trend strength). Position sizing generally prefers the latter.
    """
    if window < 3:
        raise ValueError(f"window must be >= 3, got {window}")
    recent = prices.tail(window)
    if len(recent) < 3:
        raise ValueError(f"need at least 3 rows, got {len(recent)}")

    time = np.arange(len(recent), dtype=float)
    time_centered = time - time.mean()
    denominator = float((time_centered**2).sum())

    scores = {}
    for column in recent.columns:
        series = np.log(recent[column].to_numpy(dtype=float))
        if not np.all(np.isfinite(series)):
            scores[column] = np.nan
            continue
        centered = series - series.mean()
        slope = float((time_centered * centered).sum() / denominator)
        residual = centered - slope * time_centered
        total = float((centered**2).sum())
        scores[column] = 1.0 - float((residual**2).sum()) / total if total > 0 else 0.0
    return pd.Series(scores)


def cross_sectional_zscore(values: pd.Series) -> pd.Series:
    """Standardize across assets so signals of different units combine sensibly."""
    clean = values.dropna()
    if clean.empty:
        return values
    spread = float(clean.std(ddof=0))
    if spread <= 0:
        return pd.Series(0.0, index=values.index)
    return (values - float(clean.mean())) / spread

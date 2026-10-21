"""Performance metrics for the Project II evaluation.

Every metric is computed from a realized equity curve or return series, never
from an optimizer's own expected values -- an optimizer scoring itself on its
own forecasts measures optimism, not performance.

Each definition here has a matching source-backed record in
`quantlab.knowledge.formulas`, so the LLM reviewer can cite the formula rather
than paraphrase it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

TRADING_DAYS = 252


class InsufficientObservations(ValueError):
    """Raised when a metric cannot be computed from the given series."""


@dataclass(frozen=True)
class PerformanceMetrics:
    """Realized performance over one evaluation run."""

    periods: int
    periods_per_year: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    mean_turnover: float
    mean_positions: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def equity_curve(period_returns: np.ndarray, start: float = 1.0) -> np.ndarray:
    """Compound a return series into an equity curve including the starting point."""
    returns = np.asarray(period_returns, dtype=float)
    if returns.size == 0:
        raise InsufficientObservations("cannot build an equity curve from no returns")
    return np.concatenate([[start], start * np.cumprod(1.0 + returns)])


def max_drawdown(curve: np.ndarray) -> float:
    """Largest peak-to-trough decline, as a negative fraction."""
    curve = np.asarray(curve, dtype=float)
    if curve.size < 2:
        raise InsufficientObservations("need at least 2 points for a drawdown")
    peak = np.maximum.accumulate(curve)
    return float((curve / peak - 1.0).min())


def sharpe_ratio(
    period_returns: np.ndarray, periods_per_year: float, risk_free: float = 0.0
) -> float:
    """Annualized excess return over annualized volatility.

    Returns 0.0 for a constant series rather than infinity: zero observed
    variance over a short sample means "not measurable", not "riskless".
    """
    returns = np.asarray(period_returns, dtype=float)
    if returns.size < 2:
        raise InsufficientObservations("need at least 2 returns for a Sharpe ratio")
    excess = returns - risk_free / periods_per_year
    spread = float(excess.std(ddof=1))
    if spread <= 0:
        return 0.0
    return float(excess.mean() / spread * np.sqrt(periods_per_year))


def sortino_ratio(
    period_returns: np.ndarray, periods_per_year: float, risk_free: float = 0.0
) -> float:
    """Like Sharpe, but penalizing only downside deviation.

    Upside volatility is not a risk an investor wants removed, so Sortino is the
    more honest ratio for asymmetric strategies.
    """
    returns = np.asarray(period_returns, dtype=float)
    if returns.size < 2:
        raise InsufficientObservations("need at least 2 returns for a Sortino ratio")
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    if downside.size == 0:
        return 0.0  # No losing period observed; not evidence of infinite quality.
    deviation = float(np.sqrt((downside**2).mean()))
    if deviation <= 0:
        return 0.0
    return float(excess.mean() / deviation * np.sqrt(periods_per_year))


def win_rate(period_returns: np.ndarray) -> float:
    returns = np.asarray(period_returns, dtype=float)
    if returns.size == 0:
        raise InsufficientObservations("no returns")
    return float((returns > 0).mean())


def profit_factor(period_returns: np.ndarray) -> float:
    """Gross gains divided by gross losses.

    Returns 0.0 when there are no gains, and infinity only when there are gains
    and genuinely zero losses.
    """
    returns = np.asarray(period_returns, dtype=float)
    if returns.size == 0:
        raise InsufficientObservations("no returns")
    gains = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def summarize(
    period_returns: np.ndarray,
    periods_per_year: float,
    turnovers: np.ndarray | None = None,
    position_counts: np.ndarray | None = None,
    risk_free: float = 0.0,
) -> PerformanceMetrics:
    """Full metric set from a realized period-return series."""
    returns = np.asarray(period_returns, dtype=float)
    if returns.size < 2:
        raise InsufficientObservations(
            f"need at least 2 periods to summarize, got {returns.size}"
        )

    curve = equity_curve(returns)
    total = float(curve[-1] / curve[0] - 1.0)
    years = returns.size / periods_per_year
    annualized = float((curve[-1] / curve[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
    drawdown = max_drawdown(curve)

    return PerformanceMetrics(
        periods=int(returns.size),
        periods_per_year=float(periods_per_year),
        total_return=total,
        annualized_return=annualized,
        annualized_volatility=volatility,
        sharpe_ratio=sharpe_ratio(returns, periods_per_year, risk_free),
        sortino_ratio=sortino_ratio(returns, periods_per_year, risk_free),
        max_drawdown=drawdown,
        calmar_ratio=float(annualized / abs(drawdown)) if drawdown < 0 else 0.0,
        win_rate=win_rate(returns),
        profit_factor=profit_factor(returns),
        mean_turnover=float(np.mean(turnovers)) if turnovers is not None and len(turnovers) else 0.0,
        mean_positions=float(np.mean(position_counts))
        if position_counts is not None and len(position_counts)
        else 0.0,
    )

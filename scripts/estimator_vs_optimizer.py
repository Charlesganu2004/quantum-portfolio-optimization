"""Which matters more: the expected-return estimator, or the optimizer?

Project II compares four optimizers. That comparison is only worth its page count
if the optimizer is actually the dominant lever. This runs a walk-forward
backtest across the full estimator x optimizer grid and attributes realized
out-of-sample performance to each factor.

Everything is measured out-of-sample: parameters are fit on a trailing window and
scored on the following holding period only. In-sample objective values would
flatter every method and answer nothing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from quantlab.data.estimators import ALL_ESTIMATORS, ReturnEstimator
from quantlab.data.features import annualized_moments, to_returns
from quantlab.data.sources import BarRequest, SyntheticBarSource, resolve_source
from quantlab.optimizers import ALL_OPTIMIZERS, Constraints, Optimizer, PortfolioProblem

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    estimator: str
    optimizer: str
    periods: int
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    mean_positions: float
    mean_turnover: float


def walk_forward(
    prices: pd.DataFrame,
    estimator: ReturnEstimator,
    optimizer: Optimizer,
    constraints: Constraints,
    train_window: int = 120,
    hold_period: int = 10,
) -> BacktestResult:
    """Fit on a trailing window, hold for `hold_period`, repeat."""
    equity = [1.0]
    position_counts: list[int] = []
    turnovers: list[float] = []
    previous_weights: np.ndarray | None = None

    start = train_window
    while start + hold_period <= len(prices):
        train = prices.iloc[start - train_window : start]
        future = prices.iloc[start - 1 : start + hold_period]

        train_returns = to_returns(train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, covariance, symbols, _ = annualized_moments(train_returns)
        expected = estimator.estimate(train, train_returns).reindex(symbols).to_numpy()

        problem = PortfolioProblem(
            expected_returns=expected,
            covariance=covariance,
            symbols=symbols,
            constraints=constraints,
            current_weights=previous_weights,
        )
        try:
            weights = optimizer.solve(problem).weights
        except ValueError:
            # A method that cannot solve this window sits in cash rather than
            # silently inheriting the previous allocation.
            weights = np.zeros(len(symbols))

        realized = (future[symbols].iloc[-1] / future[symbols].iloc[0] - 1.0).to_numpy()
        equity.append(equity[-1] * (1.0 + float(weights @ realized)))

        position_counts.append(int(np.count_nonzero(weights > 1e-8)))
        if previous_weights is not None:
            turnovers.append(float(np.abs(weights - previous_weights).sum()))
        previous_weights = weights
        start += hold_period

    curve = np.asarray(equity)
    periods = len(curve) - 1
    if periods < 2:
        raise ValueError("not enough history for a walk-forward backtest")

    period_returns = curve[1:] / curve[:-1] - 1.0
    periods_per_year = TRADING_DAYS / hold_period
    annual_return = float(curve[-1] ** (periods_per_year / periods) - 1.0)
    annual_vol = float(period_returns.std(ddof=1) * np.sqrt(periods_per_year))
    peak = np.maximum.accumulate(curve)
    max_drawdown = float((curve / peak - 1.0).min())

    return BacktestResult(
        estimator=estimator.name,
        optimizer=optimizer.method,
        periods=periods,
        annualized_return=annual_return,
        annualized_volatility=annual_vol,
        sharpe=annual_return / annual_vol if annual_vol > 0 else 0.0,
        max_drawdown=max_drawdown,
        mean_positions=float(np.mean(position_counts)),
        mean_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
    )


def attribute_variance(results: list[BacktestResult]) -> tuple[float, float]:
    """Spread in Sharpe explained by switching estimator vs switching optimizer.

    Reported as the mean range across the other factor -- a plain, assumption-free
    read of which knob moves the outcome further.
    """
    frame = pd.DataFrame([r.__dict__ for r in results])
    estimator_effect = float(
        frame.groupby("optimizer")["sharpe"].agg(lambda s: s.max() - s.min()).mean()
    )
    optimizer_effect = float(
        frame.groupby("estimator")["sharpe"].agg(lambda s: s.max() - s.min()).mean()
    )
    return estimator_effect, optimizer_effect


def main() -> None:
    source = resolve_source()
    if source.name == "synthetic":
        source = SyntheticBarSource(seed=20260803)
        symbols = tuple(f"SYN{i:02d}" for i in range(16))
        print("No Alpaca credentials -- running on the offline factor model.\n")
    else:
        from quantlab.data.sources import DEFAULT_UNIVERSE

        symbols = DEFAULT_UNIVERSE
        print(f"Using live source: {source.name}\n")

    prices = source.fetch(
        BarRequest(symbols=symbols, start=date(2023, 1, 1), end=date(2026, 8, 1))
    )
    print(f"History: {len(prices)} sessions x {len(prices.columns)} symbols")
    print("Walk-forward: 120-session train, 10-session hold, cap 4 positions\n")

    constraints = Constraints(max_positions=4, max_weight=0.4, risk_aversion=25.0)
    results: list[BacktestResult] = []
    for estimator_cls in ALL_ESTIMATORS:
        for optimizer_cls in ALL_OPTIMIZERS:
            results.append(
                walk_forward(
                    prices, estimator_cls(), optimizer_cls(), constraints
                )
            )

    header = (
        f"{'estimator':<15} {'optimizer':<16} {'ann.ret':>9} {'ann.vol':>9} "
        f"{'Sharpe':>8} {'maxDD':>8} {'turn':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.estimator:<15} {r.optimizer:<16} {r.annualized_return:>8.2%} "
            f"{r.annualized_volatility:>8.2%} {r.sharpe:>8.3f} "
            f"{r.max_drawdown:>7.2%} {r.mean_turnover:>7.2f}"
        )

    estimator_effect, optimizer_effect = attribute_variance(results)
    print("\n" + "=" * len(header))
    print(f"Mean Sharpe spread from changing the ESTIMATOR : {estimator_effect:.3f}")
    print(f"Mean Sharpe spread from changing the OPTIMIZER : {optimizer_effect:.3f}")
    dominant = "ESTIMATOR" if estimator_effect > optimizer_effect else "OPTIMIZER"
    ratio = (
        estimator_effect / optimizer_effect
        if optimizer_effect > 1e-12
        else float("inf")
    )
    print(f"Dominant lever: {dominant}  ({ratio:.1f}x)")


if __name__ == "__main__":
    main()

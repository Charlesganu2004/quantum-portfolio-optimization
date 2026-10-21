"""End-to-end check: bars -> annualized moments -> all optimizers.

Also demonstrates the covariance conditioning problem that a 30-session lookback
(the window in the proposal) creates as the universe grows, which determines the
usable lookback for the scalability sweep. Lookbacks here are counted in
sessions, not calendar days.
"""

from __future__ import annotations

import warnings

import numpy as np

from quantlab.config import load_settings
from quantlab.data.features import annualized_moments, to_returns
from quantlab.data.sources import (
    DEFAULT_UNIVERSE,
    BarRequest,
    SyntheticBarSource,
    default_request,
    resolve_source,
)
from quantlab.optimizers import ALL_OPTIMIZERS, Constraints, PortfolioProblem


def show_conditioning() -> None:
    """How lookback length interacts with universe size."""
    print("Covariance conditioning: does the lookback support the universe?\n")
    header = (
        f"{'assets':>7} {'lookback':>9} {'obs':>5} {'rank':>5} "
        f"{'condition':>12} {'shrinkage':>10}  verdict"
    )
    print(header)
    print("-" * (len(header) + 14))

    source = SyntheticBarSource()
    for n_assets in (8, 16, 32, 64):
        symbols = tuple(f"SYN{i:02d}" for i in range(n_assets))
        for lookback in (30, 120, 365):
            request = BarRequest(
                symbols=symbols,
                start=default_request().end.replace(year=2024),
                end=default_request().end,
            )
            # Trim to the requested number of sessions.
            prices = source.fetch(request).tail(lookback)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                _, _, _, diag = annualized_moments(to_returns(prices))
            verdict = "SINGULAR sample" if diag.is_singular_sample else "ok"
            print(
                f"{n_assets:>7} {lookback:>9} {diag.n_observations:>5} "
                f"{diag.rank:>5} {diag.condition_number:>12.3g} "
                f"{diag.shrinkage_intensity:>10.3f}  {verdict}"
            )
        print()


def run_pipeline(lookback_days: int = 120) -> None:
    settings = load_settings()
    source = resolve_source()
    print(f"Settings : {settings}")
    print(f"Source   : {source.name}")
    if source.name == "synthetic":
        print("           (no Alpaca credentials found -- using offline factor model)")

    request = default_request(DEFAULT_UNIVERSE, lookback_days=lookback_days)
    prices = source.fetch(request)
    returns = to_returns(prices)
    mu, cov, symbols, diagnostics = annualized_moments(returns)

    print(f"Window   : {request.start} .. {request.end}")
    print(f"Bars     : {prices.shape[0]} sessions x {prices.shape[1]} symbols")
    print(f"Covar    : {diagnostics.summary()}\n")

    for cap in (None, 8, 4):
        problem = PortfolioProblem(
            expected_returns=mu,
            covariance=cov,
            symbols=symbols,
            constraints=Constraints(max_positions=cap, max_weight=0.4, risk_aversion=25.0),
        )
        print(f"--- cardinality cap: {cap} ---")
        for cls in ALL_OPTIMIZERS:
            result = cls().solve(problem)
            top = np.argsort(result.weights)[::-1][: result.n_positions]
            holdings = ", ".join(
                f"{symbols[i]} {result.weights[i]:.1%}" for i in top[:5]
            )
            print(
                f"  {result.method:<16} obj={result.objective_value:>9.5f} "
                f"ret={result.expected_return:>7.2%} risk={result.expected_risk:>6.2%} "
                f"pos={result.n_positions:>2} {result.runtime_seconds * 1000:>7.1f}ms"
            )
            print(f"  {'':<16} {holdings}")
        print()


if __name__ == "__main__":
    show_conditioning()
    run_pipeline()

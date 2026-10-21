"""Realized trade analysis, and whether there is anything here to learn from.

Two questions, in order:

1. **What did the strategy actually do?** P&L, win rate, profit factor,
   expectancy, drawdown, cost drag.
2. **Is there a learning signal?** The system has *no* adaptive component — no
   online update, no reinforcement, no parameter fitting from realized outcomes.
   Before building one it is worth measuring whether it would have anything to
   learn *from*, and that has a numeric answer:

   * Does the optimizer's **expected return** predict the **realized** P&L of the
     interval it produced? (rank information coefficient)
   * Does a winning trade predict the next trade's outcome? (lag-1
     autocorrelation, P(win|win) vs P(win|loss))

   If both sit inside the noise band, "learn from our winning and losing trades"
   has no signal to learn from, and an adaptive layer would be fitting noise.

Three methodology decisions, all of which change the numbers:

* **Statistics are computed over traded intervals only.** Most intervals submit
  no order and score exactly 0.00, which is neither a win nor a loss. Averaging
  those in drags every ratio toward zero and produced a "3% win rate" in an
  earlier version of this script that was really 6 winners out of 13 trades.
* **Samples are pooled across independent seeds.** One run yields ~13 trades,
  which cannot support a correlation estimate. Each seed is a different synthetic
  universe, so the trades are independent draws.
* **The consecutive-loss halt truncates each run.** Once it engages the session
  is switched off, and the flat tail is excluded rather than averaged in.

Runs with the reviewer **bypassed**. Not a shortcut: with the real model the
reviewer abstains on every interval and submits zero orders, so the reviewed arm
contains no trades to analyze. These statistics describe the *unreviewed
optimizer*, which is the arm the ablation compares against.

Run:
    python scripts/analyze_trades.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from quantlab.data.estimators import JamesSteinEstimator
from quantlab.data.sources import BarRequest, SyntheticBarSource
from quantlab.evaluation.metrics import InsufficientObservations, max_drawdown
from quantlab.execution.broker import SimulatedBroker
from quantlab.optimizers import HeuristicOptimizer, MeanVarianceOptimizer, QuboOptimizer
from quantlab.pipeline.session import SessionConfig, SessionRunner

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

N_INTERVALS = 200
N_SEEDS = 10
STARTING_CAPITAL = 10.0

METHODS = {
    "mean_variance": MeanVarianceOptimizer,
    "heuristic": HeuristicOptimizer,
    "qubo_annealing": QuboOptimizer,
}


def load_prices(seed: int):
    symbols = tuple(f"SYN{i:02d}" for i in range(16))
    return SyntheticBarSource(seed=seed).fetch(
        BarRequest(symbols=symbols, start=date(2023, 1, 1), end=date(2026, 8, 1))
    )


@dataclass
class SeedRun:
    seed: int
    intervals: int
    effective_intervals: int
    halted_at: int | None
    traded_pnl: list[float] = field(default_factory=list)
    traded_expected: list[float] = field(default_factory=list)
    equity: list[float] = field(default_factory=list)
    runtimes: list[float] = field(default_factory=list)
    fees: float = 0.0
    slippage: float = 0.0
    orders: int = 0


def run_one(prices, method: str, seed: int) -> SeedRun:
    broker = SimulatedBroker(starting_cash_usd=STARTING_CAPITAL)
    report = SessionRunner(
        prices,
        config=SessionConfig(
            max_intervals=N_INTERVALS,
            review_enabled=False,
            starting_capital_usd=STARTING_CAPITAL,
        ),
        optimizer=METHODS[method](),
        estimator=JamesSteinEstimator(),
        broker=broker,
    ).run()

    traded = [i for i in report.intervals if i.traded()]
    return SeedRun(
        seed=seed,
        intervals=len(report.intervals),
        effective_intervals=report.effective_intervals,
        halted_at=report.halted_at_interval,
        traded_pnl=[i.realized_pnl_usd for i in traded],
        traded_expected=[i.optimizer_expected_return for i in traded],
        equity=[STARTING_CAPITAL] + [i.equity_usd for i in report.intervals],
        runtimes=[i.optimizer_runtime_ms for i in report.intervals],
        fees=float(sum(f.fee_usd for f in broker.fills)),
        slippage=float(sum(f.slippage_usd for f in broker.fills)),
        orders=int(sum(i.orders_submitted for i in report.intervals)),
    )


@dataclass
class TradeStats:
    method: str
    seeds: int
    total_intervals: int
    total_trades: int
    total_orders: int
    halted_runs: int
    mean_halt_interval: float
    total_pnl_usd: float
    mean_pnl_per_trade_usd: float
    median_pnl_per_trade_usd: float
    best_trade_usd: float
    worst_trade_usd: float
    win_rate_of_trades: float
    profit_factor: float
    expectancy_per_trade_usd: float
    gross_gains_usd: float
    gross_losses_usd: float
    fees_usd: float
    slippage_usd: float
    cost_drag_pct_of_gross_gains: float
    mean_return_per_run: float
    worst_run_return: float
    best_run_return: float
    runs_profitable: int
    mean_max_drawdown_pct: float
    mean_optimizer_runtime_ms: float


def summarize(method: str, runs: list[SeedRun]) -> tuple[TradeStats, dict]:
    pnl = np.array([p for r in runs for p in r.traded_pnl])
    expected = np.array([e for r in runs for e in r.traded_expected])
    runtimes = np.array([t for r in runs for t in r.runtimes])

    gains = float(pnl[pnl > 0].sum()) if pnl.size else 0.0
    losses = float(-pnl[pnl < 0].sum()) if pnl.size else 0.0
    fees = sum(r.fees for r in runs)
    slippage = sum(r.slippage for r in runs)

    run_returns = np.array([r.equity[-1] / STARTING_CAPITAL - 1.0 for r in runs])
    drawdowns = []
    for r in runs:
        try:
            drawdowns.append(max_drawdown(np.array(r.equity)) * 100.0)
        except InsufficientObservations:
            pass
    halted = [r for r in runs if r.halted_at is not None]

    stats = TradeStats(
        method=method,
        seeds=len(runs),
        total_intervals=sum(r.intervals for r in runs),
        total_trades=int(pnl.size),
        total_orders=sum(r.orders for r in runs),
        halted_runs=len(halted),
        mean_halt_interval=(
            float(np.mean([r.halted_at for r in halted])) if halted else float("nan")
        ),
        total_pnl_usd=float(pnl.sum()) if pnl.size else 0.0,
        mean_pnl_per_trade_usd=float(pnl.mean()) if pnl.size else 0.0,
        median_pnl_per_trade_usd=float(np.median(pnl)) if pnl.size else 0.0,
        best_trade_usd=float(pnl.max()) if pnl.size else 0.0,
        worst_trade_usd=float(pnl.min()) if pnl.size else 0.0,
        win_rate_of_trades=float((pnl > 0).mean()) if pnl.size else 0.0,
        profit_factor=(gains / losses if losses > 0 else float("inf") if gains > 0 else 0.0),
        expectancy_per_trade_usd=float(pnl.mean()) if pnl.size else 0.0,
        gross_gains_usd=gains,
        gross_losses_usd=losses,
        fees_usd=fees,
        slippage_usd=slippage,
        cost_drag_pct_of_gross_gains=(
            100.0 * (fees + slippage) / gains if gains > 0 else float("inf")
        ),
        mean_return_per_run=float(run_returns.mean()),
        worst_run_return=float(run_returns.min()),
        best_run_return=float(run_returns.max()),
        runs_profitable=int((run_returns > 0).sum()),
        mean_max_drawdown_pct=float(np.mean(drawdowns)) if drawdowns else 0.0,
        mean_optimizer_runtime_ms=float(runtimes.mean()) if runtimes.size else 0.0,
    )
    return stats, {"pnl": pnl, "expected": expected, "runs": runs}


# ---------------------------------------------------------------- learning signal


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation. Rank, not Pearson: the question is whether the optimizer
    *orders* outcomes correctly, not whether the relationship is linear."""
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    return float(
        np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]
    )


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def learning_signal(series: dict) -> dict:
    """Would an adaptive layer have anything to learn from?

    Every coefficient is reported against a ±2/sqrt(n) noise band, because a
    correlation of 0.14 on 100 observations is not distinguishable from zero and
    presenting it bare invites exactly that error.
    """
    pnl = series["pnl"]
    expected = series["expected"]
    n = int(pnl.size)
    if n < 4:
        return {"n_trades": n, "insufficient_data": True}

    # Autocorrelation is computed *within* each seed and then averaged. Pooling
    # first would create fake adjacencies between the last trade of one universe
    # and the first trade of the next.
    within: list[float] = []
    for run in series["runs"]:
        arr = np.array(run.traded_pnl)
        if arr.size >= 4:
            within.append(pearson(arr[:-1], arr[1:]))
    lag1 = float(np.mean(within)) if within else 0.0

    wins = pnl > 0
    after_win = wins[1:][wins[:-1]]
    after_loss = wins[1:][~wins[:-1]]
    p_ww = float(after_win.mean()) if after_win.size else 0.0
    p_wl = float(after_loss.mean()) if after_loss.size else 0.0

    band = 2.0 / np.sqrt(n)
    ic_spearman = spearman(expected, pnl)

    return {
        "n_trades": n,
        "insufficient_data": False,
        "ic_pearson_forecast_vs_outcome": pearson(expected, pnl),
        "ic_spearman_forecast_vs_outcome": ic_spearman,
        "lag1_autocorrelation_within_run": lag1,
        "p_win_after_win": p_ww,
        "p_win_after_loss": p_wl,
        "streak_edge": p_ww - p_wl,
        "noise_band_2se": float(band),
        "ic_exceeds_noise": bool(abs(ic_spearman) > band),
        "autocorr_exceeds_noise": bool(abs(lag1) > band),
    }


# ------------------------------------------------------------------------ figures


def plot(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    methods = list(results)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    ax = axes[0][0]
    for method in methods:
        for i, run in enumerate(results[method]["series"]["runs"]):
            ax.plot(
                run.equity,
                linewidth=0.8,
                alpha=0.45,
                label=method if i == 0 else None,
                color=f"C{methods.index(method)}",
            )
    ax.axhline(STARTING_CAPITAL, color="black", linestyle=":", linewidth=1)
    ax.set_title(f"Equity, {N_SEEDS} independent universes")
    ax.set_xlabel("interval")
    ax.set_ylabel("equity $")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    for method in methods:
        ax.hist(results[method]["series"]["pnl"], bins=30, alpha=0.55, label=method)
    ax.axvline(0, color="black", linestyle=":", linewidth=1)
    ax.set_title("Per-trade P&L (traded intervals only)")
    ax.set_xlabel("P&L $")
    ax.set_ylabel("trades")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0][2]
    width = 0.35
    positions = np.arange(len(methods))
    stats = [results[m]["stats"] for m in methods]
    ax.bar(positions - width / 2, [s["win_rate_of_trades"] * 100 for s in stats], width,
           label="win rate %")
    ax.bar(positions + width / 2, [min(s["profit_factor"], 3.0) * 100 / 3 for s in stats],
           width, label="profit factor (scaled, capped at 3)")
    ax.axhline(50, color="grey", linestyle=":", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=8)
    ax.set_title("Win rate and profit factor")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1][0]
    for method in methods:
        s = results[method]["series"]
        ax.scatter(s["expected"], s["pnl"], s=14, alpha=0.55, label=method)
    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.set_title("Forecast vs outcome — is there a signal?")
    ax.set_xlabel("optimizer expected return (annualized)")
    ax.set_ylabel("realized trade P&L $")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    ic = [results[m]["learning"].get("ic_spearman_forecast_vs_outcome", 0.0) for m in methods]
    ac = [results[m]["learning"].get("lag1_autocorrelation_within_run", 0.0) for m in methods]
    bands = [results[m]["learning"].get("noise_band_2se", 1.0) for m in methods]
    ax.bar(positions - width / 2, ic, width, label="rank IC (forecast vs outcome)")
    ax.bar(positions + width / 2, ac, width, label="lag-1 autocorrelation")
    ax.axhspan(-max(bands), max(bands), color="grey", alpha=0.25,
               label="±2 SE noise band")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=8)
    ax.set_title("Learning signal vs noise")
    ax.set_ylabel("correlation")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1][2]
    ax.bar(positions - width / 2, [s["gross_gains_usd"] for s in stats], width,
           label="gross gains")
    ax.bar(positions + width / 2, [s["gross_losses_usd"] for s in stats], width,
           label="gross losses")
    ax.plot(positions, [s["fees_usd"] + s["slippage_usd"] for s in stats], "kd--",
            label="fees + slippage", markersize=7)
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=8)
    ax.set_title("Gross P&L vs transaction costs")
    ax.set_ylabel("$")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Realized trade analysis — {N_SEEDS} universes x {N_INTERVALS} intervals, "
        f"${STARTING_CAPITAL:.2f} capital, reviewer bypassed, "
        f"statistics over traded intervals only",
        fontsize=10,
    )
    fig.tight_layout()
    path = FIGURES_DIR / "trade_analysis.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nFigure written to {path}")


def main() -> None:
    print(
        f"Trade analysis -- {N_SEEDS} universes x {N_INTERVALS} intervals, "
        f"reviewer BYPASSED\n"
    )
    price_sets = [(20260803 + s, load_prices(20260803 + s)) for s in range(N_SEEDS)]

    results: dict = {}
    for method in METHODS:
        runs = [run_one(prices, method, seed) for seed, prices in price_sets]
        stats, series = summarize(method, runs)
        results[method] = {
            "stats": asdict(stats),
            "learning": learning_signal(series),
            "series": series,
        }

        s = stats
        print(f"--- {method} ---")
        print(
            f"  {s.total_trades} trades over {s.total_intervals} intervals "
            f"({s.seeds} universes); {s.halted_runs}/{s.seeds} runs hit the "
            f"consecutive-loss halt"
            + (
                f" (mean interval {s.mean_halt_interval:.1f})"
                if s.halted_runs
                else ""
            )
        )
        print(
            f"  total P&L ${s.total_pnl_usd:+.4f}   per trade "
            f"${s.mean_pnl_per_trade_usd:+.5f} mean / "
            f"${s.median_pnl_per_trade_usd:+.5f} median"
        )
        print(
            f"  best ${s.best_trade_usd:+.4f}  worst ${s.worst_trade_usd:+.4f}  "
            f"win rate {s.win_rate_of_trades:.1%}  profit factor {s.profit_factor:.3f}"
        )
        print(
            f"  per-run return: mean {s.mean_return_per_run:+.3%}, "
            f"range {s.worst_run_return:+.3%}..{s.best_run_return:+.3%}, "
            f"{s.runs_profitable}/{s.seeds} profitable"
        )
        print(
            f"  mean max drawdown {s.mean_max_drawdown_pct:.2f}%   "
            f"costs {s.cost_drag_pct_of_gross_gains:.1f}% of gross gains"
        )
        print(f"  optimizer {s.mean_optimizer_runtime_ms:.2f} ms/interval")

        learn = results[method]["learning"]
        if learn.get("insufficient_data"):
            print(f"  LEARNING SIGNAL: only {learn['n_trades']} trades -- not enough\n")
            continue
        print(
            f"  LEARNING SIGNAL (n={learn['n_trades']} trades, "
            f"noise band +/-{learn['noise_band_2se']:.3f}):"
        )
        print(
            f"    rank IC forecast->outcome : "
            f"{learn['ic_spearman_forecast_vs_outcome']:+.4f}  "
            f"{'SIGNAL' if learn['ic_exceeds_noise'] else 'noise'}"
        )
        print(
            f"    lag-1 autocorrelation     : "
            f"{learn['lag1_autocorrelation_within_run']:+.4f}  "
            f"{'SIGNAL' if learn['autocorr_exceeds_noise'] else 'noise'}"
        )
        print(
            f"    P(win|win) {learn['p_win_after_win']:.1%} vs "
            f"P(win|loss) {learn['p_win_after_loss']:.1%}  "
            f"edge {learn['streak_edge']:+.1%}"
        )
        print()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "configuration": {
            "seeds": N_SEEDS,
            "intervals_per_seed": N_INTERVALS,
            "starting_capital_usd": STARTING_CAPITAL,
            "review_enabled": False,
            "statistics_basis": "traded intervals only",
            "notes": [
                "Reviewer bypassed: with the real model it abstains every "
                "interval and submits zero orders, so the reviewed arm has no "
                "trades to analyze.",
                "Win rate, profit factor and expectancy are computed over "
                "traded intervals. Non-traded intervals score exactly 0.00 and "
                "are neither wins nor losses.",
                "Autocorrelation is computed within each run and averaged, not "
                "on the pooled series, to avoid fake adjacencies across seeds.",
            ],
        },
        "methods": {
            m: {"stats": r["stats"], "learning": r["learning"]}
            for m, r in results.items()
        },
    }
    out = REPORTS_DIR / "trade_analysis.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Statistics written to {out}")

    plot(results)


if __name__ == "__main__":
    main()

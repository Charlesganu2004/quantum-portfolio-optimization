"""Runtime and quality as the asset universe grows -- the Project II sweep.

Produces the deliverables the proposal names: runtime comparison, scalability
curve, concurrent-position counts, and gap-to-optimal where the universe is small
enough to enumerate.

Two limits are reported rather than hidden:

* Ground truth is refused above 250,000 subsets (C(26,6) is the largest 6-subset
  search accepted). Above that, methods are only compared against each other, so
  "best" means "best of these three", not "optimal".
* QAOA simulates one qubit per asset at O(2^n) cost and is excluded above its
  qubit cap. That exclusion is printed, not silently applied.
"""

from __future__ import annotations

import json
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.data.features import annualized_moments, to_returns
from quantlab.data.sources import BarRequest, SyntheticBarSource, resolve_source
from quantlab.evaluation.ground_truth import (
    TooLargeToEnumerate,
    exhaustive_best,
    subset_count,
)
from quantlab.optimizers import (
    ALL_METHODS,
    Constraints,
    PortfolioProblem,
    QaoaNotApplicable,
)

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
UNIVERSE_SIZES = (4, 8, 16, 32, 64)
REPEATS = 3
CARDINALITY_CAP = 4
TRAIN_SESSIONS = 300  # >= 120 required; 30 would make the covariance singular.


def build_problem(n_assets: int, seed: int) -> PortfolioProblem:
    """Annualized moments from a factor-model price history."""
    source = resolve_source()
    if source.name != "synthetic":
        # Live universes are fixed size; the sweep needs arbitrary n, so the
        # synthetic generator drives the scalability axis in both cases.
        source = SyntheticBarSource(seed=seed)
    else:
        source = SyntheticBarSource(seed=seed)

    symbols = tuple(f"SYN{i:03d}" for i in range(n_assets))
    prices = source.fetch(
        BarRequest(symbols=symbols, start=date(2023, 1, 1), end=date(2026, 8, 1))
    ).tail(TRAIN_SESSIONS)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu, cov, resolved, _ = annualized_moments(to_returns(prices))

    return PortfolioProblem(
        expected_returns=mu,
        covariance=cov,
        symbols=resolved,
        constraints=Constraints(
            max_positions=CARDINALITY_CAP, max_weight=0.4, risk_aversion=25.0
        ),
    )


def run_sweep() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    skipped: list[str] = []

    for n_assets in UNIVERSE_SIZES:
        for repeat in range(REPEATS):
            problem = build_problem(n_assets, seed=1000 + repeat)

            truth = None
            try:
                if subset_count(n_assets, CARDINALITY_CAP) <= 250_000:
                    truth = exhaustive_best(problem)
            except (TooLargeToEnumerate, ValueError):
                truth = None

            for cls in ALL_METHODS:
                try:
                    result = cls().solve(problem)
                except QaoaNotApplicable as exc:
                    skipped.append(f"{cls.method} @ n={n_assets}: {exc}")
                    continue
                except ValueError as exc:
                    skipped.append(f"{cls.method} @ n={n_assets}: {exc}")
                    continue

                rows.append(
                    {
                        "n_assets": n_assets,
                        "repeat": repeat,
                        "method": result.method,
                        "runtime_ms": result.runtime_seconds * 1000.0,
                        "objective": result.objective_value,
                        "expected_return": result.expected_return,
                        "expected_risk": result.expected_risk,
                        "n_positions": result.n_positions,
                        "gap_to_optimal": (
                            truth.gap_to(result.objective_value)
                            if truth is not None
                            else np.nan
                        ),
                        # Same difference without dividing by |optimum|, which
                        # is unstable when the optimum is near zero.
                        "absolute_shortfall": (
                            truth.shortfall_to(result.objective_value)
                            if truth is not None
                            else np.nan
                        ),
                    }
                )

    if skipped:
        print("\nExclusions (reported, not silent):")
        for note in dict.fromkeys(skipped):
            print(f"  - {note}")

    return pd.DataFrame(rows)


def write_tables(frame: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REPORTS_DIR / "scalability_raw.csv", index=False)

    runtime = frame.pivot_table(
        index="n_assets", columns="method", values="runtime_ms", aggfunc="median"
    )
    positions = frame.pivot_table(
        index="n_assets", columns="method", values="n_positions", aggfunc="mean"
    )
    gap = frame.pivot_table(
        index="n_assets", columns="method", values="gap_to_optimal", aggfunc="mean"
    )
    # The mean hides single bad problems, and the percentage gap is unstable when
    # an optimum is near zero, so the worst case is reported in both forms.
    worst_gap = frame.pivot_table(
        index="n_assets", columns="method", values="gap_to_optimal", aggfunc="max"
    )
    worst_shortfall = frame.pivot_table(
        index="n_assets", columns="method", values="absolute_shortfall", aggfunc="max"
    )

    runtime.to_csv(REPORTS_DIR / "table_runtime_ms.csv")
    positions.to_csv(REPORTS_DIR / "table_positions.csv")
    gap.to_csv(REPORTS_DIR / "table_gap_to_optimal.csv")
    worst_gap.to_csv(REPORTS_DIR / "table_worst_gap_to_optimal.csv")
    worst_shortfall.to_csv(REPORTS_DIR / "table_worst_shortfall.csv")

    print("\n--- Median runtime (ms) by universe size ---")
    print(runtime.round(2).to_string())
    print("\n--- Mean concurrent positions ---")
    print(positions.round(2).to_string())
    print("\n--- Mean gap to brute-force optimum (NaN = not enumerable) ---")
    print((gap * 100).round(3).to_string())
    print("\n--- Worst gap to brute-force optimum, % of |optimum| ---")
    print((worst_gap * 100).round(3).to_string())
    print("\n--- Worst absolute shortfall, objective units ---")
    print(worst_shortfall.round(6).to_string())


def write_figures(frame: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    runtime = frame.pivot_table(
        index="n_assets", columns="method", values="runtime_ms", aggfunc="median"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for method in runtime.columns:
        series = runtime[method].dropna()
        axes[0].plot(series.index, series.to_numpy(), marker="o", label=method)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("universe size (assets)")
    axes[0].set_ylabel("median runtime (ms, log scale)")
    axes[0].set_title("Scalability: runtime vs universe size")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    risk_return = frame.groupby("method")[["expected_risk", "expected_return"]].mean()
    for method, row in risk_return.iterrows():
        axes[1].scatter(row["expected_risk"], row["expected_return"], s=90, label=method)
        axes[1].annotate(
            method,
            (row["expected_risk"], row["expected_return"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    axes[1].set_xlabel("expected risk (annualized)")
    axes[1].set_ylabel("expected return (annualized)")
    axes[1].set_title("Risk / return by method")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "scalability.png", dpi=150)
    plt.close(fig)
    print(f"\nFigures written to {REPORTS_DIR / 'scalability.png'}")


def main() -> None:
    print("Project II scalability sweep")
    print(
        f"universes={UNIVERSE_SIZES}  repeats={REPEATS}  "
        f"cap={CARDINALITY_CAP}  train_sessions={TRAIN_SESSIONS}"
    )
    print(f"methods: {[c.method for c in ALL_METHODS]}")

    frame = run_sweep()
    if frame.empty:
        raise SystemExit("sweep produced no rows")

    write_tables(frame)
    write_figures(frame)

    summary = {
        "universe_sizes": list(UNIVERSE_SIZES),
        "repeats": REPEATS,
        "cardinality_cap": CARDINALITY_CAP,
        "train_sessions": TRAIN_SESSIONS,
        "methods": sorted(frame["method"].unique().tolist()),
        "ground_truth_available_up_to": int(
            frame.loc[frame["gap_to_optimal"].notna(), "n_assets"].max()
        )
        if frame["gap_to_optimal"].notna().any()
        else None,
    }
    (REPORTS_DIR / "scalability_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Tables and summary written to {REPORTS_DIR}")


if __name__ == "__main__":
    main()

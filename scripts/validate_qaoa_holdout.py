"""Does the tuned QAOA configuration generalize, or was it fitted to the probe?

`tune_qaoa.py` selected reps=2 / penalty=5.0 because it scored 0.000% on the six
tuning cases -- while immediate neighbours in the grid scored 13% to 62%. That
pattern is what overfitting looks like, so the configuration is re-scored here on
seeds and sizes it was never selected against. The tuning cases are scored again
in this script, so the comparison uses two measured numbers.

Whatever this prints is the number that belongs in the report.

    python scripts/validate_qaoa_holdout.py [--json PATH]

By default the script prints and writes no files. With `--json PATH` it also
writes the per-case and summary results to PATH.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantlab.evaluation.ground_truth import exhaustive_best
from quantlab.optimizers import (
    Constraints,
    PortfolioProblem,
    QaoaOptimizer,
    QuboOptimizer,
)

TRADING_DAYS = 252

#: Settings under test. `max_iterations=250` is what `tune_qaoa.py` used; the
#: library default is 120.
REPS = 2
MAX_ITERATIONS = 250
PENALTY = 5.0

TUNING_CASES = [(8, 4, 1), (8, 4, 2), (8, 3, 3), (10, 3, 4), (10, 4, 5), (12, 4, 6)]
HOLDOUT_CASES = [
    (8, 3, 101), (8, 4, 102), (9, 3, 103), (9, 4, 104), (10, 3, 105),
    (10, 4, 106), (11, 3, 107), (11, 4, 108), (12, 3, 109), (12, 4, 110),
    (8, 5, 111), (10, 5, 112),
]


def problem(n: int, seed: int, cap: int) -> PortfolioProblem:
    rng = np.random.default_rng(seed)
    daily_mu = rng.normal(0.0008, 0.0006, n)
    loadings = rng.normal(0, 1, (n, 3))
    daily_cov = loadings @ loadings.T * 1e-5 + np.eye(n) * 1e-4
    return PortfolioProblem(
        expected_returns=daily_mu * TRADING_DAYS,
        covariance=(daily_cov + daily_cov.T) / 2 * TRADING_DAYS,
        symbols=[f"A{i:02d}" for i in range(n)],
        constraints=Constraints(max_positions=cap, max_weight=0.4, risk_aversion=25.0),
    )


def score(cases: list[tuple[int, int, int]], echo: bool) -> list[dict[str, float]]:
    """Gap of QAOA and of classical annealing to the exhaustive optimum, per case."""
    rows: list[dict[str, float]] = []
    for n, cap, seed in cases:
        p = problem(n, seed, cap)
        truth = exhaustive_best(p)

        quantum = QaoaOptimizer(
            reps=REPS, max_iterations=MAX_ITERATIONS, penalty_multiplier=PENALTY
        ).solve(p)
        classical = QuboOptimizer().solve(p)

        row = {
            "n": n,
            "cap": cap,
            "seed": seed,
            "qaoa_gap": truth.gap_to(quantum.objective_value),
            "qubo_gap": truth.gap_to(classical.objective_value),
            "qaoa_shortfall": truth.shortfall_to(quantum.objective_value),
            "qubo_shortfall": truth.shortfall_to(classical.objective_value),
            "feasible_mass": float(
                quantum.diagnostics.get("feasible_probability_mass", float("nan"))
            ),
            "qaoa_ms": quantum.runtime_seconds * 1000,
            "qubo_ms": classical.runtime_seconds * 1000,
        }
        rows.append(row)
        if echo:
            print(
                f"{n:>4} {cap:>4} {seed:>5} {row['qaoa_gap'] * 100:>9.3f}% "
                f"{row['qubo_gap'] * 100:>9.3f}% {row['feasible_mass']:>10.4f} "
                f"{row['qaoa_ms']:>9.1f} {row['qubo_ms']:>8.1f}"
            )
    return rows


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    qaoa = [r["qaoa_gap"] for r in rows]
    qubo = [r["qubo_gap"] for r in rows]
    return {
        "cases": len(rows),
        "qaoa_mean_gap": float(np.mean(qaoa)),
        "qaoa_worst_gap": float(np.max(qaoa)),
        "qaoa_optimal_hits": sum(1 for g in qaoa if g < 1e-9),
        "qubo_mean_gap": float(np.mean(qubo)),
        "qubo_worst_gap": float(np.max(qubo)),
        "qubo_optimal_hits": sum(1 for g in qubo if g < 1e-9),
        "qaoa_mean_ms": float(np.mean([r["qaoa_ms"] for r in rows])),
        "qubo_mean_ms": float(np.mean([r["qubo_ms"] for r in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, default=None, help="also write results here")
    args = parser.parse_args()

    assert not (set(HOLDOUT_CASES) & set(TUNING_CASES)), "holdout overlaps tuning set"

    print(
        f"QAOA holdout validation -- reps={REPS}, penalty={PENALTY}, "
        f"maxiter={MAX_ITERATIONS} (selected on other cases)\n"
    )
    header = (
        f"{'n':>4} {'cap':>4} {'seed':>5} {'qaoa gap':>10} {'qubo gap':>10} "
        f"{'feas.mass':>10} {'qaoa ms':>9} {'qubo ms':>8}"
    )
    print(header)
    print("-" * len(header))
    holdout_rows = score(HOLDOUT_CASES, echo=True)
    holdout = summarize(holdout_rows)

    print("-" * len(header))
    print(
        f"{'MEAN':>15} {holdout['qaoa_mean_gap'] * 100:>9.3f}% "
        f"{holdout['qubo_mean_gap'] * 100:>9.3f}% {'':>10} "
        f"{holdout['qaoa_mean_ms']:>9.1f} {holdout['qubo_mean_ms']:>8.1f}"
    )
    print(
        f"{'WORST':>15} {holdout['qaoa_worst_gap'] * 100:>9.3f}% "
        f"{holdout['qubo_worst_gap'] * 100:>9.3f}%"
    )
    print(f"{'OPTIMAL HITS':>15} {holdout['qaoa_optimal_hits']}/{len(HOLDOUT_CASES)} for QAOA")

    print(f"\nRe-scoring the {len(TUNING_CASES)} tuning cases with the same settings\n")
    print(header)
    print("-" * len(header))
    tuning_rows = score(TUNING_CASES, echo=True)
    tuning = summarize(tuning_rows)
    print("-" * len(header))
    print(f"{'MEAN':>15} {tuning['qaoa_mean_gap'] * 100:>9.3f}%")

    tuning_mean = tuning["qaoa_mean_gap"]
    holdout_mean = holdout["qaoa_mean_gap"]
    print()
    if holdout_mean > tuning_mean + 0.01:
        verdict = "overfit"
        print(
            f"OVERFIT CONFIRMED: {tuning_mean:.3%} on the tuning cases, "
            f"{holdout_mean:.3%} on holdout. Report the holdout number, not the "
            f"tuning number."
        )
    else:
        verdict = "generalizes"
        print(
            f"Generalizes: holdout mean {holdout_mean:.3%} is within 1 point of the "
            f"tuning mean {tuning_mean:.3%}."
        )
    print(
        f"Runtime: QAOA {holdout['qaoa_mean_ms']:.0f} ms vs classical annealing "
        f"{holdout['qubo_mean_ms']:.0f} ms "
        f"({holdout['qaoa_mean_ms'] / max(holdout['qubo_mean_ms'], 1e-9):.0f}x slower)"
    )

    if args.json is not None:
        payload = {
            "settings": {
                "reps": REPS,
                "max_iterations": MAX_ITERATIONS,
                "penalty_multiplier": PENALTY,
            },
            "verdict": verdict,
            "holdout": {"summary": holdout, "cases": holdout_rows},
            "tuning": {"summary": tuning, "cases": tuning_rows},
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()

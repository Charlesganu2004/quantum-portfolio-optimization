"""Forward paper session with per-step reports, plus the review ablation.

Produces two Project II deliverables:

  * the per-step paper-trading log (`reports/session_intervals.json`)
  * the reviewer ablation -- how often review changes the final decision

Each step advances one row of the price frame, which is one trading day for the
daily bars this script loads. The 10-minute spacing in the log's timestamps is a
label. See `quantlab.pipeline.session` for the details.

By default the script refuses to run without a reachable local model, because the
rule-based stand-in can approve a trade and so measures nothing about a language
model. Pass `--offline` to run with the stand-in anyway; the output is labelled.

A model that passes the startup check can still fail mid-run, and a failed review
abstains exactly like a judgement abstention. The script therefore stops, without
writing anything, when a model backend returns no response at any step of the main
run or of either ablation arm. `reviewer_is_language_model` in the log is true only
when every reviewed step of all three runs got a response from the model, and each
step records its review error and whether its citations verified.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from quantlab.data.estimators import JamesSteinEstimator
from quantlab.data.sources import BarRequest, SyntheticBarSource, resolve_source
from quantlab.llm.reviewer import resolve_reviewer
from quantlab.optimizers import HeuristicOptimizer, QuboOptimizer
from quantlab.pipeline.session import SessionConfig, SessionRunner, compare_review_ablation

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def load_prices():
    source = resolve_source()
    if source.name == "synthetic":
        source = SyntheticBarSource(seed=20260803)
        symbols = tuple(f"SYN{i:02d}" for i in range(16))
        note = "offline factor model (no Alpaca credentials)"
    else:
        from quantlab.data.sources import DEFAULT_UNIVERSE

        symbols = DEFAULT_UNIVERSE
        note = f"live source: {source.name}"
    prices = source.fetch(
        BarRequest(symbols=symbols, start=date(2023, 1, 1), end=date(2026, 8, 1))
    )
    return prices, note


def interval_record(i) -> dict[str, object]:
    """One step of a session as the log stores it."""
    return {
        "index": i.index,
        "method": i.method,
        "estimator": i.estimator,
        "candidate_id": i.candidate_id,
        "positions": i.optimizer_positions,
        "expected_return": i.optimizer_expected_return,
        "expected_risk": i.optimizer_expected_risk,
        "optimizer_runtime_ms": i.optimizer_runtime_ms,
        "evidence_ids": list(i.evidence_ids),
        "reviewer_verdict": i.reviewer_verdict,
        "reviewer_permits": i.reviewer_permits,
        "reviewer_error": i.reviewer_error,
        "model_responded": i.model_responded,
        "citations_verified": i.citations_verified,
        "repaired_citations": list(i.repaired_citations),
        "reviewer_observations": list(i.reviewer_observations),
        "risk_decision": i.risk_decision,
        "risk_reasons": list(i.risk_reasons),
        "orders": i.orders_submitted,
        "realized_pnl_usd": i.realized_pnl_usd,
        "equity_usd": i.equity_usd,
        "drawdown_percent": i.drawdown_percent,
        "no_trade_reason": i.no_trade_reason,
    }


def stop_on_failed_reviews(label: str, report, reviewer) -> None:
    """Exit when a model backend returned no response at some step of a run."""
    if reviewer.name == "offline_rules":
        return
    failed = [i for i in report.intervals if i.model_responded is False]
    if failed:
        steps = ", ".join(str(i.index) for i in failed)
        sys.exit(
            f"{label}: {reviewer.name} returned no response at "
            f"{len(failed)} of {len(report.intervals)} steps ({steps}); first error: "
            f"{failed[0].reviewer_error}. Those steps abstained for a transport "
            f"reason, so the run measures nothing about the model. Nothing was written."
        )


def print_intervals(report) -> None:
    header = (
        f"{'#':>3} {'method':<15} {'pos':>4} {'verdict':>9} {'risk':>8} "
        f"{'orders':>7} {'P&L $':>9} {'equity $':>9} {'DD %':>6}  reason"
    )
    print(header)
    print("-" * (len(header) + 20))
    for interval in report.intervals:
        print(
            f"{interval.index:>3} {interval.method:<15} "
            f"{interval.optimizer_positions:>4} {interval.reviewer_verdict:>9} "
            f"{interval.risk_decision:>8} {interval.orders_submitted:>7} "
            f"{interval.realized_pnl_usd:>9.4f} {interval.equity_usd:>9.4f} "
            f"{interval.drawdown_percent:>6.2f}  {interval.no_trade_reason}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--offline",
        action="store_true",
        help="allow the rule-based stand-in reviewer when no local model is reachable",
    )
    args = parser.parse_args()

    prices, note = load_prices()
    reviewer = resolve_reviewer()
    if reviewer.name == "offline_rules" and not args.offline:
        sys.exit(
            "No local model is reachable, so the only reviewer available is the "
            "rule-based stand-in, which can approve trades. "
            "Start Ollama and run `ollama pull llama3.2:3b`, or pass --offline "
            "to run with the stand-in."
        )

    print("Forward paper session -- one price-frame row per step (daily bars)")
    print(f"data     : {note}, {len(prices)} sessions x {len(prices.columns)} symbols")
    print(f"reviewer : {reviewer.name}")
    if reviewer.name == "offline_rules":
        print(
            "           (no local model reachable -- deterministic rules, NOT an LLM.\n"
            "            Start Ollama and `ollama pull llama3.2:3b` for a real reviewer.)"
        )
    print()

    config = SessionConfig(max_intervals=12, review_enabled=True)
    report = SessionRunner(
        prices,
        config=config,
        optimizer=QuboOptimizer(),
        estimator=JamesSteinEstimator(),
        reviewer=reviewer,
    ).run()
    stop_on_failed_reviews("main run", report, reviewer)

    print_intervals(report)
    print(
        f"\nstarting equity ${report.starting_equity_usd:.4f} -> "
        f"ending ${report.ending_equity_usd:.4f} "
        f"({report.total_return:+.3%}) over {len(report.intervals)} steps"
    )
    print(
        f"steps traded: {report.intervals_traded}/{len(report.intervals)}  "
        f"reviewer abstentions: {report.abstentions}"
    )

    print("\n" + "=" * 72)
    print("REVIEWER ABLATION -- does review change the decision, and how often?")
    print("=" * 72)

    ablation_results: dict[str, dict[str, object]] = {}
    reviewed_runs = [report]
    for label, optimizer in (("qubo_annealing", QuboOptimizer()), ("heuristic", HeuristicOptimizer())):
        ablation = compare_review_ablation(
            prices,
            config=SessionConfig(max_intervals=12),
            optimizer=optimizer,
            estimator=JamesSteinEstimator(),
            reviewer=reviewer,
        )
        stop_on_failed_reviews(f"{label} ablation", ablation["with_review"], reviewer)
        reviewed_runs.append(ablation["with_review"])
        print(
            f"\n{label}:"
            f"\n  intervals               : {ablation['intervals']}"
            f"\n  decisions changed       : {ablation['decisions_changed']} "
            f"({ablation['change_rate']:.1%})"
            f"\n  traded with review      : {ablation['traded_with_review']}"
            f"\n  traded without review   : {ablation['traded_without_review']}"
            f"\n  reviewer abstentions    : {ablation['abstentions']}"
            f"\n  return with review      : {ablation['return_with_review']:+.4%}"
            f"\n  return without review   : {ablation['return_without_review']:+.4%}"
        )
        ablation_results[label] = {
            key: ablation[key]
            for key in (
                "intervals",
                "decisions_changed",
                "change_rate",
                "traded_with_review",
                "traded_without_review",
                "abstentions",
                "return_with_review",
                "return_without_review",
            )
        }
        ablation_results[label]["steps_with_review"] = [
            interval_record(i) for i in ablation["with_review"].intervals
        ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_source": note,
        "reviewer": reviewer.name,
        "reviewer_is_language_model": reviewer.name != "offline_rules"
        and all(run.reviewed_by_model_every_step for run in reviewed_runs),
        "ablation": ablation_results,
        "intervals": [interval_record(i) for i in report.intervals],
    }
    (REPORTS_DIR / "session_intervals.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"\nInterval log written to {REPORTS_DIR / 'session_intervals.json'}")


if __name__ == "__main__":
    main()

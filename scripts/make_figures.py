"""Report figures built from committed result files only.

Every number plotted here is read from a file under `reports/`. Nothing is typed
in by hand, and the script does not read `docs/FINDINGS.md`.

    reports/gap_battery.json        scripts/validate_against_optimum.py --json
    reports/qaoa_holdout.json       scripts/validate_qaoa_holdout.py --json
    reports/session_intervals.json  scripts/run_session.py
    reports/trade_analysis.json     scripts/analyze_trades.py
    reports/concurrency.json        scripts/benchmark_concurrency.py

    python scripts/make_figures.py

A figure whose input file is missing is skipped with a message.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load(name: str) -> dict | None:
    path = REPORTS / name
    if not path.is_file():
        print(f"{name} missing; skipping the figure that needs it")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def figure_gap() -> bool:
    """Mean versus worst-case gap for the three classical methods, 20 problems."""
    data = load("gap_battery.json")
    if data is None:
        return False
    methods_data = data["methods"]
    methods = list(methods_data)
    labels = [m.replace("_", " ") for m in methods]
    mean = {m: methods_data[m]["mean_gap"] * 100 for m in methods}
    worst = {m: methods_data[m]["worst_gap"] * 100 for m in methods}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    positions = np.arange(len(methods))
    width = 0.36

    ax = axes[0]
    ax.bar(positions - width / 2, [mean[m] for m in methods], width,
           label="mean gap", color="#4C78A8")
    ax.bar(positions + width / 2, [worst[m] for m in methods], width,
           label="worst-case gap", color="#B03020")
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("gap to brute-force optimum (%, log scale)")
    ax.set_title("Mean gap hides the tail")
    for i, m in enumerate(methods):
        ax.text(i + width / 2, worst[m] * 1.15, f"{worst[m]:.2f}%",
                ha="center", fontsize=8, color="#B03020")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y", which="both")

    ax = axes[1]
    ratio = [worst[m] / mean[m] for m in methods]
    ax.bar(labels, ratio, color="#7A5195")
    ax.set_ylabel("worst gap / mean gap")
    ax.set_title("Tail amplification: how much worse is the worst case?")
    for i, r in enumerate(ratio):
        ax.text(i, r + 0.4, f"{r:.1f}x", ha="center", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="x", labelsize=9)

    fig.suptitle(
        f"Gap to exhaustive ground truth: {data['problems']} problems, "
        f"12 to 20 assets, classical methods only",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "optimality_gap.png", dpi=150)
    plt.close(fig)
    return True


def figure_qaoa_holdout() -> bool:
    """QAOA against QUBO annealing on the held-out problems, in its own figure.

    QAOA was not run on the 20-problem battery: 15 of its problems have more
    assets than the 14-qubit cap, and the five 12-asset problems were skipped for
    runtime. It is therefore never drawn beside those results.
    """
    data = load("qaoa_holdout.json")
    if data is None:
        return False
    summary = data["holdout"]["summary"]
    cases = summary["cases"]
    sizes = [(r["n"]) for r in data["holdout"]["cases"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    labels = ["QAOA (simulated)", "QUBO annealing"]
    mean = [summary["qaoa_mean_gap"] * 100, summary["qubo_mean_gap"] * 100]
    worst = [summary["qaoa_worst_gap"] * 100, summary["qubo_worst_gap"] * 100]
    positions = np.arange(2)
    width = 0.36

    ax = axes[0]
    ax.bar(positions - width / 2, mean, width, label="mean gap", color="#4C78A8")
    ax.bar(positions + width / 2, worst, width, label="worst-case gap", color="#B03020")
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("gap to brute-force optimum (%, log scale)")
    ax.set_title("Gap on the held-out problems")
    for i in range(2):
        ax.text(i - width / 2, mean[i] * 1.15, f"{mean[i]:.2f}%", ha="center", fontsize=8)
        ax.text(i + width / 2, worst[i] * 1.15, f"{worst[i]:.2f}%", ha="center",
                fontsize=8, color="#B03020")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y", which="both")

    ax = axes[1]
    hits = [summary["qaoa_optimal_hits"], summary["qubo_optimal_hits"]]
    ax.bar(labels, hits, color=["#B03020", "#2E7D32"])
    ax.set_ylim(0, cases + 1)
    ax.set_ylabel(f"problems solved to the optimum (of {cases})")
    ax.set_title("Optimal hits")
    for i, h in enumerate(hits):
        ax.text(i, h + 0.2, f"{h} / {cases}", ha="center", fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"{cases} holdout problems, {min(sizes)} to {max(sizes)} assets "
        f"(not used to choose QAOA's settings)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "qaoa_holdout.png", dpi=150)
    plt.close(fig)
    return True


def figure_ablation() -> bool:
    """Change rate against what abstaining actually did to the session return."""
    data = load("session_intervals.json")
    if data is None or "ablation" not in data:
        print("session_intervals.json has no ablation block; rerun run_session.py")
        return False
    ablation = data["ablation"]
    methods = list(ablation)
    labels = [m.replace("_", " ") for m in methods]
    steps = ablation[methods[0]]["intervals"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    changed = [ablation[m]["change_rate"] * 100 for m in methods]
    ax.bar(labels, changed, color="#4C78A8")
    ax.set_ylim(0, 115)
    ax.set_ylabel("decisions changed by review (%)")
    ax.set_title("Change rate")
    for i, value in enumerate(changed):
        ax.text(i, value + 3, f"{value:.1f}%", ha="center", fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    without = [ablation[m]["return_without_review"] * 100 for m in methods]
    with_review = [ablation[m]["return_with_review"] * 100 for m in methods]
    positions = np.arange(len(methods))
    width = 0.36
    ax.bar(positions - width / 2, without, width, label="review bypassed",
           color=["#2E7D32" if r > 0 else "#B03020" for r in without])
    ax.bar(positions + width / 2, with_review, width, label="reviewed",
           color="#999999")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylim(min(0.0, *without, *with_review) - 0.4, max(0.0, *without, *with_review) + 0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"return over {steps} steps (%)")
    ax.set_title("Session return with and without review")
    for i in range(len(methods)):
        ax.text(i - width / 2, without[i] + (0.08 if without[i] > 0 else -0.16),
                f"{without[i]:+.4f}%", ha="center", fontweight="bold", fontsize=9)
        ax.text(i + width / 2, with_review[i] + 0.08, f"{with_review[i]:+.4f}%",
                ha="center", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    reviewer = data.get("reviewer", "unknown")
    # The flag only means "the model answered every step" in a log that records
    # each step of both arms. An older log without that record cannot show it.
    step_log = all("steps_with_review" in arm for arm in ablation.values())
    answered = (
        f"model answered every step: {data.get('reviewer_is_language_model')}"
        if step_log
        else "per-step responses of the arms not logged"
    )
    fig.suptitle(f"Reviewer ablation, reviewer: {reviewer} ({answered})", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "ablation.png", dpi=150)
    plt.close(fig)
    return True


def figure_halt() -> bool:
    """The latched consecutive-loss halt: how much of each run it switches off."""
    data = load("trade_analysis.json")
    if data is None:
        return False
    data = data["methods"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    methods = list(data)
    labels = [m.replace("_", " ") for m in methods]
    positions = np.arange(len(methods))

    ax = axes[0]
    halt = [data[m]["stats"]["mean_halt_interval"] for m in methods]
    total = [data[m]["stats"]["total_intervals"] / data[m]["stats"]["seeds"]
             for m in methods]
    ax.bar(positions, total, 0.5, label="steps configured", color="#CCCCCC")
    ax.bar(positions, halt, 0.5, label="steps before halt engaged",
           color="#B03020")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("steps")
    ax.set_title("The session switched itself off early, every run")
    for i, h in enumerate(halt):
        ax.text(i, h + 4, f"{h:.1f}", ha="center", fontweight="bold", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    win = [data[m]["stats"]["win_rate_of_trades"] * 100 for m in methods]
    diluted = [
        100.0 * data[m]["stats"]["win_rate_of_trades"]
        * data[m]["stats"]["total_trades"] / data[m]["stats"]["total_intervals"]
        for m in methods
    ]
    width = 0.36
    ax.bar(positions - width / 2, diluted, width,
           label="over all steps (wrong)", color="#B03020")
    ax.bar(positions + width / 2, win, width,
           label="over traded steps (correct)", color="#2E7D32")
    ax.axhline(50, color="black", linestyle=":", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("win rate (%)")
    ax.set_title("Statistical basis changes the answer entirely")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "The latched consecutive-loss halt and its effect on every downstream statistic",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "halt_latch.png", dpi=150)
    plt.close(fig)
    return True


def figure_scaling() -> bool:
    """Latency scaling with a fitted growth exponent per method."""
    data = load("concurrency.json")
    if data is None:
        return False
    rows = data["latency_by_size"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in sorted({r["method"] for r in rows}):
        subset = sorted((r for r in rows if r["method"] == method),
                        key=lambda r: r["n_assets"])
        n = np.array([r["n_assets"] for r in subset], dtype=float)
        ms = np.array([r["median_ms"] for r in subset], dtype=float)
        slope = float(np.polyfit(np.log(n), np.log(ms), 1)[0])
        ax.plot(n, ms, "-o", markersize=5,
                label=f'{method.replace("_", " ")}  (n^{slope:.2f})')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("universe size, assets (log)")
    ax.set_ylabel("median solve time, ms (log)")
    ax.set_title("Empirical scaling, fitted exponent on a log-log fit")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "scaling.png", dpi=150)
    plt.close(fig)
    return True


if __name__ == "__main__":
    written = [
        ("optimality_gap", figure_gap()),
        ("qaoa_holdout", figure_qaoa_holdout()),
        ("ablation", figure_ablation()),
        ("halt_latch", figure_halt()),
        ("scaling", figure_scaling()),
    ]
    for name, ok in written:
        if ok:
            print(f"wrote {FIGURES / (name + '.png')}")

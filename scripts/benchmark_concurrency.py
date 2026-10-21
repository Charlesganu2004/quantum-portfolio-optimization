"""Does this workload parallelize? Measured, not assumed.

The project's optimizers were written single-threaded. This script measures what
actually happens when you try to run many of them at once, three ways:

  * **serial**    -- one after another, the baseline
  * **threads**   -- `ThreadPoolExecutor`, shares one interpreter and one GIL
  * **processes** -- `ProcessPoolExecutor`, separate interpreters, real cores

The interesting result is the *gap between threads and processes*, because it
measures the GIL directly. A method whose inner loop is pure Python cannot use
threads at all -- the GIL serializes it, and thread "parallelism" buys nothing
but overhead. A method that spends its time inside NumPy/BLAS *can* use threads,
because those routines release the GIL while they compute.

Three measurement hazards, all handled rather than ignored:

1. **NumPy's BLAS is already threaded.** Left alone, a process pool of 8 would
   spawn 8 processes each trying to use 8 BLAS threads -- 64 threads on 8 cores,
   and the oversubscription shows up as a *slowdown* that looks like a parallelism
   failure but is really a configuration error. Workers pin BLAS to one thread.
2. **Process startup is not free.** On Windows, `spawn` re-imports the module in
   every worker. That cost is real and is reported as part of wall time rather
   than subtracted out, because a user waiting for a result pays it.
3. **A machine that sleeps mid-run corrupts a wall time.** A laptop that suspends
   during a pool run records the suspended time as run time, and every ratio built
   on that wall time is meaningless: an earlier committed run recorded a serial
   pass of 5147.8 s for work that takes under 10 s. The process asks Windows to
   stay awake while it runs, and `validate_run` refuses to write a result whose
   speedups exceed the worker count or whose serial times disagree between the
   fine-grained and the coarse-grained run.

Run:
    python scripts/benchmark_concurrency.py
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

# Pin BLAS before NumPy is imported anywhere. Must precede the numpy import.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

from quantlab.optimizers import (  # noqa: E402
    Constraints,
    HeuristicOptimizer,
    MeanVarianceOptimizer,
    PortfolioProblem,
    QuboOptimizer,
)

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

#: Methods under test, keyed by the name passed across the process boundary.
#: Classes are looked up in the worker rather than pickled, so the payload stays
#: small and the worker cannot drift from the parent's definition.
METHODS = {
    "mean_variance": MeanVarianceOptimizer,
    "heuristic": HeuristicOptimizer,
    "qubo_annealing": QuboOptimizer,
}

N_ASSETS = 24
N_PROBLEMS = 48
WORKER_COUNTS = (1, 2, 4, 8)

#: The coarse-grained run solves this many tasks of this many problems each.
COARSE_BATCH_SIZE = 24
COARSE_BATCHES = 8

#: A speedup passes the worker count only through timing noise. Beyond this
#: factor the wall time is corrupted, not measured.
SPEEDUP_SLACK = 1.10

#: Serial time per problem in the coarse run and in the fine-grained run covers
#: the same kind of problem, so the two may differ by noise but not by more than
#: this factor in either direction.
SERIAL_RATIO_LIMIT = 3.0

#: Windows `SetThreadExecutionState` flags: keep the system awake until cleared.
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def make_problems(count: int, n_assets: int, seed: int = 20260806) -> list[PortfolioProblem]:
    """Independent, well-conditioned problems. Same set for every mode.

    Built once in the parent and reused across all modes, so a mode is never
    measured against easier inputs than another.
    """
    rng = np.random.default_rng(seed)
    problems: list[PortfolioProblem] = []
    for _ in range(count):
        loadings = rng.normal(0, 1, (n_assets, 4))
        idiosyncratic = np.diag(rng.uniform(0.01, 0.05, n_assets))
        covariance = loadings @ loadings.T * 0.01 + idiosyncratic
        covariance = (covariance + covariance.T) / 2.0
        problems.append(
            PortfolioProblem(
                expected_returns=rng.normal(0.08, 0.05, n_assets),
                covariance=covariance,
                symbols=[f"A{i:02d}" for i in range(n_assets)],
                constraints=Constraints(
                    max_positions=6, max_weight=0.4, risk_aversion=25.0
                ),
            )
        )
    return problems


def solve_one(payload: tuple[str, PortfolioProblem]) -> float:
    """Top-level so it survives pickling to a spawned worker on Windows."""
    method_name, problem = payload
    result = METHODS[method_name]().solve(problem)
    return result.runtime_seconds


def solve_batch(payload: tuple[str, int, int]) -> float:
    """One coarse task: generate and solve a whole batch of problems.

    The fine-grained benchmark dispatches one *problem* per task, which is the
    obvious thing to try and the wrong granularity -- a 3 ms problem cannot repay
    a 300 ms process spawn. This function is the same total work chopped into
    units big enough to be worth shipping to another core, which is how a sweep
    or a multi-seed backtest would actually be parallelized.

    Problems are generated *inside* the worker so the measurement includes no
    pickling of large covariance matrices across the process boundary.
    """
    method_name, batch_size, seed = payload
    problems = make_problems(batch_size, N_ASSETS, seed=seed)
    start = time.perf_counter()
    for problem in problems:
        METHODS[method_name]().solve(problem)
    return time.perf_counter() - start


@dataclass
class Measurement:
    method: str
    mode: str
    workers: int
    wall_seconds: float
    throughput_per_second: float
    speedup_vs_serial: float
    efficiency: float
    cpu_seconds_in_solver: float

    def as_dict(self) -> dict:
        return asdict(self)


def time_mode(
    method: str, problems: list[PortfolioProblem], mode: str, workers: int
) -> tuple[float, float]:
    """Wall seconds and summed in-solver seconds for one configuration."""
    payloads = [(method, problem) for problem in problems]

    start = time.perf_counter()
    if mode == "serial":
        runtimes = [solve_one(p) for p in payloads]
    elif mode == "threads":
        with ThreadPoolExecutor(max_workers=workers) as pool:
            runtimes = list(pool.map(solve_one, payloads))
    elif mode == "processes":
        with ProcessPoolExecutor(max_workers=workers) as pool:
            runtimes = list(pool.map(solve_one, payloads, chunksize=4))
    else:
        raise ValueError(f"unknown mode {mode!r}")
    wall = time.perf_counter() - start
    return wall, float(sum(runtimes))


def run_benchmark() -> list[Measurement]:
    problems = make_problems(N_PROBLEMS, N_ASSETS)
    measurements: list[Measurement] = []

    for method in METHODS:
        # Warm up so first-call import and allocation costs do not land on the
        # serial baseline and inflate every speedup that is divided by it.
        solve_one((method, problems[0]))

        serial_wall, serial_cpu = time_mode(method, problems, "serial", 1)
        measurements.append(
            Measurement(
                method=method,
                mode="serial",
                workers=1,
                wall_seconds=serial_wall,
                throughput_per_second=N_PROBLEMS / serial_wall,
                speedup_vs_serial=1.0,
                efficiency=1.0,
                cpu_seconds_in_solver=serial_cpu,
            )
        )
        print(
            f"{method:<16} serial      1w  {serial_wall:7.3f}s  "
            f"{N_PROBLEMS / serial_wall:8.2f}/s"
        )

        for mode in ("threads", "processes"):
            for workers in WORKER_COUNTS:
                wall, cpu = time_mode(method, problems, mode, workers)
                speedup = serial_wall / wall if wall > 0 else 0.0
                measurements.append(
                    Measurement(
                        method=method,
                        mode=mode,
                        workers=workers,
                        wall_seconds=wall,
                        throughput_per_second=N_PROBLEMS / wall,
                        speedup_vs_serial=speedup,
                        efficiency=speedup / workers,
                        cpu_seconds_in_solver=cpu,
                    )
                )
                print(
                    f"{method:<16} {mode:<11} {workers}w  {wall:7.3f}s  "
                    f"{N_PROBLEMS / wall:8.2f}/s  speedup {speedup:5.2f}x  "
                    f"efficiency {speedup / workers:5.1%}"
                )
        print()

    return measurements


def coarse_grained_benchmark(
    batch_size: int = COARSE_BATCH_SIZE, n_batches: int = COARSE_BATCHES
) -> list[dict]:
    """The same question at a granularity that can actually repay the overhead.

    Each task is `batch_size` problems instead of one. If the fine-grained result
    is an overhead artefact rather than a property of the workload, speedup
    appears here -- and if it does not, the workload genuinely does not
    parallelize and no amount of restructuring will change that.
    """
    rows: list[dict] = []
    print(
        f"\nCoarse-grained: {n_batches} tasks x {batch_size} problems "
        f"(vs 1 problem per task above)\n"
    )
    for method in METHODS:
        payloads = [(method, batch_size, 20260806 + i) for i in range(n_batches)]

        start = time.perf_counter()
        for payload in payloads:
            solve_batch(payload)
        serial_wall = time.perf_counter() - start

        rows.append(
            {
                "method": method,
                "mode": "serial",
                "workers": 1,
                "wall_seconds": serial_wall,
                "speedup_vs_serial": 1.0,
                "efficiency": 1.0,
            }
        )
        print(f"{method:<16} serial      1w  {serial_wall:7.3f}s")

        for workers in (2, 4, 8):
            start = time.perf_counter()
            with ProcessPoolExecutor(max_workers=workers) as pool:
                list(pool.map(solve_batch, payloads))
            wall = time.perf_counter() - start
            speedup = serial_wall / wall if wall > 0 else 0.0
            rows.append(
                {
                    "method": method,
                    "mode": "processes",
                    "workers": workers,
                    "wall_seconds": wall,
                    "speedup_vs_serial": speedup,
                    "efficiency": speedup / workers,
                }
            )
            print(
                f"{method:<16} processes  {workers}w  {wall:7.3f}s  "
                f"speedup {speedup:5.2f}x  efficiency {speedup / workers:5.1%}"
            )
        print()
    return rows


def latency_by_size() -> list[dict]:
    """Single-problem latency as the universe grows. Scaling, not throughput."""
    rows: list[dict] = []
    for n_assets in (8, 16, 24, 32, 48, 64):
        problems = make_problems(5, n_assets)
        for method in METHODS:
            times = []
            for problem in problems:
                start = time.perf_counter()
                METHODS[method]().solve(problem)
                times.append(time.perf_counter() - start)
            rows.append(
                {
                    "method": method,
                    "n_assets": n_assets,
                    "median_ms": float(np.median(times) * 1000.0),
                    "mean_ms": float(np.mean(times) * 1000.0),
                    "max_ms": float(np.max(times) * 1000.0),
                }
            )
            print(
                f"{method:<16} n={n_assets:>3}  median {np.median(times) * 1000:8.2f}ms"
            )
    return rows


def plot(
    measurements: list[Measurement], latency: list[dict], coarse: list[dict]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    methods = list(METHODS)

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))

    # --- speedup curves -------------------------------------------------------
    ax = axes[0]
    for method in methods:
        for mode, style in (("threads", "--o"), ("processes", "-s")):
            xs = [m.workers for m in measurements if m.method == method and m.mode == mode]
            ys = [
                m.speedup_vs_serial
                for m in measurements
                if m.method == method and m.mode == mode
            ]
            ax.plot(xs, ys, style, label=f"{method} / {mode}", alpha=0.85, markersize=5)
    ax.plot(WORKER_COUNTS, WORKER_COUNTS, ":", color="black", label="ideal linear")
    ax.set_xlabel("workers")
    ax.set_ylabel("speedup vs serial")
    ax.set_title("Speedup: threads cannot beat the GIL")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # --- threads vs processes at 8 workers ------------------------------------
    ax = axes[1]
    width = 0.35
    positions = np.arange(len(methods))
    for offset, mode in ((-width / 2, "threads"), (width / 2, "processes")):
        values = [
            next(
                m.speedup_vs_serial
                for m in measurements
                if m.method == method and m.mode == mode and m.workers == 8
            )
            for method in methods
        ]
        ax.bar(positions + offset, values, width, label=mode)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("speedup vs serial")
    ax.set_title("8 workers: the GIL gap")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # --- latency scaling ------------------------------------------------------
    ax = axes[2]
    for method in methods:
        rows = [r for r in latency if r["method"] == method]
        ax.plot(
            [r["n_assets"] for r in rows],
            [r["median_ms"] for r in rows],
            "-o",
            label=method,
            markersize=4,
        )
    ax.set_yscale("log")
    ax.set_xlabel("universe size (assets)")
    ax.set_ylabel("median solve time (ms, log)")
    ax.set_title("Single-problem latency scaling")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # --- granularity: fine vs coarse -----------------------------------------
    ax = axes[3]
    width = 0.35
    positions = np.arange(len(methods))
    fine_8 = [
        next(
            m.speedup_vs_serial
            for m in measurements
            if m.method == method and m.mode == "processes" and m.workers == 8
        )
        for method in methods
    ]
    coarse_8 = [
        next(
            r["speedup_vs_serial"]
            for r in coarse
            if r["method"] == method and r["mode"] == "processes" and r["workers"] == 8
        )
        for method in methods
    ]
    ax.bar(positions - width / 2, fine_8, width, label="1 problem / task")
    ax.bar(positions + width / 2, coarse_8, width, label="24 problems / task")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1, label="serial baseline")
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("speedup vs serial")
    ax.set_title("8 processes: granularity is the whole story")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Concurrency — {platform.processor() or platform.machine()}, "
        f"{os.cpu_count()} logical cores, Python {platform.python_version()}, "
        f"BLAS pinned to 1 thread",
        fontsize=9,
    )
    fig.tight_layout()
    path = FIGURES_DIR / "concurrency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nFigure written to {path}")


def validate_run(measurements: list[Measurement], coarse: list[dict]) -> list[str]:
    """Reasons to distrust a run; an empty list means the timings are consistent.

    Two checks, both aimed at a wall time stretched by a machine that slept or
    stalled. A speedup above the worker count is impossible, so it can only come
    from a corrupted serial or parallel time. The serial time per problem in the
    coarse run must also agree with the fine-grained serial time per problem,
    because both solve the same kind of problem on the same machine.
    """
    issues: list[str] = []

    rows = [(m.method, m.mode, m.workers, m.speedup_vs_serial) for m in measurements]
    rows += [
        (r["method"], r["mode"], r["workers"], r["speedup_vs_serial"]) for r in coarse
    ]
    for method, mode, workers, speedup in rows:
        if speedup > workers * SPEEDUP_SLACK:
            issues.append(
                f"{method} / {mode} / {workers} workers: speedup {speedup:.2f}x "
                f"exceeds the {workers} workers that produced it"
            )

    fine_per_problem = {
        m.method: m.wall_seconds / N_PROBLEMS
        for m in measurements
        if m.mode == "serial"
    }
    for row in coarse:
        if row["mode"] != "serial":
            continue
        coarse_per_problem = row["wall_seconds"] / (COARSE_BATCHES * COARSE_BATCH_SIZE)
        ratio = coarse_per_problem / fine_per_problem[row["method"]]
        if not 1.0 / SERIAL_RATIO_LIMIT <= ratio <= SERIAL_RATIO_LIMIT:
            issues.append(
                f"{row['method']}: coarse serial run took "
                f"{coarse_per_problem * 1000:.1f} ms per problem against "
                f"{fine_per_problem[row['method']] * 1000:.1f} ms in the "
                f"fine-grained run (ratio {ratio:.2f}, limit {SERIAL_RATIO_LIMIT:g})"
            )
    return issues


def keep_system_awake() -> None:
    """Ask Windows not to suspend while this process runs. No-op elsewhere.

    The request belongs to the calling thread and ends with the process; it does
    not change any power setting.
    """
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(
            _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
        )


def main() -> None:
    keep_system_awake()
    print("Concurrency benchmark")
    print(f"  host       : {os.cpu_count()} logical cores, {platform.machine()}")
    print(f"  python     : {platform.python_version()} ({platform.python_implementation()})")
    print(f"  GIL        : present (no free-threading build)")
    print(f"  BLAS       : pinned to 1 thread per worker to prevent oversubscription")
    print(f"  workload   : {N_PROBLEMS} independent problems x {N_ASSETS} assets\n")

    measurements = run_benchmark()
    coarse = coarse_grained_benchmark()

    issues = validate_run(measurements, coarse)
    if issues:
        raise SystemExit(
            "Run rejected, nothing written. Re-run on an idle machine.\n  "
            + "\n  ".join(issues)
        )

    print("Single-problem latency by universe size\n")
    latency = latency_by_size()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": {
            "logical_cores": os.cpu_count(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "blas_threads_per_worker": 1,
        },
        "workload": {"n_problems": N_PROBLEMS, "n_assets": N_ASSETS},
        "measurements": [m.as_dict() for m in measurements],
        "coarse_grained": coarse,
        "latency_by_size": latency,
    }
    out = REPORTS_DIR / "concurrency.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nMeasurements written to {out}")

    plot(measurements, latency, coarse)


if __name__ == "__main__":
    # Required on Windows: spawned workers re-import this module, and without the
    # guard each one would re-run the benchmark and fork-bomb the machine.
    main()

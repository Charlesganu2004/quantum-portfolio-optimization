"""The concurrency benchmark refuses a run whose timings cannot be real.

An earlier committed run recorded a serial pass of 5147.8 s for work that takes
under 10 s, and speedups of 565.9x on 2 workers, because the machine slept while
it ran. `validate_run` is what keeps that from being written to `reports/`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def load_benchmark_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_concurrency.py"
    spec = importlib.util.spec_from_file_location("benchmark_concurrency_script", path)
    module = importlib.util.module_from_spec(spec)
    # The script's dataclass resolves its annotations through `sys.modules`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = load_benchmark_script()


def measurement(method: str, mode: str, workers: int, wall: float, serial: float):
    speedup = serial / wall
    return bench.Measurement(
        method=method,
        mode=mode,
        workers=workers,
        wall_seconds=wall,
        throughput_per_second=bench.N_PROBLEMS / wall,
        speedup_vs_serial=speedup,
        efficiency=speedup / workers,
        cpu_seconds_in_solver=wall,
    )


def coarse_row(mode: str, workers: int, wall: float, serial: float) -> dict:
    return {
        "method": "heuristic",
        "mode": mode,
        "workers": workers,
        "wall_seconds": wall,
        "speedup_vs_serial": serial / wall,
        "efficiency": serial / wall / workers,
    }


#: Fine-grained serial pass of 48 problems at 0.15 s each, a coarse pass of 192
#: problems at the same rate.
FINE_SERIAL = 7.2
COARSE_SERIAL = 28.8


def consistent_run():
    measurements = [
        measurement("heuristic", "serial", 1, FINE_SERIAL, FINE_SERIAL),
        measurement("heuristic", "processes", 4, 3.0, FINE_SERIAL),
        measurement("heuristic", "threads", 8, 7.0, FINE_SERIAL),
    ]
    coarse = [
        coarse_row("serial", 1, COARSE_SERIAL, COARSE_SERIAL),
        coarse_row("processes", 2, 15.0, COARSE_SERIAL),
        coarse_row("processes", 8, 6.0, COARSE_SERIAL),
    ]
    return measurements, coarse


def test_a_consistent_run_is_accepted() -> None:
    assert bench.validate_run(*consistent_run()) == []


def test_a_speedup_beyond_the_worker_count_is_rejected() -> None:
    measurements, coarse = consistent_run()
    coarse[1] = coarse_row("processes", 2, 0.05, COARSE_SERIAL)
    issues = bench.validate_run(measurements, coarse)
    assert len(issues) == 1
    assert "2 workers" in issues[0]
    assert "exceeds" in issues[0]


def test_a_fine_grained_speedup_beyond_the_worker_count_is_rejected() -> None:
    measurements, coarse = consistent_run()
    measurements[1] = measurement("heuristic", "processes", 4, 0.01, FINE_SERIAL)
    issues = bench.validate_run(measurements, coarse)
    assert any("heuristic / processes / 4 workers" in issue for issue in issues)


def test_a_speedup_within_the_noise_slack_is_accepted() -> None:
    measurements, coarse = consistent_run()
    wall = COARSE_SERIAL / (2 * (bench.SPEEDUP_SLACK - 0.01))
    coarse[1] = coarse_row("processes", 2, wall, COARSE_SERIAL)
    assert bench.validate_run(measurements, coarse) == []


@pytest.mark.parametrize("coarse_serial", [5147.8, 0.5])
def test_a_coarse_serial_time_that_disagrees_with_the_fine_run_is_rejected(
    coarse_serial: float,
) -> None:
    measurements, coarse = consistent_run()
    coarse[0] = coarse_row("serial", 1, coarse_serial, coarse_serial)
    coarse[1] = coarse_row("processes", 2, coarse_serial * 0.6, coarse_serial)
    coarse[2] = coarse_row("processes", 8, coarse_serial * 0.3, coarse_serial)
    issues = bench.validate_run(measurements, coarse)
    assert len(issues) == 1
    assert "coarse serial run" in issues[0]
    assert "heuristic" in issues[0]


def test_the_corrupted_committed_run_would_have_been_rejected() -> None:
    """Heuristic serial 5147.8 s with 2 and 4 workers at 9.1 s and 7.8 s."""
    measurements, _ = consistent_run()
    coarse = [
        coarse_row("serial", 1, 5147.8, 5147.8),
        coarse_row("processes", 2, 9.1, 5147.8),
        coarse_row("processes", 4, 7.8, 5147.8),
    ]
    issues = bench.validate_run(measurements, coarse)
    assert len(issues) == 3

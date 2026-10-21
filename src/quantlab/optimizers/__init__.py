"""Portfolio optimization methods compared in Project II.

Two registries, because the methods do not have the same operating range:

* `ALL_OPTIMIZERS` -- the three classical methods. Always available, scale to any
  universe size, and carry the shared contract tests and the scalability sweep.
* `ALL_METHODS` -- adds the simulated-quantum QAOA path when the optional
  `quantum` extra is installed. QAOA simulates one qubit per asset at O(2^n)
  cost, so it is confined to small universes and is excluded from the sweep by
  design rather than by omission.

Keeping them separate stops an optional, size-limited dependency from silently
constraining the core benchmark.
"""

from __future__ import annotations

from .base import Constraints, Optimizer, OptimizerResult, PortfolioProblem
from .heuristic import HeuristicOptimizer
from .mean_variance import MeanVarianceOptimizer
from .qaoa_qiskit import MAX_QUBITS, QaoaNotApplicable, QaoaOptimizer, qiskit_available
from .qubo import QuboOptimizer

#: Classical methods: always runnable, any universe size.
ALL_OPTIMIZERS: list[type[Optimizer]] = [
    MeanVarianceOptimizer,
    HeuristicOptimizer,
    QuboOptimizer,
]

#: Full Project II comparison, including the optional simulated-quantum path.
ALL_METHODS: list[type[Optimizer]] = [*ALL_OPTIMIZERS]
if qiskit_available():
    ALL_METHODS.append(QaoaOptimizer)

__all__ = [
    "ALL_METHODS",
    "ALL_OPTIMIZERS",
    "Constraints",
    "HeuristicOptimizer",
    "MAX_QUBITS",
    "MeanVarianceOptimizer",
    "Optimizer",
    "OptimizerResult",
    "PortfolioProblem",
    "QaoaNotApplicable",
    "QaoaOptimizer",
    "QuboOptimizer",
    "qiskit_available",
]

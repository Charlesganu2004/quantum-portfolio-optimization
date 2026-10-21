"""Turn price bars into the annualized moments every optimizer consumes.

Two decisions here shape every downstream result, so both are explicit rather
than buried:

**Annualization.** Daily returns are ~8e-4 and daily variances ~1e-4, so with
lambda near 1 the risk term is three orders of magnitude smaller than the return
term and the objective silently degenerates into "buy the highest-return names."
Scaling by 252 puts lambda back in the conventional 1-10 range where risk
actually competes. Position counts move from 3 to 29 across lambda 1 to 100, so
any cardinality result is meaningless unless lambda is reported beside it.

**Shrinkage.** A sample covariance estimated from T observations of N assets has
rank at most T-1 after demeaning. A 30-session lookback (29 return observations,
rank at most 28) against a 32 or 64 asset universe gives a *singular* matrix --
the QP would fail or return nonsense. Ledoit-Wolf shrinkage
toward a scaled identity fixes the conditioning and is applied by default.
`covariance_diagnostics` reports rank and condition number so the report can
state what the estimate actually rests on.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


class InsufficientHistory(ValueError):
    """Raised when the sample cannot support a covariance estimate at all."""


@dataclass(frozen=True)
class CovarianceDiagnostics:
    """What the covariance estimate actually rests on."""

    n_assets: int
    n_observations: int
    rank: int
    condition_number: float
    shrinkage_intensity: float
    is_singular_sample: bool

    def summary(self) -> str:
        verdict = (
            "sample covariance is rank-deficient; shrinkage is load-bearing"
            if self.is_singular_sample
            else "sample covariance is full rank"
        )
        return (
            f"{self.n_assets} assets from {self.n_observations} observations -- "
            f"rank {self.rank}, condition {self.condition_number:.3g}, "
            f"shrinkage {self.shrinkage_intensity:.3f}. {verdict}."
        )


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from a wide price frame (columns = symbols)."""
    if prices.shape[0] < 2:
        raise InsufficientHistory(
            f"need at least 2 price rows to compute returns, got {prices.shape[0]}"
        )
    return prices.sort_index().pct_change().dropna(how="all")


def ledoit_wolf_shrinkage(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Shrink the sample covariance toward a scaled identity.

    Implements the Ledoit-Wolf estimator directly rather than pulling in
    scikit-learn, keeping the dependency surface small and the estimator
    inspectable for the methods section.

    Returns:
        (shrunk_covariance, shrinkage_intensity). Intensity 0 means the sample
        covariance was kept as-is; 1 means it was fully replaced by the target.
    """
    observations, n_assets = returns.shape
    if observations < 2:
        raise InsufficientHistory(
            f"need at least 2 observations, got {observations}"
        )

    centered = returns - returns.mean(axis=0)
    sample = centered.T @ centered / observations

    # Target: identity scaled to the sample's average variance.
    mean_variance = float(np.trace(sample) / n_assets)
    target = np.eye(n_assets) * mean_variance

    # Dispersion of the sample around the target.
    dispersion = float(np.sum((sample - target) ** 2) / n_assets)
    if dispersion <= 0:
        return sample, 0.0

    # Estimation error of the sample covariance itself.
    error = 0.0
    for t in range(observations):
        outer = np.outer(centered[t], centered[t])
        error += float(np.sum((outer - sample) ** 2))
    error /= observations**2 * n_assets
    error = min(error, dispersion)

    intensity = float(np.clip(error / dispersion, 0.0, 1.0))
    shrunk = intensity * target + (1.0 - intensity) * sample
    return shrunk, intensity


def covariance_diagnostics(
    returns: np.ndarray, shrinkage_intensity: float, shrunk: np.ndarray
) -> CovarianceDiagnostics:
    observations, n_assets = returns.shape
    centered = returns - returns.mean(axis=0)
    sample = centered.T @ centered / max(observations, 1)
    sample_rank = int(np.linalg.matrix_rank(sample))
    eigenvalues = np.linalg.eigvalsh(shrunk)
    smallest = float(max(eigenvalues.min(), 1e-300))
    return CovarianceDiagnostics(
        n_assets=n_assets,
        n_observations=observations,
        rank=sample_rank,
        condition_number=float(eigenvalues.max() / smallest),
        shrinkage_intensity=shrinkage_intensity,
        is_singular_sample=sample_rank < n_assets,
    )


def annualized_moments(
    returns: pd.DataFrame,
    *,
    shrink: bool = True,
    warn_on_short_history: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], CovarianceDiagnostics]:
    """Annualized expected returns and covariance, ready for an optimizer.

    Args:
        returns: Daily returns, columns = symbols.
        shrink: Apply Ledoit-Wolf shrinkage. Turning this off with fewer
            observations than assets produces a singular matrix and is only
            useful for demonstrating that failure in the report.
        warn_on_short_history: Emit a warning when observations <= assets.

    Returns:
        (expected_returns, covariance, symbols, diagnostics), all annualized.
    """
    clean = returns.dropna(axis=1, how="all").dropna(axis=0, how="any")
    symbols = list(clean.columns)
    matrix = clean.to_numpy(dtype=float)
    observations, n_assets = matrix.shape

    if observations < 2:
        raise InsufficientHistory(
            f"need at least 2 return observations, got {observations}"
        )

    if warn_on_short_history and observations <= n_assets:
        warnings.warn(
            f"{observations} observations for {n_assets} assets: the sample "
            f"covariance has rank <= {observations} and is singular. Shrinkage is "
            f"{'applied' if shrink else 'DISABLED -- results will be unreliable'}. "
            f"Consider a longer lookback window for larger universes.",
            RuntimeWarning,
            stacklevel=2,
        )

    if shrink:
        covariance, intensity = ledoit_wolf_shrinkage(matrix)
    else:
        centered = matrix - matrix.mean(axis=0)
        covariance, intensity = centered.T @ centered / observations, 0.0

    diagnostics = covariance_diagnostics(matrix, intensity, covariance)

    expected_returns = matrix.mean(axis=0) * TRADING_DAYS
    covariance = covariance * TRADING_DAYS
    # Enforce exact symmetry; float error otherwise trips the PortfolioProblem check.
    covariance = (covariance + covariance.T) / 2.0
    return expected_returns, covariance, symbols, diagnostics

"""MASE, quantile CRPS, and a paired bootstrap over series."""

from __future__ import annotations

import numpy as np

from ttf_bridge.constants import (
    MEDIAN_QUANTILE_INDEX,
    N_QUANTILES,
    TIMESFM_QUANTILES,
)
from ttf_bridge.eval.comparison import PairedComparison
from ttf_bridge.scaling import seasonal_naive_scale

__all__ = [
    "bootstrap_mean_diff",
    "median_forecast",
    "per_series_crps",
    "per_series_mae",
    "per_series_mase",
    "seasonal_naive_scale",
]


def per_series_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scale: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Mean absolute error over the horizon, divided by the seasonal scale."""
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    scales = np.asarray(scale, dtype=np.float64)
    if truth.shape != pred.shape or truth.ndim != 2:
        raise ValueError("y_true and y_pred must share shape (n, horizon)")
    if scales.shape != (truth.shape[0],):
        raise ValueError("scale must have one value per series")
    mae = np.mean(np.abs(truth - pred), axis=1)
    return mae / np.maximum(scales, eps)


def median_forecast(quantile_forecast: np.ndarray) -> np.ndarray:
    """Take τ=0.5, which is index 4 of the TimesFM-3 grid."""
    forecast = np.asarray(quantile_forecast)
    if forecast.shape[-1] != N_QUANTILES:
        raise ValueError("quantile forecast must end with the 9 TimesFM levels")
    return forecast[..., MEDIAN_QUANTILE_INDEX]


def per_series_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != pred.shape or truth.ndim != 2:
        raise ValueError("y_true and y_pred must share shape (n, horizon)")
    return np.mean(np.abs(truth - pred), axis=1)


def per_series_crps(
    y_true: np.ndarray,
    quantile_forecast: np.ndarray,
    quantiles: tuple[float, ...] = TIMESFM_QUANTILES,
) -> np.ndarray:
    """Quantile CRPS on the supplied grid, averaged over the horizon.

    Uses ``CRPS ≈ 2 * Σ pinball_τ * Δτ`` with ``Δτ`` the mean spacing of the
    grid. On the TimesFM levels 0.1, …, 0.9 a point forecast shifted by ``c``
    scores about ``0.9 * |c|``, because the integral does not cover the tails
    outside that grid. Compare systems that emit this same grid. This is not
    a continuous CRPS.
    """
    truth = np.asarray(y_true, dtype=np.float64)
    forecast = np.asarray(quantile_forecast, dtype=np.float64)
    levels = np.asarray(quantiles, dtype=np.float64)
    if truth.ndim != 2 or forecast.ndim != 3:
        raise ValueError("expected y_true (n, horizon) and forecast (n, horizon, quantiles)")
    if forecast.shape[0] != truth.shape[0] or forecast.shape[1] != truth.shape[1]:
        raise ValueError("forecast and y_true are misaligned")
    if forecast.shape[2] != levels.shape[0] or levels.shape[0] < 2:
        raise ValueError("quantile axis does not match the supplied levels")
    spacing = float(np.diff(levels).mean())
    error = truth[:, :, None] - forecast
    pinball = np.maximum(levels * error, (levels - 1.0) * error)
    crps_steps = 2.0 * pinball.sum(axis=-1) * spacing
    return crps_steps.mean(axis=1)


def bootstrap_mean_diff(
    bridge_scores: np.ndarray,
    reference_scores: np.ndarray,
    *,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Paired bootstrap of ``mean(bridge - reference)`` over series."""
    bridge = np.asarray(bridge_scores, dtype=np.float64)
    reference = np.asarray(reference_scores, dtype=np.float64)
    if bridge.shape != reference.shape or bridge.ndim != 1:
        raise ValueError("scores must be paired 1-d arrays")
    if bridge.shape[0] == 0:
        raise ValueError("scores must be non-empty")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in (0, 1)")
    diff = bridge - reference
    rng = np.random.default_rng(seed)
    index = rng.integers(0, bridge.shape[0], size=(n_boot, bridge.shape[0]))
    means = diff[index].mean(axis=1)
    tail = (1.0 - ci) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return PairedComparison(float(diff.mean()), float(low), float(high))

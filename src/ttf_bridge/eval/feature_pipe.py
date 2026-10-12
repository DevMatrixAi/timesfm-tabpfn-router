"""Two feature-pipe columns taken from TimesFM-3's own quantile forecast.

Column ``timesfm_q50`` is the mean over the 64-step horizon of the median,
quantile index 4 (τ=0.5).

Column ``timesfm_q90_minus_q10`` is the mean over the horizon of
(index 8, τ=0.9) minus (index 0, τ=0.1).

Both columns are appended to the static table. The inputs must be the
forecast returned by TimesFM-3, not the bridge head.
"""

from __future__ import annotations

import numpy as np

from ttf_bridge.constants import (
    FEATURE_PIPE_COLUMN_NAMES,
    FEATURE_PIPE_REDUCTION,
    HORIZON,
    MEDIAN_QUANTILE_INDEX,
    N_QUANTILES,
    TIMESFM_QUANTILES,
    WIDTH_HIGH_QUANTILE_INDEX,
    WIDTH_LOW_QUANTILE_INDEX,
)


def feature_pipe_values(
    quantile_forecast: np.ndarray,
    quantiles: tuple[float, ...] = TIMESFM_QUANTILES,
) -> np.ndarray:
    """Return ``(n, 2)`` with ``timesfm_q50`` and ``timesfm_q90_minus_q10``."""
    if tuple(float(level) for level in quantiles) != TIMESFM_QUANTILES:
        raise ValueError("feature-pipe quantiles are locked to the TimesFM-3 grid")
    if FEATURE_PIPE_REDUCTION != "mean_over_horizon":
        raise RuntimeError("feature-pipe reduction drifted from mean_over_horizon")
    forecast = np.asarray(quantile_forecast, dtype=np.float64)
    if forecast.ndim != 3 or forecast.shape[1] != HORIZON or forecast.shape[2] != N_QUANTILES:
        raise ValueError(f"quantile_forecast must have shape (n, {HORIZON}, {N_QUANTILES})")
    median = forecast[:, :, MEDIAN_QUANTILE_INDEX].mean(axis=1)
    width = (
        forecast[:, :, WIDTH_HIGH_QUANTILE_INDEX] - forecast[:, :, WIDTH_LOW_QUANTILE_INDEX]
    ).mean(axis=1)
    return np.stack([median, width], axis=1)


def append_feature_pipe(static_features: np.ndarray, pipe_values: np.ndarray) -> np.ndarray:
    """Append the two pipe columns on the right of the static table."""
    static = np.asarray(static_features, dtype=np.float64)
    pipe = np.asarray(pipe_values, dtype=np.float64)
    if static.ndim != 2 or pipe.ndim != 2:
        raise ValueError("static features and pipe values must be 2-d")
    if static.shape[0] != pipe.shape[0]:
        raise ValueError("static features and pipe values have different row counts")
    if pipe.shape[1] != len(FEATURE_PIPE_COLUMN_NAMES):
        raise ValueError("feature pipe must contribute exactly two columns")
    return np.concatenate([static, pipe], axis=1)

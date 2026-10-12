"""Seasonal-naive scale used by MASE."""

from __future__ import annotations

import numpy as np


def seasonal_naive_scale(
    history: np.ndarray,
    period: np.ndarray | int,
    eps: float = 1e-6,
) -> np.ndarray:
    """In-sample seasonal naive MAE for each series.

    ``scale_i = mean_t |y_t - y_{t-m}|`` over the observed history.
    A zero scale is lifted to ``eps`` so MASE stays finite.
    """
    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("history must have shape (n, time)")
    n_series, time = values.shape
    periods = np.asarray(period)
    if periods.ndim == 0:
        periods = np.full(n_series, int(periods), dtype=np.int64)
    periods = periods.astype(np.int64, copy=False)
    if periods.shape != (n_series,):
        raise ValueError("period must be a scalar or one value per series")

    scales = np.empty(n_series, dtype=np.float64)
    for raw_period in np.unique(periods):
        season = int(raw_period)
        if season <= 0 or season >= time:
            raise ValueError(f"seasonal period {season} is not in 1..{time - 1}")
        mask = periods == season
        delta = np.abs(values[mask, season:] - values[mask, :-season])
        scales[mask] = delta.mean(axis=1)
    return np.maximum(scales, eps)

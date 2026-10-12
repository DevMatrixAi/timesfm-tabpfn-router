"""Arrays that make one paired example, and the table columns models may see."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ttf_bridge.constants import HORIZON, STATIC_FEATURE_NAMES


@dataclass
class SyntheticBatch:
    """One entity split. ``future`` is the label and is not a model feature."""

    split: str
    entity_id: np.ndarray
    group_id: np.ndarray
    covariates: np.ndarray
    history: np.ndarray
    future: np.ndarray
    known_future_events: np.ndarray
    mase_scale: np.ndarray
    period: np.ndarray
    kappa: float
    context_length: int
    horizon: int
    # Full event indicator over context + horizon. The table only sees the
    # horizon slice, as ``known_future_events``.
    events: np.ndarray | None = None


def tabular_feature_names(horizon: int = HORIZON) -> tuple[str, ...]:
    event_names = tuple(f"known_future_event_{index}" for index in range(horizon))
    return STATIC_FEATURE_NAMES + event_names


def tabular_features(batch: SyntheticBatch) -> np.ndarray:
    """Static columns plus known-future event flags.

    Reads group, covariates, and known-future flags. Does not read ``history``
    or ``future``. Column 0 is the categorical group id.
    """
    n_series = batch.entity_id.shape[0]
    if batch.covariates.shape != (n_series, len(STATIC_FEATURE_NAMES) - 1):
        raise ValueError("covariates must have one row per entity and 4 columns")
    if batch.known_future_events.shape != (n_series, batch.horizon):
        raise ValueError("known_future_events must have shape (n, horizon)")
    group = batch.group_id.astype(np.float32, copy=False).reshape(n_series, 1)
    static = np.concatenate([group, batch.covariates.astype(np.float32, copy=False)], axis=1)
    events = batch.known_future_events.astype(np.float32, copy=False)
    return np.concatenate([static, events], axis=1)


CATEGORICAL_FEATURE_INDEX = 0

"""Skip the router when the known future is only a low-cardinality calendar."""

from __future__ import annotations

import numpy as np

CALENDAR_CARDINALITY_MAX = 31
_CALENDAR_TOKENS = (
    "hour",
    "dow",
    "day_of_week",
    "dayofweek",
    "weekday",
    "day_of_month",
    "dayofmonth",
    "month",
    "week",
    "minute",
    "quarter",
)


def calendar_guard_fires(names: tuple[str, ...] | list[str], columns: np.ndarray) -> bool:
    """True when every known-future column is a calendar indicator of cardinality ≤ 31.

    ``columns`` has shape ``(n_series, n_columns)`` and lines up with ``names``.
    A promo flag, a continuous covariate, or an event indicator does not pass.
    """
    values = np.asarray(columns)
    if values.ndim != 2 or values.shape[1] != len(names) or len(names) == 0:
        raise ValueError("known-future columns must have one vector per name")
    return all(_is_calendar(name, values[:, index]) for index, name in enumerate(names))


def _is_calendar(name: str, column: np.ndarray) -> bool:
    unique = np.unique(column)
    if unique.size > CALENDAR_CARDINALITY_MAX:
        return False
    token = name.lower()
    return any(marker in token for marker in _CALENDAR_TOKENS)

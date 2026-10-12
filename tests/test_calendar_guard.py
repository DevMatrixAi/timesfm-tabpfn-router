"""The calendar guard fires only for low-cardinality calendar columns."""

from __future__ import annotations

import numpy as np

from ttf_bridge.constants import HORIZON
from ttf_bridge.data.m4_hourly import known_future_column_names
from ttf_bridge.eval.calendar_guard import calendar_guard_fires


def test_hour_and_dow_fire_the_guard() -> None:
    names = known_future_column_names()
    columns = np.column_stack(
        [
            np.tile(np.arange(HORIZON) % 24, (5, 1)),
            np.tile(np.arange(HORIZON) % 7, (5, 1)),
        ]
    )
    assert columns.shape[1] == len(names)
    assert calendar_guard_fires(names, columns)


def test_an_event_flag_or_wide_column_does_not_fire() -> None:
    names = ("known_future_event_0", "hour_0")
    columns = np.column_stack([np.array([0.0, 1.0, 0.0]), np.array([0.0, 12.0, 23.0])])
    assert not calendar_guard_fires(names, columns)
    wide = np.arange(40, dtype=np.float64).reshape(-1, 1)
    assert not calendar_guard_fires(("hour_0",), wide)

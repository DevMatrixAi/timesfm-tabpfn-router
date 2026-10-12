"""Temporal windows for the M4 Hourly panel do not leak the test label."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from ttf_bridge.constants import HORIZON
from ttf_bridge.data.m4_hourly import CONTEXT, MIN_LENGTH, build_m4_hourly_windows, known_future_column_names
from ttf_bridge.data.real import load_real_panel


def test_unknown_real_source_stays_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="synthetic entity hold-out"):
        load_real_panel("local-csv")


def test_temporal_windows_keep_labels_apart_and_build_the_calendar() -> None:
    length = MIN_LENGTH
    start = datetime(2015, 7, 1, 12, 0)
    h1 = np.arange(length, dtype=np.float64)
    h2 = h1 + 10.0
    train, test = build_m4_hourly_windows({"H2": h2, "H1": h1}, {"H1": start, "H2": start})
    assert list(test.entity_id) == ["H1", "H2"]
    assert test.history.shape == (2, CONTEXT)
    assert test.future.shape == (2, HORIZON)
    assert test.static.shape[1] == 1 + 2 * HORIZON
    assert test.past_future.shape == (2, 2, CONTEXT + HORIZON)
    assert known_future_column_names()[0] == "hour_0"
    assert known_future_column_names()[HORIZON] == "dow_0"
    # H1 starts at 12:00 on Wednesday 1 July 2015. The test label starts at index 576.
    test_origin = length - HORIZON
    stamp_hour = (start.hour + test_origin) % 24
    assert test.hour[0, 0] == stamp_hour
    assert test.static[0, 0] == 0.0
    assert test.static[0, 1] == stamp_hour
    np.testing.assert_array_equal(test.future[0], h1[test_origin:])
    np.testing.assert_array_equal(train.future[0], h1[test_origin - HORIZON : test_origin])
    assert train.mase_scale.shape == (2,)
    assert np.all(train.mase_scale > 0.0)

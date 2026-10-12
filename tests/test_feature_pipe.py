"""Feature-pipe columns are the TimesFM median and the q90-q10 width."""

from __future__ import annotations

import numpy as np

from ttf_bridge.constants import FEATURE_PIPE_COLUMN_NAMES, TIMESFM_QUANTILES
from ttf_bridge.eval.feature_pipe import append_feature_pipe, feature_pipe_values


def test_two_columns_use_q50_and_q90_minus_q10() -> None:
    forecast = np.zeros((2, 64, 9))
    forecast[:, :, 4] = 3.0
    forecast[:, :, 8] = 5.0
    forecast[:, :, 0] = 1.0
    columns = feature_pipe_values(forecast)
    assert columns.shape == (2, 2)
    assert np.allclose(columns[:, 0], 3.0)
    assert np.allclose(columns[:, 1], 4.0)
    assert FEATURE_PIPE_COLUMN_NAMES == ("timesfm_q50", "timesfm_q90_minus_q10")
    assert TIMESFM_QUANTILES[4] == 0.5
    assert TIMESFM_QUANTILES[0] == 0.1
    assert TIMESFM_QUANTILES[8] == 0.9

    static = np.ones((2, 3))
    augmented = append_feature_pipe(static, columns)
    assert augmented.shape == (2, 5)
    assert np.allclose(augmented[:, :3], 1.0)
    assert np.allclose(augmented[:, 3:], columns)


def test_width_is_averaged_over_the_horizon() -> None:
    forecast = np.zeros((1, 64, 9))
    forecast[0, :, 4] = np.linspace(0.0, 6.3, 64)
    forecast[0, :, 8] = 2.0
    forecast[0, :32, 0] = 0.0
    forecast[0, 32:, 0] = 2.0
    columns = feature_pipe_values(forecast)
    assert np.isclose(columns[0, 0], forecast[0, :, 4].mean())
    assert np.isclose(columns[0, 1], 1.0)

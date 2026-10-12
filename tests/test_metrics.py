"""MASE, CRPS approximation, and the paired bootstrap."""

from __future__ import annotations

import numpy as np

from ttf_bridge.eval.metrics import (
    bootstrap_mean_diff,
    per_series_crps,
    per_series_mase,
)
from ttf_bridge.scaling import seasonal_naive_scale


def test_seasonal_scale_matches_a_hand_calculation() -> None:
    history = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [0, 0, 1, 1, 2, 2],
        ],
        dtype=np.float64,
    )
    scale = seasonal_naive_scale(history, np.array([2, 2]))
    assert np.allclose(scale, [2.0, 1.0])


def test_zero_seasonal_scale_is_floored() -> None:
    history = np.zeros((2, 8))
    scale = seasonal_naive_scale(history, 2, eps=1e-6)
    assert np.allclose(scale, 1e-6)


def test_mase_divides_by_the_stored_scale() -> None:
    truth = np.array([[1.0, 3.0], [0.0, 0.0]])
    pred = np.array([[0.0, 1.0], [2.0, 2.0]])
    scale = np.array([2.0, 4.0])
    # series 0 MAE = 1.5, series 1 MAE = 2
    assert np.allclose(per_series_mase(truth, pred, scale), [0.75, 0.5])


def test_crps_of_a_point_shift_on_the_timesfm_grid() -> None:
    truth = np.zeros((1, 1))
    forecast = np.ones((1, 64, 9))
    forecast = forecast[:, :1, :]
    score = per_series_crps(truth, forecast)
    # 2 * 4.5 * 0.1 = 0.9 for a unit point shift
    assert score.shape == (1,)
    assert np.allclose(score, 0.9)
    perfect = np.zeros((1, 1, 9))
    assert np.allclose(per_series_crps(truth, perfect), 0.0)


def test_bootstrap_separation_and_tie() -> None:
    rng = np.random.default_rng(0)
    bridge = rng.normal(1.0, 0.05, size=300)
    baseline = rng.normal(2.0, 0.05, size=300)
    better = bootstrap_mean_diff(bridge, baseline, n_boot=400, seed=1)
    assert better.bridge_is_better
    assert better.ci_high < 0

    same = bootstrap_mean_diff(bridge, bridge, n_boot=200, seed=2)
    assert same.diff_mean == 0.0
    assert same.tied
    assert not same.bridge_is_better

    again = bootstrap_mean_diff(bridge, baseline, n_boot=400, seed=1)
    assert again == better

"""The router split keeps hold-out labels out of the regime features."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("route_forecasts", ROOT / "scripts" / "route_forecasts.py")
ROUTE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ROUTE)


def _regime(native_mase: float, pipe_mase: float) -> dict:
    n = 20
    history = np.tile(np.sin(np.linspace(0, 8.0, 128)), (n, 1))
    flags = np.zeros((n, 64))
    flags[:10] = 1.0
    mase = np.column_stack(
        [
            np.full(n, native_mase),
            np.full(n, 1.5),
            np.full(n, pipe_mase),
        ]
    )
    return {
        "n": n,
        "history": history,
        "flags": flags,
        "group": np.tile(np.arange(4), n // 4),
        "mase": mase,
        "crps": mase.copy(),
        "timesfm_mase": np.full(n, 1.0),
        "n_numeric": 4,
    }


def test_holdout_gap_uses_only_router_training_entities() -> None:
    regime = _regime(native_mase=0.4, pipe_mase=0.8)
    train_index, hold_index = ROUTE._split_entities(regime["n"], seed=0)
    hold = ROUTE._materialize(regime, hold_index, train_index)
    pipe_column = ROUTE.FEATURE_NAMES.index("train_pipe_minus_timesfm_mase")
    expected = float(regime["mase"][train_index, 2].mean() - regime["timesfm_mase"][train_index].mean())
    assert np.allclose(hold["x"][:, pipe_column], expected)
    assert set(train_index).isdisjoint(set(hold_index))


def test_ridge_picks_the_training_winner() -> None:
    easy = _regime(native_mase=0.2, pipe_mase=0.9)
    other = _regime(native_mase=0.9, pipe_mase=0.2)
    train_a, hold_a = ROUTE._split_entities(easy["n"], seed=0)
    train_b, hold_b = ROUTE._split_entities(other["n"], seed=0)
    train = ROUTE._concat(
        [
            ROUTE._materialize(easy, train_a, train_a),
            ROUTE._materialize(other, train_b, train_b),
        ]
    )
    hold = ROUTE._concat(
        [
            ROUTE._materialize(easy, hold_a, train_a),
            ROUTE._materialize(other, hold_b, train_b),
        ]
    )
    beta = ROUTE._fit_all(train["x"], train["mase"], alpha=1.0)
    chosen = ROUTE._predict(hold["x"], beta, train["mean"], train["std"])
    assert np.all(chosen[: hold_a.size] == 0)
    assert np.all(chosen[hold_a.size :] == 2)

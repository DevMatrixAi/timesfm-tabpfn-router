"""Direct multi-horizon TabPFN baselines in long form.

Each series becomes 64 rows. Static columns, including the group id and the
known-future flags, are repeated. A step index is the last column. The feature
pipe inserts its two horizon-mean scalars after the static columns and before
that index. TabPFN-alone does not get those two columns.

The fit target is the future value at that step. The future is not a feature.
Train entities are capped by the same context-bank index used for embeddings.
"""

from __future__ import annotations

import gc
from typing import Any, Callable

import numpy as np

from ttf_bridge.constants import HORIZON, N_QUANTILES, TIMESFM_QUANTILES
from ttf_bridge.data.schema import SyntheticBatch, tabular_features
from ttf_bridge.eval.feature_pipe import append_feature_pipe, feature_pipe_values
from ttf_bridge.models.representations import as_float_array

# Safety valve for a test table larger than this. The 50k hold-out is 128,000
# step-rows and is predicted in one call. Splitting a larger table calls the
# same fitted model on each slice.
PREDICT_CHUNK_ROWS = 131072


def stack_steps(static_features: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    """Repeat rows across the horizon and append the step index.

    Returns ``(n, 64, n_features)``. Column 0 stays column 0 of ``static_features``.
    ``extra``, when given, sits between the static columns and the step index.
    """
    static = np.asarray(static_features, dtype=np.float64)
    if static.ndim != 2 or static.shape[0] < 1:
        raise ValueError("static features must have shape (n, n_features)")
    n_rows = static.shape[0]
    repeated = np.repeat(static[:, None, :], HORIZON, axis=1)
    steps = np.arange(HORIZON, dtype=np.float64).reshape(1, HORIZON, 1)
    step_column = np.broadcast_to(steps, (n_rows, HORIZON, 1))
    if extra is None:
        return np.concatenate([repeated, step_column], axis=-1)
    extra_values = np.asarray(extra, dtype=np.float64)
    if extra_values.ndim != 2 or extra_values.shape[0] != n_rows:
        raise ValueError("extra columns must have shape (n, n_extra)")
    extra_repeated = np.repeat(extra_values[:, None, :], HORIZON, axis=1)
    return np.concatenate([repeated, extra_repeated, step_column], axis=-1)


def alone_matrix(batch: SyntheticBatch) -> np.ndarray:
    """Static table plus a step index. No feature-pipe columns and no future."""
    return stack_steps(tabular_features(batch))


def pipe_matrix(batch: SyntheticBatch, timesfm_quantiles: np.ndarray) -> np.ndarray:
    """Static table, the two TimesFM summary columns, then the step index."""
    static = tabular_features(batch)
    piped = append_feature_pipe(static, feature_pipe_values(timesfm_quantiles))
    return stack_steps(piped)


def predict_horizon_quantiles(
    model: Any,
    train_x: np.ndarray,
    train_future: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    """Fit on flattened step rows and return ``(n_test, 64, 9)``."""
    train = np.asarray(train_x, dtype=np.float64)
    test = np.asarray(test_x, dtype=np.float64)
    future = np.asarray(train_future, dtype=np.float64)
    if train.ndim != 3 or train.shape[1] != HORIZON:
        raise ValueError(f"train_x must have shape (n, {HORIZON}, features)")
    if future.shape != (train.shape[0], HORIZON):
        raise ValueError("train_future must line up with train_x")
    if test.ndim != 3 or test.shape[1:] != train.shape[1:]:
        raise ValueError("test_x features do not match train_x")
    width = train.shape[-1]
    model.fit(train.reshape(-1, width), future.reshape(-1))
    predicted = np.asarray(model.predict_quantiles(test.reshape(-1, width)), dtype=np.float64)
    release_baseline_model(model)
    expected = (test.shape[0] * HORIZON, N_QUANTILES)
    if predicted.shape != expected:
        raise ValueError(f"quantile prediction shape {predicted.shape}, expected {expected}")
    return predicted.reshape(test.shape[0], HORIZON, N_QUANTILES)


def fit_capped_baselines(
    factory: Callable[[], Any],
    train_batch: SyntheticBatch,
    test_batch: SyntheticBatch,
    train_quantiles: np.ndarray,
    test_quantiles: np.ndarray,
    bank_index: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit TabPFN-alone and the feature pipe on the bank entities only."""
    index = np.asarray(bank_index, dtype=np.int64)
    train_future = np.asarray(train_batch.future)[index]
    alone_train = alone_matrix(train_batch)[index]
    alone_test = alone_matrix(test_batch)
    pipe_train = pipe_matrix(train_batch, train_quantiles)[index]
    pipe_test = pipe_matrix(test_batch, test_quantiles)
    tabpfn = predict_horizon_quantiles(factory(), alone_train, train_future, alone_test)
    gc.collect()
    feature_pipe = predict_horizon_quantiles(factory(), pipe_train, train_future, pipe_test)
    gc.collect()
    return {"tabpfn": tabpfn, "feature_pipe": feature_pipe}


class TabPFNQuantileModel:
    """Local TabPFN regressor with a point-forecast fallback.

    The fallback repeats one point across the nine quantiles when this
    version of ``predict`` rejects ``output_type='quantiles'``. Callers log
    ``used_point_fallback`` so a constant-width forecast is not silent.
    """

    def __init__(self, regressor: Any) -> None:
        self.regressor = regressor
        self.used_point_fallback = False

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        try:
            self.regressor.fit(features, target, categorical_features_indices=[0])
        except TypeError:
            self.regressor.fit(features, target)

    def predict_quantiles(self, features: np.ndarray, *, chunk_rows: int = PREDICT_CHUNK_ROWS) -> np.ndarray:
        rows = np.asarray(features)
        if chunk_rows < 1:
            raise ValueError("chunk_rows must be positive")
        if len(rows) <= chunk_rows:
            return self._predict_quantiles_chunk(rows)
        print(f"tabpfn_predict_chunk_rows={chunk_rows} n_rows={len(rows)}", flush=True)
        parts = [
            self._predict_quantiles_chunk(rows[start : start + chunk_rows])
            for start in range(0, len(rows), chunk_rows)
        ]
        return np.concatenate(parts, axis=0)

    def _predict_quantiles_chunk(self, features: np.ndarray) -> np.ndarray:
        try:
            raw = self.regressor.predict(
                features,
                output_type="quantiles",
                quantiles=list(TIMESFM_QUANTILES),
            )
        except TypeError:
            self.used_point_fallback = True
            point = np.asarray(self.regressor.predict(features), dtype=np.float64).reshape(-1)
            return np.repeat(point[:, None], N_QUANTILES, axis=1)
        coerced = _coerce_quantile_matrix(raw, len(features))
        if coerced.shape != (len(features), N_QUANTILES):
            self.used_point_fallback = True
            point = as_float_array(raw).reshape(-1)
            if point.shape[0] != len(features):
                raise ValueError(
                    f"TabPFN prediction shape {getattr(raw, 'shape', type(raw))} "
                    f"could not be read as {(len(features), N_QUANTILES)}"
                )
            return np.repeat(point[:, None], N_QUANTILES, axis=1)
        return coerced


def release_baseline_model(model: Any) -> None:
    """Drop a fitted baseline so the next fit does not keep its GPU context."""
    if hasattr(model, "regressor"):
        model.regressor = None
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _coerce_quantile_matrix(raw: object, n_rows: int) -> np.ndarray:
    if isinstance(raw, dict):
        columns = []
        for level in TIMESFM_QUANTILES:
            if level in raw:
                columns.append(np.asarray(raw[level], dtype=np.float64).reshape(-1))
            elif str(level) in raw:
                columns.append(np.asarray(raw[str(level)], dtype=np.float64).reshape(-1))
            else:
                raise ValueError(f"TabPFN quantile output is missing {level}")
        return np.stack(columns, axis=1)
    array = np.asarray(as_float_array(raw) if hasattr(raw, "detach") else raw, dtype=np.float64)
    if array.shape == (n_rows, N_QUANTILES):
        return array
    if array.shape == (N_QUANTILES, n_rows):
        return np.asarray(array.T, dtype=np.float64)
    return array

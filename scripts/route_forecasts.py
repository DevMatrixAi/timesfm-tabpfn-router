#!/usr/bin/env python3
"""Route each held-out series to one of three already-scored systems.

The router is fit on an entity split of the scored test series. It does not
refit TimesFM or TabPFN, and it does not read the future label. Regime-level
features come from the router-training entities only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from ttf_bridge.constants import HORIZON, STATIC_FEATURE_NAMES
from ttf_bridge.eval.metrics import bootstrap_mean_diff

SYSTEMS = ("timesfm_native_covariates", "tabpfn", "feature_pipe")
FEATURE_NAMES = (
    "known_future_column_fraction",
    "n_static_columns",
    "n_categorical_columns",
    "n_numeric_columns",
    "n_known_future_columns",
    "group_cardinality",
    "train_entity_count",
    "context_length",
    "horizon",
    "lag1_autocorrelation",
    "seasonal_autocorrelation",
    "seasonal_lag",
    "history_std",
    "temporal_naive_mase",
    "known_future_event_rate",
    "known_future_flag_std",
    "known_future_unique_flags",
    "train_pipe_minus_timesfm_mase",
    "train_native_minus_timesfm_mase",
    "train_tabpfn_minus_timesfm_mase",
    "train_pipe_beats_timesfm",
)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def main() -> int:
    regimes = _regimes()
    loaded = [_load_regime(item) for item in regimes]
    train_parts = []
    hold_parts = []
    for regime in loaded:
        train_index, hold_index = _split_entities(regime["n"], seed=0)
        _assert_disjoint(train_index, hold_index)
        train_parts.append(_materialize(regime, train_index, train_index))
        hold_parts.append(_materialize(regime, hold_index, train_index))
    train = _concat(train_parts)
    hold = _concat(hold_parts)
    alpha = _select_alpha(train)
    beta = _fit_all(train["x"], train["mase"], alpha)
    chosen = _predict(hold["x"], beta, train["mean"], train["std"])
    oracle = np.argmin(hold["mase"], axis=1)
    router_mase = _take(hold["mase"], chosen)
    router_crps = _take(hold["crps"], chosen)
    oracle_mase = _take(hold["mase"], oracle)
    oracle_crps = _take(hold["crps"], oracle)
    print("features " + json.dumps(list(FEATURE_NAMES)))
    print(f"ridge_alpha={alpha:g}")
    print(
        f"router_train_entities={train['mase'].shape[0]} "
        f"holdout_entities={hold['mase'].shape[0]}"
    )
    print(f"oracle_match_rate={float(np.mean(chosen == oracle)):.6f}")
    for index, name in enumerate(SYSTEMS):
        print(f"router_pick_{name}={int(np.sum(chosen == index))}")
    rows = {
        "timesfm_native_covariates": (hold["mase"][:, 0], hold["crps"][:, 0]),
        "tabpfn": (hold["mase"][:, 1], hold["crps"][:, 1]),
        "feature_pipe": (hold["mase"][:, 2], hold["crps"][:, 2]),
        "router": (router_mase, router_crps),
        "oracle": (oracle_mase, oracle_crps),
    }
    print("| system | mase | crps |")
    print("| --- | --- | --- |")
    for name, (mase, crps) in rows.items():
        print(f"| {name} | {mase.mean():.6f} | {crps.mean():.6f} |")
    win = True
    print("| metric | reference | diff_mean | ci_low | ci_high |")
    print("| --- | --- | --- | --- | --- |")
    for metric_index, metric in enumerate(("mase", "crps")):
        for name in SYSTEMS:
            paired = bootstrap_mean_diff(
                rows["router"][metric_index],
                rows[name][metric_index],
                n_boot=1000,
                ci=0.95,
                seed=0,
            )
            if metric == "mase" and not paired.ci_high < 0.0:
                win = False
            print(
                f"| {metric} | {name} | {paired.diff_mean:.6f} | "
                f"{paired.ci_low:.6f} | {paired.ci_high:.6f} |"
            )
    print("status: WIN" if win else "status: NO_WIN")
    return 0


def _regimes() -> tuple[dict, ...]:
    return (
        {"name": "synthetic_kappa1", "native": None},
        {"name": "synthetic_kappa0", "native": None},
        {
            "name": "synthetic_strong_kappa1",
            "native": ROOT / "artifacts/router/native_synthetic_strong_kappa1.npz",
        },
        {
            "name": "synthetic_strong_kappa0",
            "native": ROOT / "artifacts/router/native_synthetic_strong_kappa0.npz",
        },
    )


def _load_regime(item: dict) -> dict:
    name = item["name"]
    cache = np.load(ROOT / "artifacts/cache" / name / "test.npz")
    scored = np.load(ROOT / "artifacts/runs" / name / "per_series.npz")
    mase = []
    crps = []
    for system in SYSTEMS:
        if system == "timesfm_native_covariates" and f"{system}__mase" not in scored.files:
            if item["native"] is None or not Path(item["native"]).is_file():
                raise SystemExit(f"error: {name} has no timesfm_native_covariates scores")
            side = np.load(item["native"])
            mase.append(np.asarray(side["mase"], dtype=np.float64))
            crps.append(np.asarray(side["crps"], dtype=np.float64))
            continue
        mase.append(np.asarray(scored[f"{system}__mase"], dtype=np.float64))
        crps.append(np.asarray(scored[f"{system}__crps"], dtype=np.float64))
    history = np.asarray(cache["history"], dtype=np.float64)
    flags = np.asarray(cache["known_future_events"], dtype=np.float64)
    group = np.asarray(cache["group_id"])
    n = history.shape[0]
    if any(column.shape[0] != n for column in (*mase, *crps, flags, group)):
        raise SystemExit(f"error: {name} score length does not match the cache")
    if flags.shape[1] != HORIZON or history.shape[1] < HORIZON + 2:
        raise SystemExit(f"error: {name} history or flags have an unexpected shape")
    return {
        "name": name,
        "n": n,
        "history": history,
        "flags": flags,
        "group": group,
        "mase": np.stack(mase, axis=1),
        "crps": np.stack(crps, axis=1),
        "timesfm_mase": np.asarray(scored["timesfm_univariate__mase"], dtype=np.float64),
        "n_numeric": int(cache["covariates"].shape[1]),
    }


def _split_entities(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    mid = n // 2
    return np.sort(order[:mid]), np.sort(order[mid:])


def _assert_disjoint(train_index: np.ndarray, hold_index: np.ndarray) -> None:
    if np.intersect1d(train_index, hold_index).size:
        raise SystemExit("error: router train and holdout entities overlap")


def _materialize(regime: dict, index: np.ndarray, train_index: np.ndarray) -> dict:
    gaps = _train_gaps(regime, train_index)
    series = _series_features(regime["history"][index], regime["flags"][index])
    schema = _schema_features(regime, train_index)
    schema_block = np.repeat(schema.reshape(1, -1), index.size, axis=0)
    gap_block = np.repeat(gaps.reshape(1, -1), index.size, axis=0)
    x = np.concatenate([schema_block, series, gap_block], axis=1)
    if x.shape[1] != len(FEATURE_NAMES):
        raise SystemExit(f"error: feature width {x.shape[1]} != {len(FEATURE_NAMES)}")
    return {"x": x, "mase": regime["mase"][index], "crps": regime["crps"][index]}


def _schema_features(regime: dict, train_index: np.ndarray) -> np.ndarray:
    n_categorical = 1
    n_numeric = regime["n_numeric"]
    n_static = n_categorical + n_numeric
    n_known = HORIZON
    n_columns = n_static + n_known
    if STATIC_FEATURE_NAMES[0] != "group_id":
        raise SystemExit("error: static column 0 is not the group id")
    if n_numeric != len(STATIC_FEATURE_NAMES) - 1:
        raise SystemExit("error: numeric covariate count drifted from the static table")
    return np.array(
        [
            n_known / n_columns,
            n_static,
            n_categorical,
            n_numeric,
            n_known,
            np.unique(regime["group"][train_index]).size,
            train_index.size,
            regime["history"].shape[1],
            HORIZON,
        ],
        dtype=np.float64,
    )


def _train_gaps(regime: dict, train_index: np.ndarray) -> np.ndarray:
    mase = regime["mase"][train_index]
    timesfm = regime["timesfm_mase"][train_index]
    native = float(mase[:, 0].mean() - timesfm.mean())
    tabpfn = float(mase[:, 1].mean() - timesfm.mean())
    pipe = float(mase[:, 2].mean() - timesfm.mean())
    return np.array([pipe, native, tabpfn, 1.0 if pipe < 0.0 else 0.0], dtype=np.float64)


def _series_features(history: np.ndarray, flags: np.ndarray) -> np.ndarray:
    rows = np.empty((history.shape[0], 8), dtype=np.float64)
    for index in range(history.shape[0]):
        series = history[index]
        lag1 = _autocorrelation(series, 1)
        season_lag, season_acf = _detect_season(series)
        rows[index] = (
            lag1,
            season_acf,
            season_lag,
            float(series.std()),
            _temporal_naive_mase(series, season_lag),
            float(flags[index].mean()),
            float(flags[index].std()),
            float(np.unique(flags[index]).size),
        )
    return rows


def _autocorrelation(series: np.ndarray, lag: int) -> float:
    centered = series - series.mean()
    energy = float(np.dot(centered, centered))
    if energy <= 0.0 or lag >= series.shape[0]:
        return 0.0
    return float(np.dot(centered[:-lag], centered[lag:]) / energy)


def _detect_season(series: np.ndarray) -> tuple[int, float]:
    limit = min(168, series.shape[0] // 2)
    best_lag = 1
    best = -1.0
    for lag in range(2, limit + 1):
        value = _autocorrelation(series, lag)
        if value > best:
            best = value
            best_lag = lag
    return best_lag, best


def _temporal_naive_mase(series: np.ndarray, period: int) -> float:
    horizon = HORIZON
    if period < 1 or period + horizon > series.shape[0]:
        period = 1
    actual = series[-horizon:]
    forecast = series[-horizon - period : -period]
    prefix = series[:-horizon]
    scale = np.mean(np.abs(prefix[period:] - prefix[:-period]))
    return float(np.mean(np.abs(actual - forecast)) / max(float(scale), 1e-6))


def _concat(parts: list[dict]) -> dict:
    x = np.concatenate([part["x"] for part in parts], axis=0)
    return {
        "x": x,
        "mase": np.concatenate([part["mase"] for part in parts], axis=0),
        "crps": np.concatenate([part["crps"] for part in parts], axis=0),
        "mean": x.mean(axis=0),
        "std": np.maximum(x.std(axis=0), 1e-8),
    }


def _select_alpha(train: dict) -> float:
    inner_train, inner_hold = _split_rows_by_block(train["mase"].shape[0], seed=1)
    x_train = train["x"][inner_train]
    mean = x_train.mean(axis=0)
    std = np.maximum(x_train.std(axis=0), 1e-8)
    best_alpha = ALPHAS[0]
    best_mase = float("inf")
    for alpha in ALPHAS:
        beta = _fit_all(x_train, train["mase"][inner_train], alpha, mean, std)
        chosen = _predict(train["x"][inner_hold], beta, mean, std)
        score = float(_take(train["mase"][inner_hold], chosen).mean())
        if score < best_mase:
            best_mase = score
            best_alpha = alpha
    return best_alpha


def _split_rows_by_block(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split each regime block in half. Blocks were concatenated in regime order."""
    if n % 4 != 0:
        raise SystemExit("error: expected four equal regime blocks")
    block = n // 4
    train_parts = []
    hold_parts = []
    for offset in range(0, n, block):
        local_train, local_hold = _split_entities(block, seed)
        train_parts.append(local_train + offset)
        hold_parts.append(local_hold + offset)
    return np.concatenate(train_parts), np.concatenate(hold_parts)


def _fit_all(x: np.ndarray, mase: np.ndarray, alpha: float, mean=None, std=None) -> np.ndarray:
    if mean is None:
        mean = x.mean(axis=0)
        std = np.maximum(x.std(axis=0), 1e-8)
    scaled = (x - mean) / std
    design = np.concatenate([np.ones((scaled.shape[0], 1)), scaled], axis=1)
    beta = np.empty((design.shape[1], mase.shape[1]), dtype=np.float64)
    for column in range(mase.shape[1]):
        beta[:, column] = _ridge(design, mase[:, column], alpha)
    return beta


def _ridge(design: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    gram = design.T @ design
    penalty = alpha * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(gram + penalty, design.T @ target)


def _predict(x: np.ndarray, beta: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    scaled = (x - mean) / std
    design = np.concatenate([np.ones((scaled.shape[0], 1)), scaled], axis=1)
    return np.argmin(design @ beta, axis=1)


def _take(values: np.ndarray, choice: np.ndarray) -> np.ndarray:
    return values[np.arange(values.shape[0]), choice]


if __name__ == "__main__":
    raise SystemExit(main())

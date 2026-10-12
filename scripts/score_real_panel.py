#!/usr/bin/env python3
"""Score the three existing systems and the router on M4 Hourly.

The router is fit only on this panel's training windows. Synthetic caches
and tables are not read or written. TabPFN stays on fit_preprocessors with
memory_saving_mode off.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from ttf_bridge.eval.calendar_guard import calendar_guard_fires
from ttf_bridge.data.m4_hourly import (
    M4_HOURLY_URLS,
    M4WindowBatch,
    known_future_column_names,
    load_m4_hourly,
)
from ttf_bridge.eval.feature_pipe import append_feature_pipe, feature_pipe_values
from ttf_bridge.eval.metrics import bootstrap_mean_diff
from ttf_bridge.eval.scoring import scores_from_quantiles
from ttf_bridge.eval.tabpfn_baseline import TabPFNQuantileModel, predict_horizon_quantiles, stack_steps
from ttf_bridge.licenses import BANNER
from ttf_bridge.models.representations import context_patch_count
from ttf_bridge.models.tabpfn_encoder import load_tabpfn_regressor
from ttf_bridge.models.timesfm_encoder import (
    decode_history_batch,
    forecast_past_future_covariates,
    load_timesfm_forecaster,
    release_cuda,
)

_ROUTER_PATH = ROOT / "scripts" / "route_forecasts.py"
_SPEC = importlib.util.spec_from_file_location("route_forecasts", _ROUTER_PATH)
ROUTER = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(ROUTER)

SYSTEMS = ("timesfm_native_covariates", "tabpfn", "feature_pipe")
RUN_NAME = "real_m4_hourly"


def main() -> int:
    print(BANNER, flush=True)
    data_dir = ROOT / "data" / "real" / "m4"
    _ensure_downloads(data_dir)
    train, test = load_m4_hourly(data_dir)
    if not np.array_equal(train.entity_id, test.entity_id):
        raise SystemExit("error: train and test entities diverged")
    _describe(train)
    print("forecasting TimesFM on the training and test windows", flush=True)
    checkpoint = os.environ.get("TIMESFM_CHECKPOINT", "google/timesfm-3.0-pytorch")
    print(f"timesfm_checkpoint={checkpoint}", flush=True)
    forecaster = load_timesfm_forecaster(checkpoint, device="cuda")
    try:
        train_uni = _univariate(forecaster.model, train.history)
        test_uni = _univariate(forecaster.model, test.history)
        train_native = forecast_past_future_covariates(
            forecaster.model, train.history, train.past_future, precision="fp16", device="cuda", batch_size=16
        )
        test_native = forecast_past_future_covariates(
            forecaster.model, test.history, test.past_future, precision="fp16", device="cuda", batch_size=16
        )
    finally:
        del forecaster
        release_cuda()
    print("tabpfn_baseline_fit_mode=fit_preprocessors memory_saving_mode=False", flush=True)
    train_tab, train_pipe = _cross_fit(train, train_uni)
    test_tab, test_pipe = _fit_test(train, test, train_uni, test_uni)
    train_scores = _bundle("train", train, train_native, train_tab, train_pipe)
    test_scores = _bundle("test", test, test_native, test_tab, test_pipe)
    names = known_future_column_names()
    if calendar_guard_fires(names, test.static[:, 1:]):
        print("calendar_guard=fired; router not called", flush=True)
        chosen = np.zeros(test.entity_id.size, dtype=np.int64)
        alpha = float("nan")
    else:
        chosen, alpha = _route(train, test, train_scores, test_scores, train_uni)
    _report(test, test_scores, chosen, alpha)
    return 0


def _describe(train: M4WindowBatch) -> None:
    names = known_future_column_names()
    print("dataset M4 Hourly", flush=True)
    print("source M4-methods Dataset/Train/Hourly-train.csv and Dataset/M4-info.csv", flush=True)
    print(f"entities {train.entity_id.size} hourly series ids, sorted H1..H{train.entity_id.size}", flush=True)
    print("static_columns group_id = M4 category Other for every hourly series", flush=True)
    print(f"known_future_columns {names[0]}..{names[HORIZON_LAST]} and {names[64]}..{names[-1]}", flush=True)
    print(
        "split temporal: test label is the last 64 points, test history the 512 before it; "
        "router training label is the preceding 64 points. Labels do not overlap.",
        flush=True,
    )
    print("router_fit this panel's training windows only", flush=True)


HORIZON_LAST = 63


def _ensure_downloads(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in M4_HOURLY_URLS.items():
        path = data_dir / name
        if path.is_file():
            continue
        print(f"download {name}", flush=True)
        urllib.request.urlretrieve(url, path)


def _univariate(model, history: np.ndarray) -> np.ndarray:
    patches = context_patch_count(history.shape[1])
    parts = []
    batch = 16
    for start in range(0, history.shape[0], batch):
        _tokens, quantiles = decode_history_batch(
            model,
            history[start : start + batch],
            n_context_patches=patches,
            precision="fp16",
            device="cuda",
        )
        parts.append(np.asarray(quantiles, dtype=np.float64))
        done = min(start + batch, history.shape[0])
        if done == history.shape[0] or done % 160 == 0:
            print(f"timesfm_univariate_rows={done}/{history.shape[0]}", flush=True)
    return np.concatenate(parts, axis=0)


def _factory():
    model = TabPFNQuantileModel(
        load_tabpfn_regressor(
            seed=0,
            device="cuda",
            fit_mode="fit_preprocessors",
            memory_saving_mode=False,
        )
    )
    return model


def _cross_fit(train: M4WindowBatch, quantiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-sample TabPFN scores on the training window, split by sorted id."""
    n = train.entity_id.size
    mid = n // 2
    alone = np.empty((n, 64, 9), dtype=np.float64)
    pipe = np.empty((n, 64, 9), dtype=np.float64)
    halves = (np.arange(0, mid), np.arange(mid, n))
    for fit_index, pred_index in (halves, halves[::-1]):
        alone[pred_index], pipe[pred_index] = _predict_pair(
            train.static[fit_index],
            train.future[fit_index],
            quantiles[fit_index],
            train.static[pred_index],
            quantiles[pred_index],
        )
        print(f"cross_fit predicted {pred_index.size} training entities", flush=True)
    return alone, pipe


def _fit_test(
    train: M4WindowBatch,
    test: M4WindowBatch,
    train_quantiles: np.ndarray,
    test_quantiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _predict_pair(train.static, train.future, train_quantiles, test.static, test_quantiles)


def _predict_pair(fit_static, fit_future, fit_quantiles, pred_static, pred_quantiles):
    alone_model = _factory()
    alone = predict_horizon_quantiles(
        alone_model,
        stack_steps(fit_static),
        fit_future,
        stack_steps(pred_static),
    )
    if alone_model.used_point_fallback:
        print("tabpfn_quantile_fallback: point_forecast_broadcast", flush=True)
    gc.collect()
    release_cuda()
    piped_fit = stack_steps(append_feature_pipe(fit_static, feature_pipe_values(fit_quantiles)))
    piped_pred = stack_steps(append_feature_pipe(pred_static, feature_pipe_values(pred_quantiles)))
    pipe_model = _factory()
    pipe = predict_horizon_quantiles(pipe_model, piped_fit, fit_future, piped_pred)
    if pipe_model.used_point_fallback:
        print("feature_pipe_quantile_fallback: point_forecast_broadcast", flush=True)
    gc.collect()
    release_cuda()
    if not np.isfinite(alone).all() or not np.isfinite(pipe).all():
        raise SystemExit("error: TabPFN quantiles are not finite")
    return alone, pipe


def _bundle(label: str, batch: M4WindowBatch, native, tabpfn, pipe) -> dict[str, np.ndarray]:
    forecasts = {
        "timesfm_native_covariates": native,
        "tabpfn": tabpfn,
        "feature_pipe": pipe,
    }
    mase = []
    crps = []
    for name in SYSTEMS:
        scored = scores_from_quantiles(batch.future, forecasts[name], batch.mase_scale)
        mase.append(scored["mase"])
        crps.append(scored["crps"])
        print(
            f"{label} {name} mase={scored['mase'].mean():.6f} crps={scored['crps'].mean():.6f}",
            flush=True,
        )
    return {"mase": np.stack(mase, axis=1), "crps": np.stack(crps, axis=1)}


def _route(train, test, train_scores, test_scores, train_uni):
    schema = _schema(train)
    gaps = _gaps(train_scores["mase"], scores_from_quantiles(train.future, train_uni, train.mase_scale)["mase"])
    x_train = _features(train, schema, gaps)
    x_test = _features(test, schema, gaps)
    alpha = _select_alpha(x_train, train_scores["mase"])
    beta = ROUTER._fit_all(x_train, train_scores["mase"], alpha)
    chosen = ROUTER._predict(
        x_test,
        beta,
        x_train.mean(axis=0),
        np.maximum(x_train.std(axis=0), 1e-8),
    )
    return chosen, alpha


def _schema(train: M4WindowBatch) -> np.ndarray:
    n_known = 2 * 64
    n_columns = train.static.shape[1]
    if n_columns != 1 + n_known:
        raise SystemExit("error: static table is not group id plus known-future columns")
    return np.array(
        [
            n_known / n_columns,
            1,
            1,
            0,
            n_known,
            np.unique(train.group_id).size,
            train.entity_id.size,
            train.history.shape[1],
            64,
        ],
        dtype=np.float64,
    )


def _gaps(mase: np.ndarray, timesfm_mase: np.ndarray) -> np.ndarray:
    pipe = float(mase[:, 2].mean() - timesfm_mase.mean())
    native = float(mase[:, 0].mean() - timesfm_mase.mean())
    tabpfn = float(mase[:, 1].mean() - timesfm_mase.mean())
    return np.array([pipe, native, tabpfn, 1.0 if pipe < 0.0 else 0.0], dtype=np.float64)


def _features(batch: M4WindowBatch, schema: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    series = ROUTER._series_features(batch.history, batch.hour)
    n = batch.entity_id.size
    x = np.concatenate(
        [
            np.repeat(schema.reshape(1, -1), n, axis=0),
            series,
            np.repeat(gaps.reshape(1, -1), n, axis=0),
        ],
        axis=1,
    )
    if x.shape[1] != len(ROUTER.FEATURE_NAMES):
        raise SystemExit(f"error: feature width {x.shape[1]}")
    return x


def _select_alpha(x: np.ndarray, mase: np.ndarray) -> float:
    mid = x.shape[0] // 2
    best_alpha = ROUTER.ALPHAS[0]
    best_score = float("inf")
    folds = ((slice(0, mid), slice(mid, None)), (slice(mid, None), slice(0, mid)))
    for alpha in ROUTER.ALPHAS:
        fold_scores = []
        for fit, ev in folds:
            beta = ROUTER._fit_all(x[fit], mase[fit], alpha)
            chosen = ROUTER._predict(
                x[ev],
                beta,
                x[fit].mean(axis=0),
                np.maximum(x[fit].std(axis=0), 1e-8),
            )
            fold_scores.append(float(ROUTER._take(mase[ev], chosen).mean()))
        score = float(np.mean(fold_scores))
        if score < best_score:
            best_score = score
            best_alpha = alpha
    return best_alpha


def _report(test: M4WindowBatch, scores: dict[str, np.ndarray], chosen: np.ndarray, alpha: float) -> None:
    oracle = np.argmin(scores["mase"], axis=1)
    router_mase = ROUTER._take(scores["mase"], chosen)
    router_crps = ROUTER._take(scores["crps"], chosen)
    oracle_mase = ROUTER._take(scores["mase"], oracle)
    oracle_crps = ROUTER._take(scores["crps"], oracle)
    rows = {
        "timesfm_native_covariates": (scores["mase"][:, 0], scores["crps"][:, 0]),
        "tabpfn": (scores["mase"][:, 1], scores["crps"][:, 1]),
        "feature_pipe": (scores["mase"][:, 2], scores["crps"][:, 2]),
        "router": (router_mase, router_crps),
        "oracle": (oracle_mase, oracle_crps),
    }
    print("features " + json.dumps(list(ROUTER.FEATURE_NAMES)), flush=True)
    print(f"ridge_alpha={alpha:g}", flush=True)
    print(f"holdout_entities={test.entity_id.size}", flush=True)
    print(f"oracle_match_rate={float(np.mean(chosen == oracle)):.6f}", flush=True)
    for index, name in enumerate(SYSTEMS):
        print(f"router_pick_{name}={int(np.sum(chosen == index))}", flush=True)
    lines = ["| system | mase | crps |", "| --- | --- | --- |"]
    for name, (mase, crps) in rows.items():
        lines.append(f"| {name} | {mase.mean():.6f} | {crps.mean():.6f} |")
    lines.extend(
        [
            "| metric | reference | diff_mean | ci_low | ci_high |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    win = True
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
            lines.append(
                f"| {metric} | {name} | {paired.diff_mean:.6f} | {paired.ci_low:.6f} | {paired.ci_high:.6f} |"
            )
    lines.append("status: WIN" if win else "status: NO_WIN")
    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    destination = ROOT / "artifacts" / "runs" / RUN_NAME
    if "synthetic" in destination.name:
        raise SystemExit("error: refusing to write a synthetic run directory")
    destination.mkdir(parents=True, exist_ok=True)
    payload = {f"{name}__mase": rows[name][0] for name in rows}
    payload.update({f"{name}__crps": rows[name][1] for name in rows})
    np.savez_compressed(destination / "per_series.npz", **payload)
    (destination / "table.md").write_text(text, encoding="utf-8")
    (destination / "meta.json").write_text(
        json.dumps(
            {
                "dataset": "M4 Hourly",
                "router_fit": "this panel's training windows only",
                "split": "temporal",
                "n_holdout": int(test.entity_id.size),
                "static_columns": ["group_id (M4 category Other)"],
                "known_future_columns": list(known_future_column_names()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {destination / 'table.md'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

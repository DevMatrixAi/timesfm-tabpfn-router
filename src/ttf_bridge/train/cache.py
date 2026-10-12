"""On-disk cache of frozen tokens. Training reads this and not the backbones.

A TimesFM pass may omit ``tab_tokens``. Training and scoring require them.
Quantiles stored here are TimesFM's own forecast, sorted on the 9-level grid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ttf_bridge.constants import HORIZON, N_QUANTILES

SERIES_KEYS = (
    "entity_id",
    "group_id",
    "covariates",
    "history",
    "future",
    "known_future_events",
    "mase_scale",
    "period",
)
TIMESFM_KEYS = ("ts_tokens", "timesfm_quantiles")
TAB_KEY = "tab_tokens"


def cache_dir(root: Path, experiment: str) -> Path:
    return root / "artifacts" / "cache" / experiment


def data_dir(root: Path, experiment: str) -> Path:
    return root / "data" / "synthetic" / experiment


def run_dir(root: Path, experiment: str) -> Path:
    return root / "artifacts" / "runs" / experiment


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.array(payload[key]) for key in payload.files}


def save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items()})


def load_meta(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a mapping")
    if data.get("non_commercial") is not True:
        raise RuntimeError(f"{path} is not marked non_commercial")
    return data


def save_meta(path: Path, meta: Mapping[str, Any]) -> None:
    if meta.get("non_commercial") is not True:
        raise RuntimeError("refusing to write cache meta without non_commercial: true")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(meta), indent=2) + "\n", encoding="utf-8")


def dims_from_cache_meta(meta: Mapping[str, Any]) -> tuple[int, int]:
    """Token widths actually stored. Do not substitute the yaml defaults."""
    if "ts_dim" not in meta:
        raise KeyError("cache meta is missing ts_dim; run the TimesFM extract stage")
    tab_dim = meta.get("tab_dim")
    if tab_dim in (None, 0):
        raise KeyError("cache meta is missing tab_dim; run the TabPFN extract stage")
    return int(meta["ts_dim"]), int(tab_dim)


def require_training_arrays(arrays: Mapping[str, np.ndarray], *, n_context_patches: int | None = None) -> None:
    """Reject a split that cannot train or score the bridge."""
    missing = [key for key in (*SERIES_KEYS, *TIMESFM_KEYS, TAB_KEY) if key not in arrays]
    if missing:
        raise KeyError(f"cached split is missing {missing}")
    n_rows = int(np.asarray(arrays["entity_id"]).shape[0])
    future = np.asarray(arrays["future"])
    tokens = np.asarray(arrays["ts_tokens"])
    tab = np.asarray(arrays["tab_tokens"])
    quantiles = np.asarray(arrays["timesfm_quantiles"])
    if future.shape != (n_rows, HORIZON):
        raise ValueError(f"future must have shape (n, {HORIZON})")
    if tokens.ndim != 3 or tokens.shape[0] != n_rows:
        raise ValueError("ts_tokens must have shape (n, patches, dim)")
    if n_context_patches is not None and tokens.shape[1] != n_context_patches:
        raise ValueError("ts_tokens patch count does not match the cache meta")
    if tab.shape != (n_rows, 1, tab.shape[-1]):
        raise ValueError("tab_tokens must have shape (n, 1, dim)")
    if quantiles.shape != (n_rows, HORIZON, N_QUANTILES):
        raise ValueError(f"timesfm_quantiles must have shape (n, {HORIZON}, {N_QUANTILES})")
    if int(np.asarray(arrays["mase_scale"]).shape[0]) != n_rows:
        raise ValueError("mase_scale must have one value per series")

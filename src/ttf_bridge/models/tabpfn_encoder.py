"""Frozen TabPFN-3.5 embeddings. The import is lazy so CPU tests do not need the package.

The context label is the mean of the 64-step future. That scalar conditions
the embedding only. The bridge target stays the full future. Rows inside the
context bank are embedded out of fold (``n_fold >= 2``) so a row's own label
is not in its token. Validation and test rows are ``transform`` on the model
fit to the bank. Their labels never enter that fit.

The TabPFN client has no embeddings. This path uses the local ``tabpfn`` package.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ttf_bridge.constants import HORIZON
from ttf_bridge.data.context_bank import format_context_bank_log, resolve_context_bank_size
from ttf_bridge.data.schema import CATEGORICAL_FEATURE_INDEX, SyntheticBatch, tabular_features
from ttf_bridge.licenses import BANNER
from ttf_bridge.models.representations import assemble_tab_tokens, pack_tabpfn_embeddings
from ttf_bridge.train.cache import load_meta, load_npz, save_meta, save_npz

_IMPORT_MESSAGE = (
    "TabPFN embeddings require the local tabpfn package and tabpfn-extensions. "
    "Install them with `pip install 'tabpfn>=9' tabpfn-extensions`. "
    "The TabPFN client does not expose embeddings. "
    "Weights are under the TabPFN-3.5 License v1.0. "
    "Accept that license at https://ux.priorlabs.ai and export TABPFN_TOKEN. "
    "This project does not set that variable and does not bypass the prompt. "
    + BANNER
)

LICENSE_HELP = (
    "TabPFN-3.5 weights need an accepted TabPFN-3.5 License v1.0. "
    "Open https://ux.priorlabs.ai, log in, accept the license, copy the API key, "
    "and export TABPFN_TOKEN. This project does not accept the license for you "
    "and does not download gated weights another way."
)


def load_tabpfn_embedder(
    *,
    n_estimators: int = 1,
    n_fold: int = 2,
    seed: int = 0,
    device: str | None = None,
) -> Any:
    """Build a ``TabPFNEmbedding`` around one local regressor."""
    if n_estimators != 1:
        raise ValueError("the bridge path uses n_estimators=1")
    if n_fold != 0 and n_fold < 2:
        raise ValueError("tabpfn_n_fold must be 0 (no OOF) or >= 2")
    regressor_cls, embedding_cls = _import_tabpfn()
    regressor = construct_tabpfn_regressor(regressor_cls, seed=seed, device=device)
    return embedding_cls(n_fold=n_fold, model=regressor)


def load_tabpfn_regressor(
    *,
    seed: int,
    device: str | None = None,
    fit_mode: str | None = None,
    memory_saving_mode: bool | str | None = None,
) -> Any:
    """Build a fresh local regressor for a baseline fit. Does not reuse an embedder.

    ``TABPFN_MODEL_PATH``, when set, is passed as ``model_path`` so the regressor
    loads a local official checkpoint instead of the library default ``auto``.
    """
    regressor_cls, _embedding_cls = _import_tabpfn()
    return construct_tabpfn_regressor(
        regressor_cls,
        seed=seed,
        device=device,
        fit_mode=fit_mode,
        memory_saving_mode=memory_saving_mode,
    )


def construct_tabpfn_regressor(
    regressor_cls: Any,
    *,
    seed: int,
    device: str | None,
    fit_mode: str | None = None,
    memory_saving_mode: bool | str | None = None,
) -> Any:
    """Try the documented constructor kwargs, then drop the ones this version rejects.

    ``ignore_pretraining_limits`` is set when the class has it. The long-form
    baseline has one row per horizon step, so a few hundred series already pass
    TabPFN's default 10k-row pretraining limit. The flag is that library's
    supported escape hatch. It is not a license bypass.

    ``fit_mode`` and ``memory_saving_mode`` are passed through when set. A build
    that rejects them is constructed again with the library defaults.
    """
    extra: dict[str, Any] = {}
    if fit_mode is not None:
        extra["fit_mode"] = fit_mode
    if memory_saving_mode is not None:
        extra["memory_saving_mode"] = memory_saving_mode
    extras: tuple[dict[str, Any], ...] = (extra, {}) if extra else ({},)
    last_error: TypeError | None = None
    for extra in extras:
        model, last_error = _construct_with_candidates(
            regressor_cls,
            seed=seed,
            device=device,
            extra=extra,
        )
        if model is not None:
            if hasattr(model, "ignore_pretraining_limits"):
                model.ignore_pretraining_limits = True
            return model
    raise TypeError("TabPFNRegressor rejected every supported constructor") from last_error


def _construct_with_candidates(
    regressor_cls: Any,
    *,
    seed: int,
    device: str | None,
    extra: dict[str, Any],
) -> tuple[Any | None, TypeError | None]:
    base: dict[str, Any] = {"n_estimators": 1, "random_state": int(seed), **extra}
    model_path = os.environ.get("TABPFN_MODEL_PATH")
    if model_path:
        base["model_path"] = model_path
    candidates: list[dict[str, Any]] = []
    if device is not None:
        candidates.append(
            {**base, "device": device, "categorical_features_indices": [CATEGORICAL_FEATURE_INDEX]}
        )
        candidates.append({**base, "device": device})
    candidates.append({**base, "categorical_features_indices": [CATEGORICAL_FEATURE_INDEX]})
    candidates.append(base)
    last_error: TypeError | None = None
    for kwargs in candidates:
        try:
            return regressor_cls(**kwargs), None
        except TypeError as exc:
            last_error = exc
    return None, last_error


def context_targets(future: np.ndarray) -> np.ndarray:
    """Mean of the 64-step future. A conditioning label, not the bridge target."""
    values = np.asarray(future, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != HORIZON:
        raise ValueError(f"future must have shape (n, {HORIZON})")
    return values.mean(axis=1)


def select_bank_index(n_available: int, requested: int, seed: int) -> np.ndarray:
    """Sorted train-row positions. The size is ``min(requested, n_available)``."""
    actual = resolve_context_bank_size(requested, n_available)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_available, size=actual, replace=False)
    return np.sort(chosen.astype(np.int64))


def embed_with_context_bank(
    embedder: Any,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    other_features: Mapping[str, np.ndarray],
    bank_index: np.ndarray,
) -> dict[str, np.ndarray]:
    """OOF-embed the bank, then transform every row that was not in that fit."""
    features = np.asarray(train_features)
    targets = np.asarray(train_targets)
    index = np.asarray(bank_index, dtype=np.int64)
    if features.shape[0] != targets.shape[0]:
        raise ValueError("train features and context targets differ in length")
    bank_embeddings = pack_tabpfn_embeddings(embedder.fit_transform(features[index], targets[index]))
    rest_index = np.setdiff1d(np.arange(features.shape[0], dtype=np.int64), index)
    if rest_index.size:
        rest_embeddings = pack_tabpfn_embeddings(embedder.transform(features[rest_index]))
    else:
        rest_embeddings = np.zeros((0, 1, bank_embeddings.shape[-1]), dtype=np.float32)
    packed = {
        "train": assemble_tab_tokens(
            bank_embeddings,
            rest_embeddings,
            index,
            features.shape[0],
        )
    }
    width = packed["train"].shape[-1]
    for name, rows in other_features.items():
        rows_array = np.asarray(rows)
        if rows_array.shape[0] == 0:
            packed[name] = np.zeros((0, 1, width), dtype=np.float32)
            continue
        packed[name] = pack_tabpfn_embeddings(embedder.transform(rows_array))
        if packed[name].shape[-1] != width:
            raise ValueError(f"{name} embedding width does not match the train bank")
    return packed


def write_tabpfn_cache(
    cache_dir: Path,
    embedder: Any,
    *,
    requested: int,
    seed: int,
    n_fold: int,
) -> str:
    """Attach TabPFN tokens to a cache that already holds TimesFM arrays.

    Returns the context-bank log line.
    """
    if n_fold != 0 and n_fold < 2:
        raise ValueError("tabpfn_n_fold must be 0 or >= 2")
    meta = load_meta(cache_dir / "meta.json")
    splits = {name: load_npz(cache_dir / f"{name}.npz") for name in ("train", "val", "test")}
    for name, arrays in splits.items():
        if "ts_tokens" not in arrays or "timesfm_quantiles" not in arrays:
            raise FileNotFoundError(
                f"{name} is missing TimesFM arrays. Run extract --stage timesfm first."
            )
    features = {
        name: tabular_features(batch_from_arrays(name, arrays, meta))
        for name, arrays in splits.items()
    }
    targets = context_targets(splits["train"]["future"])
    bank_index = select_bank_index(targets.shape[0], requested, seed)
    if n_fold >= 2 and bank_index.shape[0] < n_fold:
        raise ValueError(
            f"context bank actual={bank_index.shape[0]} is smaller than tabpfn_n_fold={n_fold}"
        )
    tokens = embed_with_context_bank(
        embedder,
        features["train"],
        targets,
        {"val": features["val"], "test": features["test"]},
        bank_index,
    )
    for name, arrays in splits.items():
        arrays["tab_tokens"] = tokens[name]
        save_npz(cache_dir / f"{name}.npz", arrays)
    log_line = format_context_bank_log(requested, int(bank_index.shape[0]), int(targets.shape[0]))
    stages = list(meta.get("stages", []))
    if "tabpfn" not in stages:
        stages.append("tabpfn")
    meta.update(
        {
            "tab_dim": int(tokens["train"].shape[-1]),
            "context_bank_log": log_line,
            "context_target": "mean_future",
            "tabpfn_n_fold": int(n_fold),
            "stages": stages,
            "non_commercial": True,
            "license_banner": BANNER,
        }
    )
    save_meta(cache_dir / "meta.json", meta)
    np.save(cache_dir / "context_bank_index.npy", bank_index)
    (cache_dir / "context_bank.txt").write_text(log_line + "\n", encoding="utf-8")
    return log_line


def batch_from_arrays(split: str, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> SyntheticBatch:
    """Rebuild the table view of one cached split. ``future`` stays the label."""
    return SyntheticBatch(
        split=split,
        entity_id=np.asarray(arrays["entity_id"]),
        group_id=np.asarray(arrays["group_id"]),
        covariates=np.asarray(arrays["covariates"]),
        history=np.asarray(arrays["history"]),
        future=np.asarray(arrays["future"]),
        known_future_events=np.asarray(arrays["known_future_events"]),
        mase_scale=np.asarray(arrays["mase_scale"]),
        period=np.asarray(arrays["period"]),
        kappa=float(meta["kappa"]),
        context_length=int(meta["context_length"]),
        horizon=int(meta["horizon"]),
    )


def license_failure(exc: BaseException) -> bool:
    blob = f"{type(exc).__name__} {exc}".lower()
    return "license" in blob or "tabpfn_token" in blob or "auth_token" in blob


def _import_tabpfn() -> tuple[Any, Any]:
    try:
        from tabpfn import TabPFNRegressor
        from tabpfn_extensions.embedding import TabPFNEmbedding
    except ImportError as exc:
        raise ImportError(_IMPORT_MESSAGE) from exc
    return TabPFNRegressor, TabPFNEmbedding

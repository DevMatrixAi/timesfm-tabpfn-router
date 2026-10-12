"""Frozen TimesFM-3 encoder. The import is lazy so CPU tests do not need the package.

The univariate baseline is this model's own 64×9 quantile forecast, cached
beside the context-patch tokens. The bridge is not loaded here.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from ttf_bridge.constants import HORIZON, N_QUANTILES, PATCH_LENGTH, TIMESFM_QUANTILES
from ttf_bridge.licenses import BANNER
from ttf_bridge.models.representations import (
    TRANSFORMER_OUTPUT_KEY,
    context_patch_count,
    pack_timesfm_outputs,
)
from ttf_bridge.train.cache import load_npz, save_meta, save_npz

_IMPORT_MESSAGE = (
    "TimesFM is not installed. Install it with `pip install 'timesfm[torch]'`. "
    "The weights (google/timesfm-3.0-pytorch) are under the TimesFM "
    "Non-Commercial License v1.0. This project does not bypass that license. "
    + BANNER
)


def load_timesfm_forecaster(checkpoint: str, device: str | None = None) -> Any:
    """Load TimesFM-3 and freeze it. Imports ``timesfm3`` only when called."""
    forecaster_cls = _import_forecaster_cls()
    forecaster = forecaster_cls.from_pretrained(checkpoint, device=device)
    freeze_module(forecaster.model)
    return forecaster


def freeze_module(module: Any) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def decode_history_batch(
    model: Any,
    history: np.ndarray,
    *,
    n_context_patches: int,
    horizon: int = HORIZON,
    precision: str = "fp16",
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one history batch to context tokens and the model's own quantiles.

    ``precision='fp16'`` uses CUDA autocast. Non-finite outputs are retried in fp32.
    CPU runs stay in fp32 because autocast is CUDA-only here.
    """
    if horizon != HORIZON:
        raise ValueError(f"horizon is locked to {HORIZON}")
    if precision not in ("fp16", "fp32"):
        raise ValueError("precision must be fp16 or fp32")
    series = np.asarray(history, dtype=np.float32)
    if series.ndim != 2:
        raise ValueError("history must have shape (batch, context)")
    if context_patch_count(series.shape[1]) != n_context_patches:
        raise ValueError("n_context_patches does not match the history length")

    attempts = ("fp16", "fp32") if precision == "fp16" else ("fp32",)
    last = "no attempt"
    for attempt in attempts:
        tokens, quantiles = _decode_once(
            model,
            series,
            n_context_patches=n_context_patches,
            horizon=horizon,
            precision=attempt,
            device=device,
        )
        if np.isfinite(tokens).all() and np.isfinite(quantiles).all():
            return tokens, quantiles
        last = f"{attempt} produced non-finite TimesFM outputs"
    raise RuntimeError(f"TimesFM decode failed: {last}")


def write_timesfm_cache(
    data_dir: Path,
    cache_dir: Path,
    decode_batch: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    meta: dict[str, Any],
    batch_size: int,
    n_context_patches: int,
) -> dict[str, Any]:
    """Copy each synthetic split and attach TimesFM tokens and quantiles."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    cache_dir.mkdir(parents=True, exist_ok=True)
    ts_dim: int | None = None
    for split in ("train", "val", "test"):
        arrays = load_npz(data_dir / f"{split}.npz")
        history = np.asarray(arrays["history"])
        token_parts: list[np.ndarray] = []
        quantile_parts: list[np.ndarray] = []
        for start in range(0, history.shape[0], batch_size):
            tokens, quantiles = decode_batch(history[start : start + batch_size])
            token_parts.append(np.asarray(tokens, dtype=np.float32))
            quantile_parts.append(np.asarray(quantiles, dtype=np.float32))
        ts_tokens = np.concatenate(token_parts, axis=0)
        quantiles = np.concatenate(quantile_parts, axis=0)
        _check_timesfm_arrays(
            history,
            ts_tokens,
            quantiles,
            n_context_patches=n_context_patches,
            ts_dim=ts_dim,
        )
        ts_dim = int(ts_tokens.shape[-1])
        arrays["ts_tokens"] = ts_tokens
        arrays["timesfm_quantiles"] = quantiles
        save_npz(cache_dir / f"{split}.npz", arrays)

    written = dict(meta)
    written.update(
        {
            "patch_length": PATCH_LENGTH,
            "n_context_patches": int(n_context_patches),
            "ts_dim": int(ts_dim),
            "quantiles": list(TIMESFM_QUANTILES),
            "timesfm_quantiles_sorted": True,
            "timesfm_layer": "last",
            "timesfm_alone_source": "timesfm3_forecaster_output",
            "stages": ["timesfm"],
            "non_commercial": True,
            "license_banner": BANNER,
            "splits": ["train", "val", "test"],
        }
    )
    save_meta(cache_dir / "meta.json", written)
    return written


def forecast_past_future_covariates(
    model: Any,
    history: np.ndarray,
    past_future: np.ndarray,
    *,
    horizon: int = HORIZON,
    precision: str = "fp16",
    device: str | None = None,
    batch_size: int = 16,
) -> np.ndarray:
    """TimesFM quantiles with known past-and-future covariates. No token cache is written.

    ``past_future`` is ``(n, channels, context + horizon)`` or ``(n, context + horizon)``.
    The horizon slice is what the model treats as known future. Quantiles are sorted
    so index 4 is the median, matching the univariate baseline.
    """
    if horizon != HORIZON:
        raise ValueError(f"horizon is locked to {HORIZON}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if precision not in ("fp16", "fp32"):
        raise ValueError("precision must be fp16 or fp32")
    series = np.asarray(history, dtype=np.float32)
    covariates = np.asarray(past_future, dtype=np.float32)
    if series.ndim != 2:
        raise ValueError("history must have shape (batch, context)")
    if covariates.ndim == 2:
        covariates = covariates[:, None, :]
    if covariates.ndim != 3 or covariates.shape[0] != series.shape[0]:
        raise ValueError("past_future covariates must align with the history batch")
    context = series.shape[1]
    if covariates.shape[-1] != context + horizon:
        raise ValueError(
            f"past_future width must be context + horizon ({context + horizon}), "
            f"got {covariates.shape[-1]}"
        )
    parts: list[np.ndarray] = []
    for start in range(0, series.shape[0], batch_size):
        stop = min(start + batch_size, series.shape[0])
        parts.append(
            _forecast_covariate_batch(
                model,
                series[start:stop],
                covariates[start:stop],
                horizon=horizon,
                precision=precision,
                device=device,
            )
        )
    return np.concatenate(parts, axis=0)


def release_cuda() -> None:
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _import_forecaster_cls() -> Any:
    try:
        from timesfm3 import TimesFM3Forecaster
    except ImportError:
        TimesFM3Forecaster = None
    if TimesFM3Forecaster is not None:
        return TimesFM3Forecaster
    try:
        from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster as TorchForecaster
    except ImportError as exc:
        raise ImportError(_IMPORT_MESSAGE) from exc
    return TorchForecaster


def _decode_once(
    model: Any,
    history: np.ndarray,
    *,
    n_context_patches: int,
    horizon: int,
    precision: str,
    device: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    torch_device = _device_of(model, device)
    target = torch.as_tensor(history, dtype=torch.float32, device=torch_device).unsqueeze(1)
    use_autocast = precision == "fp16" and torch_device.type == "cuda"
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_autocast
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        decoded = model.decode(target=target, horizon=horizon, return_aux_outputs=True)
    if not isinstance(decoded, tuple) or len(decoded) != 2:
        raise TypeError("TimesFM decode(return_aux_outputs=True) must return (logits, forward_out)")
    horizon_logits, forward_out = decoded
    if not isinstance(forward_out, dict) or TRANSFORMER_OUTPUT_KEY not in forward_out:
        raise KeyError(
            f"TimesFM forward output is missing {TRANSFORMER_OUTPUT_KEY}. "
            "Context-patch tokens come from that auxiliary tensor."
        )
    return pack_timesfm_outputs(
        horizon_logits,
        forward_out[TRANSFORMER_OUTPUT_KEY],
        n_context_patches=n_context_patches,
    )


def _forecast_covariate_batch(
    model: Any,
    history: np.ndarray,
    past_future: np.ndarray,
    *,
    horizon: int,
    precision: str,
    device: str | None,
) -> np.ndarray:
    attempts = ("fp16", "fp32") if precision == "fp16" else ("fp32",)
    last = "no attempt"
    for attempt in attempts:
        quantiles = _decode_quantiles(
            model,
            history,
            past_future,
            horizon=horizon,
            precision=attempt,
            device=device,
        )
        if np.isfinite(quantiles).all():
            return quantiles
        last = f"{attempt} produced non-finite TimesFM outputs"
    raise RuntimeError(f"TimesFM covariate decode failed: {last}")


def _decode_quantiles(
    model: Any,
    history: np.ndarray,
    past_future: np.ndarray,
    *,
    horizon: int,
    precision: str,
    device: str | None,
) -> np.ndarray:
    import torch

    torch_device = _device_of(model, device)
    target = torch.as_tensor(history, dtype=torch.float32, device=torch_device).unsqueeze(1)
    covariates = torch.as_tensor(past_future, dtype=torch.float32, device=torch_device)
    use_autocast = precision == "fp16" and torch_device.type == "cuda"
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_autocast
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        decoded = model.decode(
            target=target,
            horizon=horizon,
            past_future_covariates=covariates,
            return_aux_outputs=False,
        )
    logits = decoded[0] if isinstance(decoded, tuple) else decoded
    array = np.asarray(
        logits.detach().float().cpu().numpy() if hasattr(logits, "detach") else logits,
        dtype=np.float32,
    )
    if array.ndim == 4:
        array = array[:, 0]
    if array.shape != (history.shape[0], horizon, N_QUANTILES):
        raise ValueError(
            f"TimesFM covariate quantiles must be (batch, {horizon}, {N_QUANTILES}), got {array.shape}"
        )
    return np.sort(array, axis=-1)


def _device_of(model: Any, device: str | None) -> Any:
    import torch

    if device is not None:
        return torch.device(device)
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return torch.device(model_device)
    parameters = getattr(model, "parameters", None)
    if parameters is not None:
        for param in parameters():
            return param.device
    return torch.device("cpu")


def _check_timesfm_arrays(
    history: np.ndarray,
    ts_tokens: np.ndarray,
    quantiles: np.ndarray,
    *,
    n_context_patches: int,
    ts_dim: int | None,
) -> None:
    n_rows = history.shape[0]
    if ts_tokens.shape[0] != n_rows or ts_tokens.ndim != 3:
        raise ValueError("ts_tokens must have shape (n, patches, dim)")
    if ts_tokens.shape[1] != n_context_patches:
        raise ValueError("ts_tokens patch count does not match the context length")
    if ts_dim is not None and ts_tokens.shape[-1] != ts_dim:
        raise ValueError("TimesFM token width changed between splits")
    if quantiles.shape != (n_rows, HORIZON, N_QUANTILES):
        raise ValueError(f"timesfm_quantiles must have shape (n, {HORIZON}, {N_QUANTILES})")

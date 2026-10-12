"""Pack frozen-backbone outputs into the cache layout.

TimesFM context patches are the leading patches of variate 0. Context lengths
in this project are multiples of 32, so decode does not left-pad and those
patches line up with the history. Quantiles are sorted on the last axis so
index 4 is the median, matching ``TimesFM3Forecaster.predict(sort_quantiles=True)``.
That sorted grid is the TimesFM-alone baseline. It is not the bridge head.

TabPFN embeddings arrive as ``(n_estimators, n_samples, dim)`` or ``(n_samples, dim)``.
The bridge sees one token per row, ``(n_samples, 1, dim)``, from estimator 0.
"""

from __future__ import annotations

import numpy as np

from ttf_bridge.constants import HORIZON, N_QUANTILES, PATCH_LENGTH

TRANSFORMER_OUTPUT_KEY = "__call__:transformer_output"


def as_float_array(value: object) -> np.ndarray:
    """Copy a numpy or torch value into a float32 array."""
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def context_patch_count(context_length: int, patch_length: int = PATCH_LENGTH) -> int:
    if int(context_length) % int(patch_length) != 0 or int(context_length) < int(patch_length):
        raise ValueError(
            f"context_length {context_length} must be a positive multiple of {patch_length}"
        )
    return int(context_length) // int(patch_length)


def slice_context_patches(transformer_output: object, n_context_patches: int) -> np.ndarray:
    """Take variate 0 and the first ``n_context_patches`` from ``(batch, variate, patch, dim)``."""
    if n_context_patches < 1:
        raise ValueError("n_context_patches must be positive")
    output = as_float_array(transformer_output)
    if output.ndim != 4:
        raise ValueError(
            "transformer output must have shape (batch, variate, patch, dim), "
            f"got {output.shape}"
        )
    if output.shape[2] < n_context_patches:
        raise ValueError(
            f"transformer output has {output.shape[2]} patches, "
            f"need {n_context_patches} context patches"
        )
    return output[:, 0, :n_context_patches, :]


def pack_timesfm_outputs(
    horizon_logits: object,
    transformer_output: object,
    *,
    n_context_patches: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return context tokens ``(batch, patches, dim)`` and sorted quantiles ``(batch, 64, 9)``."""
    logits = as_float_array(horizon_logits)
    if logits.ndim == 4:
        logits = logits[:, 0]
    if logits.ndim != 3 or logits.shape[1] != HORIZON or logits.shape[2] != N_QUANTILES:
        raise ValueError(
            f"TimesFM horizon logits must be (batch, {HORIZON}, {N_QUANTILES}) "
            f"after selecting variate 0, got {getattr(horizon_logits, 'shape', logits.shape)}"
        )
    quantiles = np.sort(logits, axis=-1)
    tokens = slice_context_patches(transformer_output, n_context_patches)
    if tokens.shape[0] != quantiles.shape[0]:
        raise ValueError("TimesFM tokens and quantiles have different batch sizes")
    return tokens, quantiles


def pack_tabpfn_embeddings(embeddings: object) -> np.ndarray:
    """Return ``(n_samples, 1, dim)`` from one TabPFN estimator."""
    array = as_float_array(embeddings)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(
                "bridge embeddings require n_estimators=1, "
                f"got shape {tuple(int(dim) for dim in array.shape)}"
            )
        array = array[0]
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"TabPFN embeddings must be (n_samples, dim), got {array.shape}")
    return array[:, None, :]


def assemble_tab_tokens(
    bank_tokens: np.ndarray,
    rest_tokens: np.ndarray,
    bank_index: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    """Put bank and non-bank tokens back in the original row order.

    ``rest_tokens`` follows the increasing row indices outside ``bank_index``.
    """
    bank = np.asarray(bank_tokens, dtype=np.float32)
    index = np.asarray(bank_index, dtype=np.int64)
    if bank.ndim != 3 or bank.shape[1] != 1:
        raise ValueError("bank tokens must have shape (n_bank, 1, dim)")
    if bank.shape[0] != index.shape[0]:
        raise ValueError("bank tokens and bank index differ in length")
    if index.ndim != 1:
        raise ValueError("bank index must be 1-d")
    if len(np.unique(index)) != len(index):
        raise ValueError("bank index has duplicates")
    if n_rows < 1 or index.min() < 0 or index.max() >= n_rows:
        raise ValueError("bank index is outside the train split")
    dim = bank.shape[-1]
    rest = np.asarray(rest_tokens, dtype=np.float32)
    rest_index = np.setdiff1d(np.arange(n_rows, dtype=np.int64), index)
    if rest.size == 0:
        rest = np.zeros((0, 1, dim), dtype=np.float32)
    if rest.ndim != 3 or rest.shape[1] != 1 or rest.shape[-1] != dim:
        raise ValueError("rest tokens must have shape (n_rest, 1, dim) matching the bank")
    if rest.shape[0] != rest_index.shape[0]:
        raise ValueError("rest tokens do not cover the rows outside the bank")
    assembled = np.empty((n_rows, 1, dim), dtype=np.float32)
    assembled[index] = bank
    assembled[rest_index] = rest
    return assembled

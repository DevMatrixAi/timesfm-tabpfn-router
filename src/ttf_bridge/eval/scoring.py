"""Comparison table and the JSON ``evaluate.py`` reads.

``build_win_payload`` is the both-kappa record. A smoke run that only has
kappa=1 uses ``build_single_kappa_record``, which sets ``kappa0_ran`` false
and does not invent the kappa=0 comparisons.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ttf_bridge.constants import N_QUANTILES, PRIMARY_METRICS, TIMESFM_QUANTILES
from ttf_bridge.eval.metrics import bootstrap_mean_diff, median_forecast, per_series_crps, per_series_mase
from ttf_bridge.gates import GateValue, head_gate_direction

SYSTEM_ORDER = (
    "timesfm_univariate",
    "timesfm_native_covariates",
    "tabpfn",
    "feature_pipe",
    "bridge_no_cross_attention",
    "bridge",
    "bridge_tabpfn_head_no_cross_attention",
    "bridge_tabpfn_head",
    "bridge_residual_off",
    "bridge_residual",
)
# Extra paired blocks. Each is written only when that candidate was scored.
EXTRA_PAIRED: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("timesfm_native_covariates", ("timesfm_univariate", "feature_pipe", "tabpfn")),
    (
        "bridge_residual",
        ("timesfm_univariate", "feature_pipe", "tabpfn", "bridge_residual_off"),
    ),
)
TABPFN_HEAD = "bridge_tabpfn_head"
TABPFN_HEAD_ABLATION = "bridge_tabpfn_head_no_cross_attention"
TABPFN_HEAD_REFERENCES = (
    "timesfm_univariate",
    "tabpfn",
    "feature_pipe",
    TABPFN_HEAD_ABLATION,
    "bridge",
)
KAPPA1_REFERENCES = ("timesfm_univariate", "tabpfn", "feature_pipe")
_COMPARISON_FIELDS = {
    "timesfm_univariate": "kappa1_vs_timesfm",
    "tabpfn": "kappa1_vs_tabpfn",
    "feature_pipe": "kappa1_vs_feature_pipe",
}


def scores_from_quantiles(
    future: np.ndarray,
    quantiles: np.ndarray,
    scale: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-series MASE (median, index 4) and quantile CRPS."""
    point = median_forecast(quantiles)
    return {
        "mase": per_series_mase(future, point, scale),
        "crps": per_series_crps(future, quantiles),
    }


def score_systems(
    future: np.ndarray,
    scale: np.ndarray,
    forecasts: Mapping[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    scored: dict[str, dict[str, np.ndarray]] = {}
    for name, quantiles in forecasts.items():
        forecast = np.asarray(quantiles, dtype=np.float64)
        if forecast.shape[-1] != N_QUANTILES or forecast.shape[-2] != future.shape[-1]:
            raise ValueError(f"{name} forecast does not match the 64×9 grid")
        scored[name] = scores_from_quantiles(future, forecast, scale)
    return scored


def render_table(
    per_series: Mapping[str, Mapping[str, np.ndarray]],
    *,
    kappa: float,
    n_bootstrap: int,
    ci: float,
    seed: int,
    kappa0_ran: bool,
    note: str | None = None,
) -> str:
    """Markdown table of mean scores and paired (bridge − reference) intervals."""
    lines = [
        f"kappa={float(kappa):g} n_series={_n_series(per_series)}",
        "",
        "| system | mase | crps |",
        "| --- | --- | --- |",
    ]
    for name in SYSTEM_ORDER:
        if name not in per_series:
            continue
        lines.append(
            "| {name} | {mase:.6f} | {crps:.6f} |".format(
                name=name,
                mase=float(np.mean(per_series[name]["mase"])),
                crps=float(np.mean(per_series[name]["crps"])),
            )
        )
    lines.extend(
        [
            "",
            "paired difference (bridge - reference), lower error is better",
            "| metric | reference | diff_mean | ci_low | ci_high |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    if "bridge" in per_series:
        _append_paired(
            lines,
            per_series,
            candidate="bridge",
            references=tuple(name for name in SYSTEM_ORDER if name != "bridge"),
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            header=False,
        )
    if TABPFN_HEAD in per_series:
        lines.extend(
            [
                "",
                "paired difference (bridge_tabpfn_head - reference), lower error is better",
                "| metric | reference | diff_mean | ci_low | ci_high |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        _append_paired(
            lines,
            per_series,
            candidate=TABPFN_HEAD,
            references=TABPFN_HEAD_REFERENCES,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            header=False,
        )
    for candidate, references in EXTRA_PAIRED:
        if candidate not in per_series:
            continue
        lines.extend(
            [
                "",
                f"paired difference ({candidate} - reference), lower error is better",
                "| metric | reference | diff_mean | ci_low | ci_high |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        _append_paired(
            lines,
            per_series,
            candidate=candidate,
            references=references,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            header=False,
        )
    if not kappa0_ran:
        lines.extend(
            [
                "",
                "kappa=0 was not run. This table cannot satisfy the win rule.",
            ]
        )
    if note:
        lines.extend(["", note])
    return "\n".join(lines) + "\n"


def build_single_kappa_record(
    per_series: Mapping[str, Mapping[str, np.ndarray]],
    gates: Sequence[GateValue],
    *,
    kappa: float,
    n_bootstrap: int,
    ci: float,
    seed: int,
    experiment: str,
) -> dict[str, object]:
    """One-kappa hold-out. ``kappa0_ran`` is false and kappa=0 numbers are absent."""
    table = render_table(
        per_series,
        kappa=kappa,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
        kappa0_ran=False,
    )
    return {
        "kappa0_ran": False,
        "kappa": float(kappa),
        "experiment": experiment,
        "complete": True,
        "non_commercial": True,
        "table": table,
        "kappa1_gates": [_gate_dict(gate) for gate in gates],
        "score_means": _means(per_series),
    }


def build_win_payload(
    kappa1_scores: Mapping[str, Mapping[str, np.ndarray]],
    kappa0_scores: Mapping[str, Mapping[str, np.ndarray]],
    kappa1_gates: Sequence[GateValue],
    *,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    candidate: str = "bridge",
    ablation: str = "bridge_no_cross_attention",
    head_stream: str = "timesfm",
) -> dict[str, object]:
    """Both kappas, in the shape ``outcome_from_payload`` parses.

    ``candidate`` is the forecast the win rule judges. The TimesFM head uses
    ``bridge``. The TabPFN head uses ``bridge_tabpfn_head`` and its own ablation.
    """
    if head_gate_direction(head_stream) not in ("ts", "tab"):
        raise ValueError("head_stream must be 'timesfm' or 'tabpfn'")
    _require_systems(kappa1_scores, (candidate, *KAPPA1_REFERENCES), "kappa=1")
    _require_systems(
        kappa0_scores,
        (candidate, ablation, "timesfm_univariate"),
        "kappa=0",
    )
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for metric in PRIMARY_METRICS:
        block: dict[str, dict[str, float]] = {}
        for reference, field in _COMPARISON_FIELDS.items():
            block[field] = _comparison_dict(
                kappa1_scores[candidate][metric],
                kappa1_scores[reference][metric],
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed,
            )
        block["kappa0_vs_no_cross_attention"] = _comparison_dict(
            kappa0_scores[candidate][metric],
            kappa0_scores[ablation][metric],
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        )
        block["kappa0_vs_timesfm"] = _comparison_dict(
            kappa0_scores[candidate][metric],
            kappa0_scores["timesfm_univariate"][metric],
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        )
        metrics[metric] = block
    table = "\n".join(
        [
            render_table(
                kappa1_scores,
                kappa=1.0,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed,
                kappa0_ran=True,
            ).rstrip(),
            "",
            render_table(
                kappa0_scores,
                kappa=0.0,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed,
                kappa0_ran=True,
            ).rstrip(),
        ]
    )
    return {
        "kappa0_ran": True,
        "non_commercial": True,
        "head_stream": head_stream,
        "kappa1_gates": [_gate_dict(gate) for gate in kappa1_gates],
        "metrics": metrics,
        "table": table + "\n",
    }


def _append_paired(
    lines: list[str],
    per_series: Mapping[str, Mapping[str, np.ndarray]],
    *,
    candidate: str,
    references: Sequence[str],
    n_bootstrap: int,
    ci: float,
    seed: int,
    header: bool = True,
) -> None:
    if header:
        lines.extend(
            [
                "",
                f"paired difference ({candidate} - reference), lower error is better",
                "| metric | reference | diff_mean | ci_low | ci_high |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
    for metric in PRIMARY_METRICS:
        for reference in references:
            if reference not in per_series or reference == candidate:
                continue
            comparison = bootstrap_mean_diff(
                per_series[candidate][metric],
                per_series[reference][metric],
                n_boot=n_bootstrap,
                ci=ci,
                seed=seed,
            )
            lines.append(
                f"| {metric} | {reference} | {comparison.diff_mean:.6f} | "
                f"{comparison.ci_low:.6f} | {comparison.ci_high:.6f} |"
            )


def _comparison_dict(
    bridge_scores: np.ndarray,
    reference_scores: np.ndarray,
    *,
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> dict[str, float]:
    comparison = bootstrap_mean_diff(
        bridge_scores,
        reference_scores,
        n_boot=n_bootstrap,
        ci=ci,
        seed=seed,
    )
    return {
        "diff_mean": comparison.diff_mean,
        "ci_low": comparison.ci_low,
        "ci_high": comparison.ci_high,
    }


def _gate_dict(gate: GateValue) -> dict[str, float | int | str]:
    if not math.isclose(gate.tanh_alpha, math.tanh(gate.alpha), rel_tol=0.0, abs_tol=1e-5):
        raise ValueError("tanh_alpha does not match alpha")
    return {
        "block": gate.block,
        "direction": gate.direction,
        "alpha": gate.alpha,
        "tanh_alpha": gate.tanh_alpha,
    }


def _means(per_series: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    return {
        name: {metric: float(np.mean(values)) for metric, values in scores.items()}
        for name, scores in per_series.items()
    }


def _n_series(per_series: Mapping[str, Mapping[str, np.ndarray]]) -> int:
    first = next(iter(per_series.values()))
    return int(np.asarray(first["mase"]).shape[0])


def load_per_series(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Read ``{system}__mase`` and ``{system}__crps`` arrays."""
    if not path.is_file():
        raise FileNotFoundError(path)
    stored = np.load(path)
    scores: dict[str, dict[str, np.ndarray]] = {}
    for key in stored.files:
        name, _, metric = key.rpartition("__")
        if metric not in PRIMARY_METRICS or not name:
            continue
        scores.setdefault(name, {})[metric] = np.asarray(stored[key], dtype=np.float64)
    if not scores:
        raise ValueError(f"{path} has no per-series scores")
    return scores


def save_per_series(path: Path, per_series: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    arrays = {}
    for name, scores in per_series.items():
        arrays[f"{name}__mase"] = np.asarray(scores["mase"], dtype=np.float64)
        arrays[f"{name}__crps"] = np.asarray(scores["crps"], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def upsert_system_row(
    run_dir: Path,
    name: str,
    scores: Mapping[str, np.ndarray],
    *,
    kappa: float,
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> str:
    """Add one system's per-series scores and rewrite that run's table.

    Existing systems stay. The single-kappa record still cannot declare a win.
    """
    path = run_dir / "per_series.npz"
    per_series = load_per_series(path)
    per_series[name] = {
        "mase": np.asarray(scores["mase"], dtype=np.float64),
        "crps": np.asarray(scores["crps"], dtype=np.float64),
    }
    length = _n_series(per_series)
    for system, values in per_series.items():
        for metric in PRIMARY_METRICS:
            if int(np.asarray(values[metric]).shape[0]) != length:
                raise ValueError(f"{system} {metric} length does not match the table")
    save_per_series(path, per_series)
    table = render_table(
        per_series,
        kappa=kappa,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
        kappa0_ran=False,
    )
    (run_dir / "table.md").write_text(table, encoding="utf-8")
    comparison_path = run_dir / "comparisons.json"
    if comparison_path.is_file():
        payload = json.loads(comparison_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload["table"] = table
            payload["score_means"] = _means(per_series)
            comparison_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return table


def _require_systems(
    scores: Mapping[str, Mapping[str, np.ndarray]],
    required: Sequence[str],
    label: str,
) -> None:
    missing = [name for name in required if name not in scores]
    if missing:
        raise KeyError(f"{label} scores are missing {missing}")
    for name in required:
        for metric in PRIMARY_METRICS:
            if metric not in scores[name]:
                raise KeyError(f"{label} {name} is missing {metric}")

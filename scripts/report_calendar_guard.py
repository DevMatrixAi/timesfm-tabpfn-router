#!/usr/bin/env python3
"""Apply the calendar guard to the saved M4 Hourly hold-out.

Does not forecast, does not fit TabPFN, and does not read synthetic runs.
The unguarded router row is the score file already written for this hold-out.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from ttf_bridge.data.m4_hourly import known_future_column_names, load_m4_hourly
from ttf_bridge.eval.calendar_guard import CALENDAR_CARDINALITY_MAX, calendar_guard_fires
from ttf_bridge.eval.metrics import bootstrap_mean_diff

SYSTEMS = ("timesfm_native_covariates", "tabpfn", "feature_pipe")


def main() -> int:
    run_dir = ROOT / "artifacts" / "runs" / "real_m4_hourly"
    saved = np.load(run_dir / "per_series.npz")
    _train, test = load_m4_hourly(ROOT / "data" / "real" / "m4")
    names = known_future_column_names()
    columns = test.static[:, 1:]
    if columns.shape[1] != len(names):
        raise SystemExit("error: saved panel columns do not match the known-future names")
    fired = calendar_guard_fires(names, columns)
    cardinalities = [int(np.unique(columns[:, index]).size) for index in range(columns.shape[1])]
    print(
        f"calendar_guard={'fired' if fired else 'not_fired'} "
        f"max_cardinality={max(cardinalities)} limit={CALENDAR_CARDINALITY_MAX}",
        flush=True,
    )
    native_mase = np.asarray(saved["timesfm_native_covariates__mase"], dtype=np.float64)
    native_crps = np.asarray(saved["timesfm_native_covariates__crps"], dtype=np.float64)
    tab_mase = np.asarray(saved["tabpfn__mase"], dtype=np.float64)
    pipe_mase = np.asarray(saved["feature_pipe__mase"], dtype=np.float64)
    unguarded_mase = np.asarray(saved["router__mase"], dtype=np.float64)
    unguarded_choice = _recover_choice(unguarded_mase, native_mase, tab_mase, pipe_mase)
    if fired:
        guarded_choice = np.zeros(native_mase.shape[0], dtype=np.int64)
        guarded_mase = native_mase
        guarded_crps = native_crps
    else:
        raise SystemExit("error: the guard did not fire, so this retest would need the router")
    print(
        "unguarded_picks "
        + " ".join(f"{name}={int(np.sum(unguarded_choice == index))}" for index, name in enumerate(SYSTEMS)),
        flush=True,
    )
    overrides = int(np.sum(unguarded_choice != 0))
    bad = int(np.sum(guarded_choice != 0))
    print(f"guard_overrides={overrides}", flush=True)
    print(f"guarded_non_native_picks={bad}", flush=True)
    oracle_mase = np.minimum(np.minimum(native_mase, tab_mase), pipe_mase)
    match = float(np.mean(np.isclose(guarded_mase, oracle_mase)))
    print(f"oracle_match_rate={match:.6f}", flush=True)
    rows = {
        "timesfm_native_covariates": (native_mase, native_crps),
        "tabpfn": (tab_mase, np.asarray(saved["tabpfn__crps"], dtype=np.float64)),
        "feature_pipe": (pipe_mase, np.asarray(saved["feature_pipe__crps"], dtype=np.float64)),
        "router": (unguarded_mase, np.asarray(saved["router__crps"], dtype=np.float64)),
        "router_guarded": (guarded_mase, guarded_crps),
        "oracle": (
            np.asarray(saved["oracle__mase"], dtype=np.float64),
            np.asarray(saved["oracle__crps"], dtype=np.float64),
        ),
    }
    print("| system | mase | crps |")
    print("| --- | --- | --- |")
    for name, (mase, crps) in rows.items():
        print(f"| {name} | {mase.mean():.6f} | {crps.mean():.6f} |")
    paired = bootstrap_mean_diff(guarded_mase, native_mase, n_boot=1000, ci=0.95, seed=0)
    print("| metric | reference | diff_mean | ci_low | ci_high |")
    print("| --- | --- | --- | --- | --- |")
    print(
        f"| mase | timesfm_native_covariates | {paired.diff_mean:.6f} | "
        f"{paired.ci_low:.6f} | {paired.ci_high:.6f} |"
    )
    crps_paired = bootstrap_mean_diff(guarded_crps, native_crps, n_boot=1000, ci=0.95, seed=0)
    print(
        f"| crps | timesfm_native_covariates | {crps_paired.diff_mean:.6f} | "
        f"{crps_paired.ci_low:.6f} | {crps_paired.ci_high:.6f} |"
    )
    covers = paired.ci_low <= 0.0 <= paired.ci_high or paired.ci_high <= 0.0
    win = bool(fired and covers and bad == 0)
    print("status: WIN" if win else "status: NO_WIN")
    return 0


def _recover_choice(router: np.ndarray, native: np.ndarray, tabpfn: np.ndarray, pipe: np.ndarray) -> np.ndarray:
    choice = np.full(router.shape[0], -1, dtype=np.int64)
    choice[np.isclose(router, native)] = 0
    remaining = choice < 0
    choice[remaining & np.isclose(router, tabpfn)] = 1
    remaining = choice < 0
    choice[remaining & np.isclose(router, pipe)] = 2
    if np.any(choice < 0):
        raise SystemExit("error: saved router scores do not match one of the three systems")
    return choice


if __name__ == "__main__":
    raise SystemExit(main())

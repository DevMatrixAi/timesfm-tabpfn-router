# timesfm-tabpfn-router

This is research code for the paper “When to Use a Time-Series Foundation Model, a Tabular Foundation Model, or Both: A Router with a Calendar Guard”.

The router chooses among three already-scored forecasters: TimesFM-3 with known-future covariates (`timesfm_native_covariates`), TabPFN-3.5 (`tabpfn`), and the same TabPFN table with two TimesFM horizon summaries appended (`feature_pipe`). A ridge model predicts each system’s MASE and picks the minimum. Before that choice, the calendar guard forces TimesFM with covariates when every known-future column is a low-cardinality calendar field (hour, day of week, month, or the same kind of periodic indicator, at most 31 distinct values).

TimesFM-3 and TabPFN-3.5 weights are not included and are under non-commercial licenses. TimesFM-3 (`google/timesfm-3.0-pytorch`) is under the TimesFM Non-Commercial License v1.0. TabPFN-3.5 is under the TabPFN-3.5 License v1.0. The MIT license in this repository covers the router and the calendar guard only. It does not relicense those weights, and this repository does not accept either weight license for you.

## Checkpoints

Download the official checkpoints yourself. Point the scoring script at the local files with environment variables. Do not commit the files or a token.

TimesFM-3 is loaded by `TimesFM3Forecaster.from_pretrained`. The default argument is the Hub id `google/timesfm-3.0-pytorch`. Set `TIMESFM_CHECKPOINT` to a local directory that is an official snapshot of that checkpoint (the directory `from_pretrained` accepts):

```bash
export TIMESFM_CHECKPOINT=checkpoints/timesfm-3.0-pytorch
```

Leave it unset to use the Hub id. The package then downloads into the Hugging Face cache on the machine that runs the script.

TabPFN-3.5 is constructed as `TabPFNRegressor` from the installed `tabpfn` package. Accept the TabPFN-3.5 license at https://ux.priorlabs.ai and export the token that library asks for. The variable name is `TABPFN_TOKEN`. This repository never sets it and does not read a token file. To load a checkpoint you already downloaded, set `TABPFN_MODEL_PATH` to a path that TabPFN 9.1.0 accepts as `model_path`:

```bash
export TABPFN_MODEL_PATH=checkpoints/tabpfn-3.5
```

Unset, `model_path` stays the library default `"auto"`. The library’s own cache directory is `TABPFN_MODEL_CACHE_DIR`.

## Setup

Versions pinned in `requirements.txt` are the ones used for the reported runs: NumPy 2.5.4, PyTorch 2.11.0, TimesFM 3.0.2, TabPFN 9.1.0, tabpfn-extensions 0.6.3, and pytest 9.1.1. The M4 scoring run used the CUDA 12.8 build of that PyTorch version (`2.11.0+cu128`). Install that wheel from the PyTorch CUDA 12.8 index before the rest, so a later install does not replace it:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

CPU checks of the router and the guard need NumPy and pytest only (`pip install -e ".[dev]"`). They do not load either foundation model.

## Synthetic router result

`scripts/route_forecasts.py` does not call TimesFM or TabPFN. It fits the ridge router on an entity split of series that are already scored, then prints the MASE/CRPS table and the paired bootstrap intervals (1000 resamples, 95 percent, seed 0). The scored files are not in this repository. Place them like this, with `<regime>` one of `synthetic_kappa1`, `synthetic_kappa0`, `synthetic_strong_kappa1`, and `synthetic_strong_kappa0`:

```text
artifacts/cache/<regime>/test.npz
artifacts/runs/<regime>/per_series.npz
artifacts/router/native_synthetic_strong_kappa1.npz
artifacts/router/native_synthetic_strong_kappa0.npz
```

`test.npz` must contain `history`, `known_future_events`, `group_id`, and `covariates`. `per_series.npz` must contain `timesfm_univariate__mase` and both `__mase` and `__crps` for `timesfm_native_covariates`, `tabpfn`, and `feature_pipe`. A strong-prior regime may omit the native keys and use the sidecar file instead. Each sidecar has arrays `mase` and `crps`.

From the repository root:

```bash
python scripts/route_forecasts.py
```

The router wins when it beats all three always-on systems on MASE and the interval of (router − baseline) lies entirely below 0. The script then prints `status: WIN`.

## M4 Hourly guarded result

`scripts/score_real_panel.py` downloads the public M4 Hourly files into `data/real/m4/` when they are absent (`Hourly-train.csv` and `M4-info.csv` from the M4-methods repository), scores the three systems on a temporal hold-out of the last 64 hours, and applies the calendar guard before the router. On this panel the known future is hour-of-day and day-of-week, so the guard fires and the router is not called. The run expects a CUDA device. From the repository root, with the checkpoint variables set if the weights are local:

```bash
python scripts/score_real_panel.py
```

It writes `artifacts/runs/real_m4_hourly/`. The `router` row in that file is the guarded choice.

`scripts/report_calendar_guard.py` does not forecast and does not fit TabPFN. It reads a saved score file and reprints the MASE/CRPS table, including the unguarded router and `router_guarded`, plus the paired interval of (router_guarded − timesfm_native_covariates):

```bash
python scripts/report_calendar_guard.py
```

That command expects `artifacts/runs/real_m4_hourly/per_series.npz` from a scoring run that stored the unguarded router, and the M4 csv files in `data/real/m4/`. Neither file is in this repository. A fresh `score_real_panel.py` run stores the guarded choice in the `router` row, so keep the unguarded score file aside if you want the side-by-side table. The guard wins when it fires, that paired interval covers 0 or lies entirely at or below 0, and the guarded choice is native on every series.

## Tests

```bash
pytest
```

## License

The router and calendar guard are MIT. See `LICENSE`. TimesFM-3 and TabPFN-3.5 weights are not included and stay under their own non-commercial licenses.

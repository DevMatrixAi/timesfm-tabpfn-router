"""Locked experiment constants.

The TimesFM-alone baseline is the checkpoint's own quantile forecast from
``TimesFM3Forecaster``. It is not the bridge head in ``models/bridge.py``.
"""

HORIZON = 64
PATCH_LENGTH = 32
MAX_GROUP_PERIOD = 96

TIMESFM_QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
N_QUANTILES = 9
MEDIAN_QUANTILE_INDEX = 4
WIDTH_LOW_QUANTILE_INDEX = 0
WIDTH_HIGH_QUANTILE_INDEX = 8

FEATURE_PIPE_COLUMN_NAMES = ("timesfm_q50", "timesfm_q90_minus_q10")
FEATURE_PIPE_REDUCTION = "mean_over_horizon"

DEFAULT_CONTEXT_BANK_SIZE = 8000
GATE_OPEN_TOL = 1e-3

PRIMARY_METRICS = ("mase", "crps")
KAPPA1_BEATS = ("timesfm_univariate", "tabpfn", "feature_pipe")
KAPPA0_TIE = ("bridge_no_cross_attention",)
KAPPA0_MUST_NOT_BEAT = ("timesfm_univariate",)

TS_DIM = 1280
TAB_DIM = 1024
D_MODEL = 256
N_BLOCKS = 2
N_HEADS = 8
DEFAULT_LR = 1e-4
MAX_TRAINABLE_PARAMETERS = 5_000_000

N_COVARIATES = 4
COVARIATE_NAMES = ("cov_level", "cov_trend", "cov_noise", "cov_season_amp")
STATIC_FEATURE_NAMES = ("group_id",) + COVARIATE_NAMES


def _check_quantile_grid() -> None:
    if len(TIMESFM_QUANTILES) != N_QUANTILES:
        raise RuntimeError("TimesFM quantile grid must have 9 levels")
    if tuple(sorted(TIMESFM_QUANTILES)) != TIMESFM_QUANTILES:
        raise RuntimeError("TimesFM quantiles must be sorted")
    if TIMESFM_QUANTILES[MEDIAN_QUANTILE_INDEX] != 0.5:
        raise RuntimeError("median index must be τ=0.5")
    if TIMESFM_QUANTILES[WIDTH_LOW_QUANTILE_INDEX] != 0.1:
        raise RuntimeError("width low index must be τ=0.1")
    if TIMESFM_QUANTILES[WIDTH_HIGH_QUANTILE_INDEX] != 0.9:
        raise RuntimeError("width high index must be τ=0.9")
    if HORIZON != 64:
        raise RuntimeError("horizon is locked to TimesFM-3's forecast patch of 64")


_check_quantile_grid()

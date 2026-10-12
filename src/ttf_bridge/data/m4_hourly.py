"""M4 Hourly panel with a temporal split and calendar side information.

Source: the public M4 Hourly training series and the M4 info file
(https://github.com/Mcompetitions/M4-methods, Dataset/Train/Hourly-train.csv
and Dataset/M4-info.csv). Each entity is one hourly series id (H1, H2, ...).
The series values are the Hourly-train observations. Every hourly series has
M4 category Other, so the static column is that category code. The
known-future columns are the hour of day and the day of week, built from
StartingDate. They are known before the horizon label.

The split is temporal. The test label is the last 64 observations. The test
history is the 512 observations before that. The training label is the 64
observations before the test label, with its own 512-observation history.
The two labels do not overlap. There is no shuffled split.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ttf_bridge.constants import HORIZON
from ttf_bridge.scaling import seasonal_naive_scale

CONTEXT = 512
SEASON = 24
MIN_LENGTH = CONTEXT + HORIZON + HORIZON
DATE_FORMAT = "%d-%m-%y %H:%M"
M4_HOURLY_URLS = {
    "Hourly-train.csv": (
        "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/"
        "Dataset/Train/Hourly-train.csv"
    ),
    "M4-info.csv": (
        "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/"
        "Dataset/M4-info.csv"
    ),
}


@dataclass(frozen=True)
class M4WindowBatch:
    """One temporal slice of the M4 Hourly panel."""

    entity_id: np.ndarray
    group_id: np.ndarray
    history: np.ndarray
    future: np.ndarray
    static: np.ndarray
    past_future: np.ndarray
    hour: np.ndarray
    mase_scale: np.ndarray


def load_m4_hourly(data_dir: Path) -> tuple[M4WindowBatch, M4WindowBatch]:
    """Return the training window and the later test window."""
    series = _read_series(data_dir / "Hourly-train.csv")
    starts = _read_starts(data_dir / "M4-info.csv")
    return build_m4_hourly_windows(series, starts)


def build_m4_hourly_windows(
    series: dict[str, np.ndarray],
    starts: dict[str, datetime],
) -> tuple[M4WindowBatch, M4WindowBatch]:
    """Build both windows from values and start timestamps already in memory."""
    ids = [entity for entity in sorted(series, key=_entity_sort_key) if entity in starts]
    short = [entity for entity in ids if series[entity].shape[0] < MIN_LENGTH]
    if short:
        raise ValueError(f"{len(short)} M4 hourly series are shorter than {MIN_LENGTH}")
    train_rows = [_window(series[entity], starts[entity], "train") for entity in ids]
    test_rows = [_window(series[entity], starts[entity], "test") for entity in ids]
    return _stack(ids, train_rows), _stack(ids, test_rows)


def known_future_column_names() -> tuple[str, ...]:
    """Hour and day-of-week at each horizon step."""
    hours = tuple(f"hour_{step}" for step in range(HORIZON))
    days = tuple(f"dow_{step}" for step in range(HORIZON))
    return hours + days


def _window(values: np.ndarray, start: datetime, which: str) -> dict[str, np.ndarray]:
    length = values.shape[0]
    test_future = length - HORIZON
    train_future = test_future - HORIZON
    origin = test_future if which == "test" else train_future
    history_start = origin - CONTEXT
    history = values[history_start:origin]
    future = values[origin : origin + HORIZON]
    if history.shape[0] != CONTEXT or future.shape[0] != HORIZON:
        raise ValueError("window bounds do not match the locked context and horizon")
    if which == "train" and origin + HORIZON != test_future:
        raise ValueError("training label must end where the test label starts")
    span = _calendar(start, history_start, CONTEXT + HORIZON)
    hour = span[CONTEXT:, 0]
    dow = span[CONTEXT:, 1]
    return {
        "history": history,
        "future": future,
        "past_future": np.stack([span[:, 0] / 23.0, span[:, 1] / 6.0], axis=0),
        "hour": hour,
        "static": np.concatenate([np.array([0.0]), hour, dow]),
    }


def _calendar(start: datetime, offset: int, length: int) -> np.ndarray:
    rows = np.empty((length, 2), dtype=np.float64)
    for index in range(length):
        stamp = start + timedelta(hours=offset + index)
        rows[index, 0] = stamp.hour
        rows[index, 1] = stamp.weekday()
    return rows


def _stack(ids: list[str], rows: list[dict[str, np.ndarray]]) -> M4WindowBatch:
    history = np.stack([row["history"] for row in rows], axis=0)
    return M4WindowBatch(
        entity_id=np.asarray(ids),
        group_id=np.zeros(len(ids), dtype=np.int64),
        history=history.astype(np.float64),
        future=np.stack([row["future"] for row in rows], axis=0).astype(np.float64),
        static=np.stack([row["static"] for row in rows], axis=0).astype(np.float64),
        past_future=np.stack([row["past_future"] for row in rows], axis=0).astype(np.float32),
        hour=np.stack([row["hour"] for row in rows], axis=0).astype(np.float64),
        mase_scale=seasonal_naive_scale(history, SEASON).astype(np.float64),
    )


def _read_series(path: Path) -> dict[str, np.ndarray]:
    series: dict[str, np.ndarray] = {}
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            values = [float(cell) for cell in row[1:] if cell.strip()]
            series[row[0]] = np.asarray(values, dtype=np.float64)
    if not series:
        raise ValueError(f"no series in {path}")
    return series


def _read_starts(path: Path) -> dict[str, datetime]:
    starts: dict[str, datetime] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["SP"] != "Hourly":
                continue
            starts[row["M4id"]] = datetime.strptime(row["StartingDate"], DATE_FORMAT)
    if not starts:
        raise ValueError(f"no hourly start dates in {path}")
    return starts


def _entity_sort_key(entity_id: str) -> int:
    if not entity_id.startswith("H"):
        raise ValueError(f"unexpected M4 hourly id {entity_id}")
    return int(entity_id[1:])

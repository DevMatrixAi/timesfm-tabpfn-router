"""Real panel loader. v1 keeps the interface and does not ship a dataset.

A later loader must split by time and keep future values out of the features.
The v1 result is the synthetic entity hold-out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class PanelWindows:
    """Windows a real-panel loader would return. Unused in v1."""

    entity_id: np.ndarray
    history: np.ndarray
    future: np.ndarray
    static_features: np.ndarray
    known_future: np.ndarray
    mase_scale: np.ndarray
    time_index: np.ndarray


class RealPanelLoader(Protocol):
    def load(self) -> PanelWindows:
        """Load a temporally split panel."""


def load_real_panel(source: str = "unavailable") -> PanelWindows:
    """v1 has no real panel. ``source`` is accepted so the call shape is stable."""
    raise NotImplementedError(
        "Real panel loading is not implemented in v1 "
        f"(source={source!r}). The v1 result is the synthetic entity hold-out."
    )

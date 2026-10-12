"""Paired error differences. Lower error is better for the bridge."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedComparison:
    """Mean of ``bridge_score - reference_score`` and its confidence interval.

    ``bridge_is_better`` is true only when the whole interval lies below zero.
    ``tied`` is true when the interval contains zero.
    """

    diff_mean: float
    ci_low: float
    ci_high: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.diff_mean, self.ci_low, self.ci_high)):
            raise ValueError("comparison values must be finite")
        if self.ci_low > self.ci_high:
            raise ValueError("ci_low must be <= ci_high")

    @property
    def bridge_is_better(self) -> bool:
        return self.ci_high < 0.0

    @property
    def tied(self) -> bool:
        return self.ci_low <= 0.0 <= self.ci_high

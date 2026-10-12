"""Gate snapshots shared by the bridge and the win rule.

Raw α and tanh(α) are both recorded for both directions. A TimesFM-head
forecast uses the TimesFM-direction gates. A TabPFN-head forecast uses the
TabPFN-direction gates. tanh(α) within ``GATE_OPEN_TOL`` of 0 on the head's
direction means that forecast is not using the other stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ttf_bridge.constants import GATE_OPEN_TOL


@dataclass(frozen=True)
class GateValue:
    block: int
    direction: str
    alpha: float
    tanh_alpha: float

    def __post_init__(self) -> None:
        if self.direction not in ("ts", "tab"):
            raise ValueError("gate direction must be 'ts' or 'tab'")
        if not math.isfinite(self.alpha) or not math.isfinite(self.tanh_alpha):
            raise ValueError("gate values must be finite")


def head_gate_direction(head_stream: str) -> str:
    """Direction whose gate carries the other stream into the forecast head."""
    if head_stream == "timesfm":
        return "ts"
    if head_stream == "tabpfn":
        return "tab"
    raise ValueError("head_stream must be 'timesfm' or 'tabpfn'")


def forecast_uses_cross_attention(
    gates: Sequence[GateValue],
    tol: float = GATE_OPEN_TOL,
    direction: str = "ts",
) -> bool:
    """True when ``direction`` has a gate far enough from 0 to open.

    ``direction`` is ``ts`` for the TimesFM head and ``tab`` for the TabPFN head.
    """
    if direction not in ("ts", "tab"):
        raise ValueError("gate direction must be 'ts' or 'tab'")
    return any(gate.direction == direction and abs(gate.tanh_alpha) > tol for gate in gates)


def format_gate_log(gates: Sequence[GateValue]) -> str:
    """One line per gate, including raw α and tanh(α)."""
    if not gates:
        return "gates: none"
    return "\n".join(
        (
            f"block={gate.block} direction={gate.direction} "
            f"alpha={gate.alpha:.8g} tanh_alpha={gate.tanh_alpha:.8g}"
        )
        for gate in gates
    )

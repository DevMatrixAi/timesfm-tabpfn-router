"""v1 does not load a real panel."""

from __future__ import annotations

import pytest

from ttf_bridge.data.real import load_real_panel


def test_real_panel_stays_an_interface() -> None:
    with pytest.raises(NotImplementedError, match="synthetic entity hold-out"):
        load_real_panel("local-csv")

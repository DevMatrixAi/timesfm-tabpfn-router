"""TabPFN context-bank size. The configured value is a cap, not a promise."""

from __future__ import annotations


def resolve_context_bank_size(requested: int, n_available: int) -> int:
    """Return the bank size that will actually be fit.

    ``requested`` is the config cap (default 8_000 on the full runs).
    The actual size is the smaller of that cap and the training rows on hand.
    """
    if requested < 1:
        raise ValueError(f"tabpfn_context_bank_size must be positive, got {requested}")
    if n_available < 1:
        raise ValueError(f"no training rows are available to build a context bank, got {n_available}")
    return min(int(requested), int(n_available))


def format_context_bank_log(requested: int, actual: int, n_available: int) -> str:
    return (
        "tabpfn_context_bank "
        f"requested={int(requested)} available={int(n_available)} actual={int(actual)}"
    )

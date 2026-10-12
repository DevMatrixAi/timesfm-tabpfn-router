"""License boundary for this repository.

The router and calendar guard are under the MIT License (see LICENSE).
TimesFM-3 weights (``google/timesfm-3.0-pytorch``) are under the TimesFM
Non-Commercial License v1.0. TabPFN-3.5 weights are under the TabPFN-3.5
License v1.0. Those weights are not included here, and this code does not
relicense them.
"""

BANNER = (
    "TimesFM-3 weights are under the TimesFM Non-Commercial License v1.0. "
    "TabPFN-3.5 weights are under the TabPFN-3.5 License v1.0. "
    "Those weights are not included in this repository. "
    "The router and calendar guard are under the MIT License and do not "
    "relicense the weights."
)


def assert_research_use(non_commercial_flag: object) -> None:
    """Refuse a run that has not acknowledged the weight licenses."""
    if non_commercial_flag is not True:
        raise RuntimeError(
            "Refusing to run without non_commercial: true. "
            "The foundation-model weights are under non-commercial licenses "
            "and are not included in this repository. " + BANNER
        )

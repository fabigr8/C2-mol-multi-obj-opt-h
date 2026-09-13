"""DrugBank-approved percentile normalization (brief SS3.1, decided in Phase 0):
maps a raw predictor value to [0, 1] via its percentile rank against a fixed
reference distribution, direction-aware ("min" components are inverted so
lower raw values score higher). Identical transform must be used for every arm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_VALID_DIRECTIONS = {"max", "min"}


def percentile_rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Fraction of `reference` <= each value, in [0, 1], with mean-rank tie handling."""
    sorted_ref = np.sort(np.asarray(reference, dtype=float))
    n = len(sorted_ref)
    values = np.asarray(values, dtype=float)
    lo = np.searchsorted(sorted_ref, values, side="left")
    hi = np.searchsorted(sorted_ref, values, side="right")
    return (lo + hi) / (2 * n)


class DrugBankPercentileNormalizer:
    """Reference distributions keyed by component name (one column per component)."""

    def __init__(self, reference: pd.DataFrame):
        self._reference = reference

    @classmethod
    def from_admet_ai_drugbank(cls) -> "DrugBankPercentileNormalizer":
        from admet_ai.constants import DEFAULT_DRUGBANK_PATH

        return cls(pd.read_csv(DEFAULT_DRUGBANK_PATH))

    def normalize(self, component: str, direction: str, values: np.ndarray) -> np.ndarray:
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(f"direction must be one of {_VALID_DIRECTIONS}, got {direction!r}")
        if component not in self._reference.columns:
            raise KeyError(f"no DrugBank reference distribution for component {component!r}")
        percentile = percentile_rank(self._reference[component].to_numpy(dtype=float), values)
        return percentile if direction == "max" else 1.0 - percentile

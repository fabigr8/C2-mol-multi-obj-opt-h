"""Composite objective (brief SS3.1): per-component DrugBank-percentile
normalization to [0, 1], combined via weighted geometric mean so no single
component can be fully sacrificed. The aggregation function is identical
across all arms -- only constraints (Phase 5) differ.
"""

from __future__ import annotations

import pandas as pd

from rhmoo.config import ObjectiveConfig
from rhmoo.normalization import DrugBankPercentileNormalizer


def weighted_geometric_mean(scores: dict[str, float], weights: dict[str, float]) -> float:
    if scores.keys() != weights.keys():
        raise ValueError("scores and weights must cover the same components")
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive number")

    product = 1.0
    for name, score in scores.items():
        if score < 0:
            raise ValueError(f"component {name!r} score must be >= 0, got {score!r}")
        product *= score ** (weights[name] / total_weight)
    return product


def normalize_components(
    config: ObjectiveConfig,
    normalizer: DrugBankPercentileNormalizer,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """`raw` must have one column per configured component with unnormalized values."""
    normalized = pd.DataFrame(index=raw.index)
    for component in config.components:
        normalized[component.name] = normalizer.normalize(
            component.name, component.direction, raw[component.name].to_numpy()
        )
    return normalized


def aggregate(config: ObjectiveConfig, normalized: pd.DataFrame) -> pd.Series:
    """Weighted geometric mean of normalized per-component scores, one row per molecule."""
    if config.aggregation != "weighted_geometric_mean":
        raise NotImplementedError(f"unsupported aggregation method: {config.aggregation!r}")
    weights = {c.name: c.weight for c in config.components}
    return normalized.apply(lambda row: weighted_geometric_mean(row.to_dict(), weights), axis=1)

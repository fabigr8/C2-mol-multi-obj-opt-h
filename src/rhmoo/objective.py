"""Composite objective (brief SS3.1): per-component DrugBank-percentile
normalization to [0, 1], combined via weighted geometric mean so no single
component can be fully sacrificed. The aggregation function is identical
across all arms within a comparison -- only constraints differ.

The alternative aggregators exist for the brief SS6.2 ablation, which tests
whether the geometric mean's "no component can be fully sacrificed" property
actually buys anything. They are a config knob, never a silent default:
`weighted_geometric_mean` is the declared default everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rhmoo.config import ObjectiveConfig
from rhmoo.normalization import DrugBankPercentileNormalizer

#: The only normalization this module implements. Declared in config and
#: checked, so an unrecognized value fails loudly instead of being ignored.
SUPPORTED_NORMALIZATION = "drugbank_percentile"

AGGREGATION_METHODS = ("weighted_geometric_mean", "arithmetic_mean", "soft_min")


def weighted_geometric_mean(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Scalar reference implementation. `aggregate` uses the vectorized path;
    this stays as the readable definition the golden-value tests pin.
    """
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
    if config.normalization != SUPPORTED_NORMALIZATION:
        raise NotImplementedError(
            f"unsupported normalization: {config.normalization!r} "
            f"(only {SUPPORTED_NORMALIZATION!r} is implemented)"
        )
    normalized = pd.DataFrame(index=raw.index)
    for component in config.components:
        normalized[component.name] = normalizer.normalize(
            component.name, component.direction, raw[component.name].to_numpy()
        )
    return normalized


def _normalized_weights(config: ObjectiveConfig) -> tuple[list[str], np.ndarray]:
    names = [c.name for c in config.components]
    weights = np.array([c.weight for c in config.components], dtype=float)
    total = weights.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    return names, weights / total


def aggregate(config: ObjectiveConfig, normalized: pd.DataFrame) -> pd.Series:
    """Aggregate normalized per-component scores into one composite per molecule.

    Vectorized: the GA calls this on the order of 10^5 times per experiment, so
    a per-row `.apply` is a meaningful share of the non-inference runtime.
    """
    if config.aggregation not in AGGREGATION_METHODS:
        raise NotImplementedError(
            f"unsupported aggregation method: {config.aggregation!r} (expected one of {AGGREGATION_METHODS})"
        )

    names, weights = _normalized_weights(config)
    missing = [name for name in names if name not in normalized.columns]
    if missing:
        raise ValueError(f"normalized frame is missing configured components: {missing}")

    values = normalized[names].to_numpy(dtype=float)
    if np.any(values < 0):
        raise ValueError("component scores must be >= 0")

    if config.aggregation == "weighted_geometric_mean":
        # log(0) is -inf, so a zero component correctly zeroes the product --
        # that is the property the geometric mean is chosen for. NaN-safe:
        # a NaN component propagates to a NaN composite rather than being skipped.
        with np.errstate(divide="ignore"):
            result = np.exp((weights * np.log(values)).sum(axis=1))
    elif config.aggregation == "arithmetic_mean":
        result = (weights * values).sum(axis=1)
    else:  # soft_min
        tau = config.aggregation_params.get("soft_min_tau", 0.1)
        if tau <= 0:
            raise ValueError(f"soft_min_tau must be > 0, got {tau!r}")
        # -tau * log(sum_i w_i * exp(-s_i / tau)): tau -> 0 recovers min(),
        # tau -> inf recovers the weighted arithmetic mean, so the SS6.2
        # ablation spans a continuous family rather than three unrelated rules.
        shifted = -values / tau
        offset = shifted.max(axis=1, keepdims=True)
        log_sum = offset.ravel() + np.log((weights * np.exp(shifted - offset)).sum(axis=1))
        result = -tau * log_sum

    return pd.Series(result, index=normalized.index)

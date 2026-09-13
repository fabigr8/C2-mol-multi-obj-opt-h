import pandas as pd
import pytest

from rhmoo.config import ObjectiveComponent, ObjectiveConfig
from rhmoo.objective import aggregate, normalize_components, weighted_geometric_mean
from rhmoo.normalization import DrugBankPercentileNormalizer

pytestmark = pytest.mark.smoke


def test_weighted_geometric_mean_golden_value():
    # sqrt(0.81 * 0.25) = 0.45
    score = weighted_geometric_mean({"a": 0.81, "b": 0.25}, {"a": 0.5, "b": 0.5})
    assert score == pytest.approx(0.45)


def test_weighted_geometric_mean_equal_scores_unaffected_by_weights():
    score = weighted_geometric_mean({"a": 0.7, "b": 0.7}, {"a": 0.8, "b": 0.2})
    assert score == pytest.approx(0.7)


def test_weighted_geometric_mean_zero_component_zeros_result():
    score = weighted_geometric_mean({"a": 1.0, "b": 0.0}, {"a": 0.8, "b": 0.2})
    assert score == pytest.approx(0.0)


def test_weighted_geometric_mean_mismatched_keys_raises():
    with pytest.raises(ValueError):
        weighted_geometric_mean({"a": 1.0}, {"a": 0.5, "b": 0.5})


def test_weighted_geometric_mean_rejects_negative_score():
    with pytest.raises(ValueError):
        weighted_geometric_mean({"a": -0.1, "b": 1.0}, {"a": 0.5, "b": 0.5})


def _config(aggregation: str = "weighted_geometric_mean") -> ObjectiveConfig:
    return ObjectiveConfig(
        components=(
            ObjectiveComponent(name="X", source="rdkit", direction="max", weight=0.5),
            ObjectiveComponent(name="Y", source="rdkit", direction="min", weight=0.5),
        ),
        normalization="drugbank_percentile",
        aggregation=aggregation,
        cache_path="unused.parquet",
    )


def test_normalize_and_aggregate_end_to_end():
    config = _config()
    reference = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0, 5.0], "Y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    normalizer = DrugBankPercentileNormalizer(reference)
    raw = pd.DataFrame({"X": [3.0], "Y": [3.0]})

    normalized = normalize_components(config, normalizer, raw)
    assert normalized.loc[0, "X"] == pytest.approx(0.5)
    assert normalized.loc[0, "Y"] == pytest.approx(0.5)

    scores = aggregate(config, normalized)
    assert scores.iloc[0] == pytest.approx(0.5)


def test_aggregate_rejects_unsupported_method():
    config = _config(aggregation="arithmetic_mean")
    normalized = pd.DataFrame({"X": [0.5], "Y": [0.5]})
    with pytest.raises(NotImplementedError):
        aggregate(config, normalized)

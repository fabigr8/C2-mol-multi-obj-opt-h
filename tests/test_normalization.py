import numpy as np
import pandas as pd
import pytest

from rhmoo.normalization import DrugBankPercentileNormalizer, percentile_rank

pytestmark = pytest.mark.smoke

REFERENCE = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def test_percentile_rank_golden_values():
    result = percentile_rank(REFERENCE, np.array([0, 1, 3, 5, 6]))
    # below min -> 0; at min -> 0.1; median -> 0.5; at max -> 0.9; above max -> 1.0
    np.testing.assert_allclose(result, [0.0, 0.1, 0.5, 0.9, 1.0])


def test_normalizer_max_direction_matches_percentile():
    normalizer = DrugBankPercentileNormalizer(pd.DataFrame({"X": REFERENCE}))
    result = normalizer.normalize("X", "max", np.array([3.0]))
    np.testing.assert_allclose(result, [0.5])


def test_normalizer_min_direction_inverts_percentile():
    normalizer = DrugBankPercentileNormalizer(pd.DataFrame({"X": REFERENCE}))
    low = normalizer.normalize("X", "min", np.array([1.0]))
    high = normalizer.normalize("X", "min", np.array([5.0]))
    np.testing.assert_allclose(low, [0.9])
    np.testing.assert_allclose(high, [0.1])


def test_normalizer_unknown_component_raises():
    normalizer = DrugBankPercentileNormalizer(pd.DataFrame({"X": REFERENCE}))
    with pytest.raises(KeyError):
        normalizer.normalize("Y", "max", np.array([1.0]))


def test_normalizer_invalid_direction_raises():
    normalizer = DrugBankPercentileNormalizer(pd.DataFrame({"X": REFERENCE}))
    with pytest.raises(ValueError):
        normalizer.normalize("X", "sideways", np.array([1.0]))

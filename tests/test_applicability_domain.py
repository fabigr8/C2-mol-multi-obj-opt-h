import numpy as np
import pytest

from rhmoo.applicability_domain import (
    ReferenceFingerprintIndex,
    bulk_ecfp4_bits,
    bulk_max_tanimoto,
    soft_penalty_above_threshold_is_bad,
)

pytestmark = pytest.mark.smoke


def test_bulk_max_tanimoto_hand_computed():
    reference = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    query = np.array([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.uint8)

    result = bulk_max_tanimoto(query, reference)

    # row 0 is identical to reference row 0 -> similarity 1.0
    assert result[0] == pytest.approx(1.0)
    # row 1: vs ref0 intersection=1,union=1+2-1=2 -> 0.5; vs ref1 intersection=0 -> 0.0
    assert result[1] == pytest.approx(0.5)


def test_bulk_max_tanimoto_all_zero_fingerprint_is_zero_not_nan():
    reference = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    query = np.array([[0, 0, 0, 0]], dtype=np.uint8)

    result = bulk_max_tanimoto(query, reference)

    assert result[0] == 0.0


def test_soft_penalty_equals_one_at_and_above_theta():
    values = np.array([0.3, 0.4, 0.5, 1.0])
    penalty = soft_penalty_above_threshold_is_bad(values, theta=0.4)
    np.testing.assert_allclose(penalty[1:], [1.0, 1.0, 1.0])


def test_soft_penalty_decays_continuously_below_theta():
    penalty = soft_penalty_above_threshold_is_bad(np.array([0.2, 0.4]), theta=0.4)
    assert penalty[0] == pytest.approx(0.5)
    assert penalty[1] == pytest.approx(1.0)


def test_bulk_ecfp4_bits_identical_for_equivalent_smiles():
    bits = bulk_ecfp4_bits(["CCO", "OCC", "not_a_smiles"])
    assert np.array_equal(bits[0], bits[1])
    assert bits[2].sum() == 0  # invalid SMILES -> all-zero row


def test_reference_fingerprint_index_from_csv_and_cache(tmp_path):
    csv_path = tmp_path / "reference.csv"
    csv_path.write_text("smiles\nCCO\nc1ccccc1\n")
    cache_path = tmp_path / "cache.npz"

    index = ReferenceFingerprintIndex.from_csv(csv_path, cache_path=cache_path)
    assert cache_path.exists()

    result = index.max_tanimoto(["CCO"])
    assert result[0] == pytest.approx(1.0)

    def _fail(*args, **kwargs):
        raise AssertionError("should not recompute fingerprints when a cache hit is available")

    import rhmoo.applicability_domain as ad_module

    original = ad_module.bulk_ecfp4_bits
    ad_module.bulk_ecfp4_bits = _fail
    try:
        cached_index = ReferenceFingerprintIndex.from_csv(csv_path, cache_path=cache_path)
    finally:
        ad_module.bulk_ecfp4_bits = original

    assert cached_index.smiles == index.smiles
    np.testing.assert_array_equal(cached_index.bits, index.bits)

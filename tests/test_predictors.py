"""Not marked smoke: exercises the real ADMET-AI model (Chemprop checkpoint
load + inference), so it's slower than the rest of the suite. Runs under
`make test`, not `make smoke`.
"""

import pandas as pd
import pytest

from rhmoo.predictors import ADMET_COMPONENTS, ADMETPredictor, canonicalize, compute_qed


def test_canonicalize_and_qed_basic():
    assert canonicalize("CCO") == canonicalize("OCC")
    assert canonicalize("not a smiles") is None
    assert 0.0 <= compute_qed("CCO") <= 1.0


def test_predict_realigns_handles_invalid_and_dedups(tmp_path):
    predictor = ADMETPredictor(cache_path=tmp_path / "cache.parquet")
    smiles = ["CCO", "not_a_smiles", "OCC"]  # OCC canonicalizes to the same molecule as CCO
    result = predictor.predict(smiles)

    assert len(result) == 3
    assert result["valid"].tolist() == [True, False, True]
    assert result.loc[0, "canonical_smiles"] == result.loc[2, "canonical_smiles"]
    assert result.loc[0, "QED"] == pytest.approx(result.loc[2, "QED"])
    assert pd.isna(result.loc[1, "QED"])
    for col in ADMET_COMPONENTS:
        assert col in result.columns


def test_predict_uses_persistent_cache_across_instances(tmp_path):
    cache_path = tmp_path / "cache.parquet"
    first = ADMETPredictor(cache_path=cache_path)
    first.predict(["CCO"])
    assert cache_path.exists()

    second = ADMETPredictor(cache_path=cache_path)

    def _fail_load_model():
        raise AssertionError("should not need to load the model for a fully cached request")

    second._load_model = _fail_load_model
    result = second.predict(["CCO"])
    assert bool(result.loc[0, "valid"])

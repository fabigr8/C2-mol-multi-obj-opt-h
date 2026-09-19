import json

import numpy as np
import pandas as pd
import pytest

from rhmoo.independent_scorer import FEATURE_COLUMNS, MODEL_DIR, IndependentScorer, featurize

pytestmark = pytest.mark.smoke


def test_featurize_shape_and_determinism():
    smiles = ["CCO", "c1ccccc1"]
    a = featurize(smiles)
    b = featurize(smiles)
    assert list(a.columns) == list(FEATURE_COLUMNS)
    assert a.shape == (2, len(FEATURE_COLUMNS))
    pd.testing.assert_frame_equal(a, b)


def test_featurize_invalid_smiles_is_all_nan_row():
    features = featurize(["CCO", "not_a_smiles"])
    assert features.iloc[1].isna().all()
    assert not features.iloc[0].isna().all()


def test_metadata_records_held_out_metrics_and_feature_columns():
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
    assert metadata["feature_columns"] == list(FEATURE_COLUMNS)
    assert set(metadata["held_out_metrics"]) == {"hERG", "Solubility_AqSolDB"}
    assert 0.0 <= metadata["held_out_metrics"]["hERG"]["held_out_score"] <= 1.0


def test_independent_scorer_scores_valid_and_invalid_smiles():
    scorer = IndependentScorer()
    result = scorer.score(["CCO", "c1ccccc1", "not_a_smiles"])

    assert list(result.columns) == ["hERG_independent", "Solubility_AqSolDB_independent"]
    assert result.loc[0, "hERG_independent"] == pytest.approx(result.loc[0, "hERG_independent"])
    assert 0.0 <= result.loc[0, "hERG_independent"] <= 1.0
    assert np.isnan(result.loc[2, "hERG_independent"])
    assert np.isnan(result.loc[2, "Solubility_AqSolDB_independent"])

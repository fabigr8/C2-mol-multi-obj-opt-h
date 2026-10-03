import json

import numpy as np
import pandas as pd
import pytest

from rhmoo.independent_scorer import (
    FEATURE_COLUMNS,
    HALF_SPLIT_SEED,
    INDEPENDENT_ENDPOINTS,
    VARIANTS,
    IndependentScorer,
    IndependentScorerSet,
    featurize,
    half_assignment,
    half_mask,
    independent_column,
    independent_pairs,
    model_dir,
    split_independent_column,
)

pytestmark = pytest.mark.smoke

TDC_DIR = "data/raw/tdc/admet_group"
_TDC_NAMES = {"hERG": "herg", "Solubility_AqSolDB": "solubility_aqsoldb"}


def _canonical_set(smiles_list) -> set[str]:
    from rhmoo.predictors import canonicalize

    return {c for c in (canonicalize(s) for s in smiles_list) if c is not None}


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


def test_independent_column_names_round_trip():
    column = independent_column("Solubility_AqSolDB", "data_control_test")
    # The endpoint name itself contains underscores, so the variant separator
    # has to survive that.
    assert split_independent_column(column) == ("Solubility_AqSolDB", "data_control_test")
    assert independent_pairs("model_control") == {
        "hERG": "hERG_independent__model_control",
        "Solubility_AqSolDB": "Solubility_AqSolDB_independent__model_control",
    }
    with pytest.raises(ValueError):
        split_independent_column("hERG")


@pytest.mark.parametrize("variant", VARIANTS)
def test_metadata_records_held_out_metrics_and_feature_columns(variant):
    metadata = json.loads((model_dir(variant) / "metadata.json").read_text())
    assert metadata["variant"] == variant
    assert metadata["feature_columns"] == list(FEATURE_COLUMNS)
    assert set(metadata["held_out_metrics"]) == set(INDEPENDENT_ENDPOINTS)
    for endpoint, recorded in metadata["held_out_metrics"].items():
        assert recorded["n_train"] > 0
        assert recorded["n_eval"] > 0
        # The gate is pre-declared in scripts/train_independent_scorer.py; the
        # outcome is recorded either way, so this asserts it was evaluated, not
        # that it passed.
        assert isinstance(recorded["passes_competence_gate"], bool)
        assert recorded["metric"] in {"roc_auc", "mae"}


@pytest.mark.parametrize("variant", VARIANTS)
def test_scorer_scores_valid_and_invalid_smiles(variant):
    scorer = IndependentScorer(variant)
    result = scorer.score(["CCO", "c1ccccc1", "not_a_smiles"])

    assert list(result.columns) == [independent_column(e, variant) for e in INDEPENDENT_ENDPOINTS]
    herg = independent_column("hERG", variant)
    assert 0.0 <= result.loc[0, herg] <= 1.0
    # Invalid SMILES must survive as a NaN row, never be dropped (brief SS9).
    assert result[herg].isna().tolist() == [False, False, True]


def test_scorer_set_scores_every_variant_over_one_featurization(tmp_path):
    scorers = IndependentScorerSet(cache_path=tmp_path / "independent.parquet")
    result = scorers.score(["CCO", "c1ccccc1"])

    expected = [independent_column(e, v) for v in VARIANTS for e in INDEPENDENT_ENDPOINTS]
    assert sorted(result.columns) == sorted(expected)
    assert len(result) == 2

    # The set must agree with each variant scored on its own -- the shared
    # featurization is an optimization, not a different computation.
    for variant in VARIANTS:
        alone = IndependentScorer(variant).score(["CCO", "c1ccccc1"])
        for endpoint in INDEPENDENT_ENDPOINTS:
            column = independent_column(endpoint, variant)
            np.testing.assert_allclose(result[column].to_numpy(), alone[column].to_numpy())


def test_scorer_set_cache_is_reused_without_refeaturizing(tmp_path, monkeypatch):
    cache_path = tmp_path / "independent.parquet"
    IndependentScorerSet(cache_path=cache_path).score(["CCO"])
    assert cache_path.exists()

    reloaded = IndependentScorerSet(cache_path=cache_path)

    def _fail(*args, **kwargs):
        raise AssertionError("a fully cached request must not re-featurize")

    monkeypatch.setattr("rhmoo.independent_scorer.featurize", _fail)
    result = reloaded.score(["CCO"])
    assert not result.isna().all(axis=None)


def test_half_split_is_canonical_and_deterministic():
    # Same molecule, two spellings -> the same half. This is the leak the
    # zero-overlap test below exists to rule out.
    assert half_assignment("CCO") == half_assignment("OCC")
    assert half_assignment("CCO") == half_assignment("CCO", HALF_SPLIT_SEED)
    assert half_mask(["CCO", "OCC"], "A").tolist() in ([True, True], [False, False])


@pytest.mark.parametrize("endpoint", INDEPENDENT_ENDPOINTS)
def test_train_val_halves_do_not_overlap(endpoint):
    """brief SS7: zero SMILES overlap between the objective-side and
    data-control training splits.
    """
    train_val = pd.read_csv(f"{TDC_DIR}/{_TDC_NAMES[endpoint]}/train_val.csv")
    drugs = train_val["Drug"].tolist()
    half_a = _canonical_set([s for s, keep in zip(drugs, half_mask(drugs, "A")) if keep])
    half_b = _canonical_set([s for s, keep in zip(drugs, half_mask(drugs, "B")) if keep])

    assert half_a and half_b
    assert half_a & half_b == set()
    assert half_a | half_b == _canonical_set(drugs)


@pytest.mark.parametrize("endpoint", INDEPENDENT_ENDPOINTS)
def test_tdc_train_val_and_test_do_not_overlap(endpoint):
    """`data_control_test` trains on `test`, so its disjointness from
    ADMET-AI's training data rests entirely on this holding.
    """
    tdc = _TDC_NAMES[endpoint]
    train_val = _canonical_set(pd.read_csv(f"{TDC_DIR}/{tdc}/train_val.csv")["Drug"].tolist())
    test = _canonical_set(pd.read_csv(f"{TDC_DIR}/{tdc}/test.csv")["Drug"].tolist())
    assert train_val & test == set()


@pytest.mark.parametrize("endpoint", INDEPENDENT_ENDPOINTS)
def test_data_control_half_is_not_disjoint_from_admet_ai_training_data(endpoint):
    """Deliberately asserts a *limitation*, so nobody later "corrects" the claim
    in the wrong direction.

    ADMET-AI is a fixed pretrained artifact fit on the whole TDC `train_val`
    split, and brief SS2 forbids retraining the objective's predictors. So
    `data_control_half`, which trains on half of `train_val`, is necessarily a
    subset of ADMET-AI's training data -- it is a data control only relative to
    a hypothetical half-trained objective. Only `data_control_test` is
    genuinely disjoint. See docs/LIMITATIONS.md.
    """
    tdc = _TDC_NAMES[endpoint]
    admet_ai_training_data = _canonical_set(pd.read_csv(f"{TDC_DIR}/{tdc}/train_val.csv")["Drug"].tolist())
    drugs = pd.read_csv(f"{TDC_DIR}/{tdc}/train_val.csv")["Drug"].tolist()
    half_b = _canonical_set([s for s, keep in zip(drugs, half_mask(drugs, "B")) if keep])

    assert half_b <= admet_ai_training_data
    assert half_b & admet_ai_training_data == half_b

    half_metadata = json.loads((model_dir("data_control_half") / "metadata.json").read_text())
    assert half_metadata["disjoint_from_admet_ai_training_data"] is False
    test_metadata = json.loads((model_dir("data_control_test") / "metadata.json").read_text())
    assert test_metadata["disjoint_from_admet_ai_training_data"] is True

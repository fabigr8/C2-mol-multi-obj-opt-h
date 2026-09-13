import numpy as np
import pytest

from rhmoo.synthesizability import sa_penalty, sa_score

pytestmark = pytest.mark.smoke


def test_sa_score_simple_molecule_is_low():
    # empirically ~1.98 with the bundled fpscores model; assert a loose bound
    # rather than pin the exact float to avoid coupling to model internals.
    assert 1.0 <= sa_score("CCO") <= 3.0


def test_sa_score_invalid_smiles_returns_none():
    assert sa_score("not_a_smiles") is None


def test_sa_score_more_complex_molecule_scores_higher_than_ethanol():
    assert sa_score("CC(C)Cc1ccc(cc1)C(C)C(=O)O") > sa_score("CCO")


def test_sa_penalty_equals_one_at_and_below_threshold():
    penalty = sa_penalty(np.array([2.0, 3.0]), threshold=3.0)
    np.testing.assert_allclose(penalty, [1.0, 1.0])


def test_sa_penalty_decays_continuously_above_threshold():
    penalty = sa_penalty(np.array([6.0]), threshold=3.0)
    assert penalty[0] == pytest.approx(0.5)

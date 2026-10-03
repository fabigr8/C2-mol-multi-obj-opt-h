import numpy as np
import pandas as pd
import pytest

from rhmoo.config import load_config
from rhmoo.decoupling import (
    decoupling_curves,
    decoupling_generation,
    decoupling_summary,
    restricted_objective,
)
from rhmoo.independent_scorer import INDEPENDENT_ENDPOINTS, independent_column

pytestmark = pytest.mark.smoke


def _curve(gap_deltas):
    return pd.DataFrame({"generation": range(len(gap_deltas)), "gap_delta": gap_deltas})


def test_detects_the_first_sustained_crossing():
    curve = _curve([0, 0.01, 0.02, 0.05, 0.09, 0.12, 0.13, 0.15, 0.2, 0.3])
    assert decoupling_generation(curve, margin=0.10, persistence=3) == 5


def test_a_transient_spike_does_not_count_as_decoupling():
    """Persistence is what separates a signal from a one-generation excursion."""
    curve = _curve([0, 0.5, 0, 0, 0.5, 0.5])
    assert decoupling_generation(curve, margin=0.1, persistence=3) is None
    assert decoupling_generation(curve, margin=0.1, persistence=2) == 4


def test_never_decoupling_is_a_reportable_outcome_not_an_error():
    """brief SS9: report the null."""
    curve = _curve([0, 0.01, 0.02, 0.01])
    assert decoupling_generation(curve, margin=0.1, persistence=2) is None


def test_restricted_objective_keeps_only_the_covered_endpoints():
    config = load_config("configs/main.yaml")
    restricted = restricted_objective(config)
    names = [c.name for c in restricted.components]

    assert set(names) == set(INDEPENDENT_ENDPOINTS)
    # Uncovered endpoints must be excluded -- no independent scorer can speak to
    # them, so including them would compare incomparable things.
    assert "Caco2_Wang" not in names
    assert "QED" not in names
    # Original weights are preserved; aggregate() renormalises them.
    original = {c.name: c.weight for c in config.objective.components}
    assert {c.name: c.weight for c in restricted.components} == {n: original[n] for n in names}


def test_curves_use_the_delta_so_a_constant_oracle_offset_cancels():
    """The scorers are imperfect, so a constant objective-minus-independent gap
    is oracle error, not exploitation (brief SS9).
    """
    from rhmoo.normalization import DrugBankPercentileNormalizer

    config = load_config("configs/main.yaml")
    variant = "model_control"
    smiles = ["CCO", "c1ccccc1", "CCN"]

    elites = pd.DataFrame(
        {"generation": [0, 0, 1, 1, 2, 2], "rank": [0, 1] * 3, "smiles": smiles * 2}
    )
    # A fixed offset between the two sources at every generation.
    scores = pd.DataFrame(
        {
            "smiles": smiles,
            "hERG": [0.2, 0.3, 0.4],
            "Solubility_AqSolDB": [-2.0, -3.0, -1.0],
            independent_column("hERG", variant): [0.2, 0.3, 0.4],
            independent_column("Solubility_AqSolDB", variant): [-2.0, -3.0, -1.0],
        }
    )

    normalizer = DrugBankPercentileNormalizer.from_admet_ai_drugbank()
    curve = decoupling_curves(elites, scores, config, normalizer, variant)

    assert list(curve["generation"]) == [0, 1, 2]
    # Identical inputs -> identical composites -> zero gap, and a flat delta.
    np.testing.assert_allclose(curve["gap"].to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(curve["gap_delta"].to_numpy(), 0.0, atol=1e-12)
    assert decoupling_generation(curve, 0.1, 2) is None


def test_summary_covers_the_declared_margin_sweep_and_marks_the_primary():
    config = load_config("configs/main.yaml")
    curve = pd.DataFrame(
        {"generation": range(8), "gap_delta": [0, 0.02, 0.06, 0.08, 0.12, 0.14, 0.16, 0.2]}
    )
    rows = decoupling_summary(curve, config, "data_control_half")

    assert [r["margin"] for r in rows] == list(config.decoupling.margin_sweep)
    primary = [r for r in rows if r["is_primary_margin"]]
    assert len(primary) == 1
    assert primary[0]["margin"] == config.decoupling.margin
    # A looser margin cannot decouple later than a tighter one.
    detected = [r["decoupling_generation"] for r in rows if r["decoupled"]]
    assert detected == sorted(detected)

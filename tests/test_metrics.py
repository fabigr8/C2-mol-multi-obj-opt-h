import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from rhmoo.applicability_domain import ReferenceFingerprintIndex
from rhmoo.metrics import (
    component_breakdown,
    divergence_headline,
    implausibility_battery,
    independent_divergence,
    longest_aliphatic_chain,
    nearest_neighbor_similarity,
    objective_trajectory,
    physchem_panel,
    sa_scores,
    select_top_k,
    structural_alerts,
)

pytestmark = pytest.mark.smoke


def test_longest_aliphatic_chain_hexane():
    assert longest_aliphatic_chain(Chem.MolFromSmiles("CCCCCC")) == 6


def test_longest_aliphatic_chain_all_ring_atoms_is_zero():
    assert longest_aliphatic_chain(Chem.MolFromSmiles("C1CCCCC1")) == 0


def test_longest_aliphatic_chain_aromatic_ring_is_zero():
    assert longest_aliphatic_chain(Chem.MolFromSmiles("c1ccccc1")) == 0


def test_longest_aliphatic_chain_substituent_on_ring():
    # cyclohexane with a 4-carbon chain substituent: ring atoms excluded, chain counted.
    assert longest_aliphatic_chain(Chem.MolFromSmiles("C1CCCCC1CCCC")) == 4


def test_longest_aliphatic_chain_toluene_methyl():
    assert longest_aliphatic_chain(Chem.MolFromSmiles("Cc1ccccc1")) == 1


def test_physchem_panel_valid_and_invalid():
    panel = physchem_panel(["CCO", "not_a_smiles"])
    assert panel.loc[0, "molecular_weight"] == pytest.approx(46.07, abs=0.1)
    assert panel.loc[0, "hbd"] == 1
    assert panel.loc[0, "formal_charge"] == 0
    assert panel.loc[1].isna().all()


def test_structural_alerts_flags_known_pains_and_brenk_hit():
    alerts = structural_alerts(["O=C(Nc1ccccc1)c1ccc(N=Nc2ccc(O)cc2)cc1", "CCO", "not_a_smiles"])
    assert alerts.loc[0, "pains_alerts"] >= 1
    assert alerts.loc[0, "brenk_alerts"] >= 1
    assert alerts.loc[1, "pains_alerts"] == 0
    assert alerts.loc[1, "brenk_alerts"] == 0
    assert alerts.loc[2].isna().all()


def test_sa_scores_valid_and_invalid():
    scores = sa_scores(["CCO", "not_a_smiles"])
    assert scores[0] > 0
    assert np.isnan(scores[1])


def test_implausibility_battery_counts_and_threshold():
    panel = pd.DataFrame(
        {
            "molecular_weight": [100.0, 800.0],
            "clogp": [1.0, 9.0],
            "tpsa": [50.0, 50.0],
            "rotatable_bonds": [2, 2],
        }
    )
    alerts = pd.DataFrame({"pains_alerts": [0, 0], "brenk_alerts": [0, 0]})
    sa = np.array([2.0, 2.0])
    battery = {
        "mw_max": 700,
        "clogp_max": 7,
        "tpsa_max": 200,
        "rotatable_bonds_max": 15,
        "sa_score_max": 6.0,
        "pains_alerts_max": 0,
        "brenk_alerts_max": 0,
        "fail_threshold_count": 2,
    }
    result = implausibility_battery(panel, alerts, sa, battery)
    assert result.loc[0, "fail_count"] == 0
    assert not result.loc[0, "implausible"]
    assert result.loc[1, "fail_count"] == 2  # mw and clogp
    assert result.loc[1, "implausible"]


def test_implausibility_battery_nan_input_fails_every_check_it_touches():
    panel = pd.DataFrame({"molecular_weight": [np.nan], "clogp": [np.nan], "tpsa": [np.nan], "rotatable_bonds": [np.nan]})
    alerts = pd.DataFrame({"pains_alerts": [np.nan], "brenk_alerts": [np.nan]})
    battery = {
        "mw_max": 700,
        "clogp_max": 7,
        "tpsa_max": 200,
        "rotatable_bonds_max": 15,
        "sa_score_max": 6.0,
        "pains_alerts_max": 0,
        "brenk_alerts_max": 0,
        "fail_threshold_count": 2,
    }
    result = implausibility_battery(panel, alerts, np.array([np.nan]), battery)
    assert result.loc[0, "fail_count"] == 7  # all 7 checks unevaluable -> all fail
    assert result.loc[0, "implausible"]


def test_select_top_k_orders_by_fitness_descending():
    smiles = ["a", "b", "c", "d"]
    fitness = np.array([0.1, 0.9, 0.5, 0.3])
    top, scores = select_top_k(smiles, fitness, k=2)
    assert top == ["b", "c"]
    np.testing.assert_allclose(scores, [0.9, 0.5])


def test_select_top_k_caps_at_population_size():
    top, scores = select_top_k(["a", "b"], np.array([0.1, 0.2]), k=10)
    assert len(top) == 2


def test_objective_trajectory_ignores_invalid_rows():
    trajectory = pd.DataFrame(
        {
            "generation": [0, 0, 1, 1],
            "fitness": [0.2, 0.4, np.nan, 0.6],
        }
    )
    result = objective_trajectory(trajectory)
    gen0 = result[result["generation"] == 0].iloc[0]
    gen1 = result[result["generation"] == 1].iloc[0]
    assert gen0["mean"] == pytest.approx(0.3)
    assert gen0["count"] == 2
    assert gen1["mean"] == pytest.approx(0.6)
    assert gen1["count"] == 1


def test_component_breakdown_summary_stats():
    df = pd.DataFrame({"QED": [0.2, 0.4, 0.6], "hERG": [0.1, 0.1, 0.1]})
    result = component_breakdown(df, ["QED", "hERG"])
    qed_row = result[result["component"] == "QED"].iloc[0]
    assert qed_row["mean"] == pytest.approx(0.4)
    assert qed_row["min"] == pytest.approx(0.2)
    assert qed_row["max"] == pytest.approx(0.6)


def test_nearest_neighbor_similarity_uses_reference_index(tmp_path):
    csv_path = tmp_path / "reference.csv"
    csv_path.write_text("smiles\nCCO\nc1ccccc1\n")
    index = ReferenceFingerprintIndex.from_csv(csv_path)
    result = nearest_neighbor_similarity(["CCO"], index)
    assert result[0] == pytest.approx(1.0)


def test_independent_divergence_perfect_agreement_zero_disagreement():
    objective = pd.DataFrame({"hERG": [0.1, 0.5, 0.9]})
    independent = pd.DataFrame({"hERG_independent": [0.1, 0.5, 0.9]})
    result = independent_divergence(objective, independent, {"hERG": "hERG_independent"})
    row = result.iloc[0]
    assert row["spearman_r"] == pytest.approx(1.0)
    assert row["mean_abs_disagreement"] == pytest.approx(0.0)
    assert row["n"] == 3


def test_independent_divergence_drops_nan_rows():
    objective = pd.DataFrame({"hERG": [0.1, np.nan, 0.9]})
    independent = pd.DataFrame({"hERG_independent": [0.2, 0.5, np.nan]})
    result = independent_divergence(objective, independent, {"hERG": "hERG_independent"})
    row = result.iloc[0]
    assert row["n"] == 1
    assert row["mean_abs_disagreement"] == pytest.approx(0.1)


def test_divergence_headline_computes_delta():
    starting = pd.DataFrame({"property": ["hERG"], "mean_abs_disagreement": [0.1], "spearman_r": [0.9]})
    optimized = pd.DataFrame({"property": ["hERG"], "mean_abs_disagreement": [0.4], "spearman_r": [0.2]})
    result = divergence_headline(starting, optimized)
    row = result.iloc[0]
    assert row["mean_abs_disagreement_starting"] == pytest.approx(0.1)
    assert row["mean_abs_disagreement_optimized"] == pytest.approx(0.4)
    assert row["mean_abs_disagreement_delta"] == pytest.approx(0.3)

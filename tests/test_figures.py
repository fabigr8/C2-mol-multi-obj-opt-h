import pandas as pd
import pytest

from rhmoo.figures import (
    _representative_b_label,
    fig1_structure_grid,
    fig2_score_vs_nn_distance,
    fig3_objective_vs_independent,
    fig4_ad_tradeoff,
    fig5_trajectories,
    generate_all_figures,
)

pytestmark = pytest.mark.smoke


def _detail_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "arm": ["A", "A", "B_theta0.30", "B_theta0.40", "B_theta0.50", "D"],
            "smiles": ["CCO", "c1ccccc1", "CCN", "CC(=O)O", "CCCC", "CNC"],
            "objective_score": [0.9, 0.5, 0.8, 0.7, 0.6, 0.3],
            "ad_similarity_vs_reference": [0.9, 0.2, 0.5, 0.4, 0.3, 0.6],
            "hERG": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "hERG_independent": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65],
            "Solubility_AqSolDB": [-2.0, -3.0, -4.0, -1.0, -2.5, -3.5],
            "Solubility_AqSolDB_independent": [-2.1, -2.9, -3.9, -1.1, -2.4, -3.6],
        }
    )


def test_representative_b_label_picks_middle_theta():
    detail = _detail_df()
    assert _representative_b_label(detail) == "B_theta0.40"


def test_representative_b_label_none_when_no_b_arm():
    detail = pd.DataFrame({"arm": ["A", "D"]})
    assert _representative_b_label(detail) is None


def test_fig1_structure_grid_creates_file(tmp_path):
    out = fig1_structure_grid(_detail_df(), tmp_path / "fig1.png", n=2)
    assert out.exists()
    assert out.stat().st_size > 0


def test_fig2_score_vs_nn_distance_creates_file(tmp_path):
    out = fig2_score_vs_nn_distance(_detail_df(), tmp_path / "fig2.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_fig3_objective_vs_independent_creates_file(tmp_path):
    starting_divergence = pd.DataFrame(
        {"property": ["hERG", "Solubility_AqSolDB"], "n": [10, 10], "spearman_r": [0.8, 0.7], "mean_abs_disagreement": [0.1, 0.2]}
    )
    out = fig3_objective_vs_independent(_detail_df(), starting_divergence, tmp_path / "fig3.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_fig4_ad_tradeoff_creates_file(tmp_path):
    summary_by_arm = pd.DataFrame(
        {
            "arm": ["B_theta0.30", "B_theta0.40", "B_theta0.50"],
            "theta": [0.3, 0.4, 0.5],
            "objective_score_mean_mean": [0.6, 0.65, 0.7],
            "objective_score_mean_std": [0.01, 0.02, 0.03],
            "implausibility_rate_mean": [0.2, 0.1, 0.05],
        }
    )
    out = fig4_ad_tradeoff(summary_by_arm, tmp_path / "fig4.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_fig5_trajectories_creates_file_and_ignores_arm_without_generation(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame({"generation": [0, 0, 1, 1], "fitness": [0.1, 0.2, 0.3, 0.4]}).to_parquet(raw_dir / "A_seed0.parquet")
    pd.DataFrame({"generation": [0, 0, 1, 1], "fitness": [0.15, 0.25, 0.35, 0.45]}).to_parquet(raw_dir / "A_seed1.parquet")
    pd.DataFrame({"objective_score": [0.1, 0.2]}).to_parquet(raw_dir / "D_seed0.parquet")

    out = fig5_trajectories(raw_dir, tmp_path / "fig5.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_generate_all_figures_creates_five_files(tmp_path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    _detail_df().to_csv(tables_dir / "topk_detail.csv", index=False)
    pd.DataFrame(
        {
            "arm": ["B_theta0.30", "B_theta0.40", "B_theta0.50"],
            "theta": [0.3, 0.4, 0.5],
            "objective_score_mean_mean": [0.6, 0.65, 0.7],
            "objective_score_mean_std": [0.01, 0.02, 0.03],
            "implausibility_rate_mean": [0.2, 0.1, 0.05],
        }
    ).to_csv(tables_dir / "summary_by_arm.csv", index=False)
    pd.DataFrame(
        {"property": ["hERG", "Solubility_AqSolDB"], "n": [10, 10], "spearman_r": [0.8, 0.7], "mean_abs_disagreement": [0.1, 0.2]}
    ).to_csv(tables_dir / "starting_population_divergence.csv", index=False)
    pd.DataFrame({"generation": [0, 1], "fitness": [0.1, 0.2]}).to_parquet(raw_dir / "A_seed0.parquet")

    paths = generate_all_figures(tmp_path)
    assert len(paths) == 5
    for path in paths:
        assert path.exists()

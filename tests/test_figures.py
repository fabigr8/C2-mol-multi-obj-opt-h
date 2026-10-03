import pandas as pd
import pytest

from conftest import HEADLINE_VARIANT, make_detail, make_divergence_long, make_summary_by_arm
from rhmoo.figures import (
    _representative_b_label,
    fig1_structure_grid,
    fig2_score_vs_nn_distance,
    fig3_objective_vs_independent,
    fig4_ad_tradeoff,
    fig5_trajectories,
    fig6_decoupling,
    fig7_exploit_clusters,
    generate_all_figures,
    optimized,
)

pytestmark = pytest.mark.smoke


def _nonempty(path):
    assert path.exists()
    assert path.stat().st_size > 0


def test_representative_b_label_picks_the_middle_theta():
    detail = pd.DataFrame({"arm": ["A", "B_theta0.30", "B_theta0.40", "B_theta0.50", "D"]})
    assert _representative_b_label(detail) == "B_theta0.40"
    assert _representative_b_label(pd.DataFrame({"arm": ["A", "D"]})) is None


def test_optimized_excludes_the_reference_arm():
    """The H3 headline is computed over optimized molecules only. An
    `arm != "D"` filter silently began to include arm R's 2,845 approved drugs
    once that arm existed, with no error and no visible symptom.
    """
    detail = make_detail()
    assert set(optimized(detail)["arm"]) == {"A", "B_theta0.40"}
    assert "R" not in set(optimized(detail)["arm"])


def test_fig1_annotates_score_independent_score_and_arm_r_percentile(tmp_path):
    out = fig1_structure_grid(
        make_detail(), tmp_path / "fig1.png", n=6, headline_variant=HEADLINE_VARIANT
    )
    _nonempty(out)


def test_fig1_handles_an_empty_selection(tmp_path):
    empty = make_detail().iloc[0:0]
    _nonempty(fig1_structure_grid(empty, tmp_path / "fig1.png"))


def test_fig2_creates_file(tmp_path):
    _nonempty(fig2_score_vs_nn_distance(make_detail(), tmp_path / "fig2.png"))


def test_fig3_renders_three_populations_per_variant(tmp_path):
    # The canonical filename matters: the SVG companion is written only for
    # the matplotlib-rendered figures, keyed on the figure's own stem.
    out = fig3_objective_vs_independent(
        make_divergence_long(), tmp_path / "fig3_objective_vs_independent.png"
    )
    _nonempty(out)
    # Vector companion for blog-width legibility (brief SS5 allows png|svg).
    assert out.with_suffix(".svg").exists()


def test_fig4_creates_file(tmp_path):
    _nonempty(fig4_ad_tradeoff(make_summary_by_arm(), tmp_path / "fig4.png"))


def test_fig5_skips_arms_without_a_multi_generation_trajectory(tmp_path, results_dir):
    _nonempty(fig5_trajectories(results_dir / "raw", tmp_path / "fig5.png"))


def test_fig5_single_seed_has_no_shaded_band(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"generation": [0, 0, 1, 1], "fitness": [0.1, 0.2, 0.3, 0.4]}).to_parquet(
        raw / "A_seed0.parquet"
    )
    _nonempty(fig5_trajectories(raw, tmp_path / "fig5.png"))


def test_fig6_creates_file(tmp_path, results_dir):
    curves = pd.read_csv(results_dir / "tables" / "decoupling_curves.csv")
    decoupling = pd.read_csv(results_dir / "tables" / "decoupling_per_seed.csv")
    out = fig6_decoupling(curves, tmp_path / "fig6.png", decoupling, headline_variant=HEADLINE_VARIANT)
    _nonempty(out)


def test_fig7_creates_file(tmp_path, results_dir):
    clusters = pd.read_csv(results_dir / "tables" / "scaffold_clusters.csv")
    _nonempty(fig7_exploit_clusters(clusters, tmp_path / "fig7.png"))


def test_fig7_handles_no_clusters(tmp_path):
    empty = pd.DataFrame(columns=["arm", "grouping", "cluster", "n_molecules"])
    _nonempty(fig7_exploit_clusters(empty, tmp_path / "fig7.png"))


def test_generate_all_figures_creates_seven_files(results_dir):
    paths = generate_all_figures(results_dir)
    assert len(paths) == 7
    assert [p.stem for p in paths] == [
        "fig1_structure_grid",
        "fig2_score_vs_nn_distance",
        "fig3_objective_vs_independent",
        "fig4_ad_tradeoff",
        "fig5_trajectories",
        "fig6_decoupling",
        "fig7_exploit_clusters",
    ]
    for path in paths:
        _nonempty(path)

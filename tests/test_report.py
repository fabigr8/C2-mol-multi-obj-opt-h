import pandas as pd
import pytest

from rhmoo.report import generate_results_md

pytestmark = pytest.mark.smoke


def _write_tables(tables_dir):
    tables_dir.mkdir(parents=True)
    pd.DataFrame(
        {"arm": ["A", "D"], "theta": [None, None], "objective_score_mean_mean": [0.6, 0.3]}
    ).to_csv(tables_dir / "summary_by_arm.csv", index=False)
    pd.DataFrame(
        {"property": ["hERG", "Solubility_AqSolDB"], "n": [10, 10], "spearman_r": [0.8, 0.7], "mean_abs_disagreement": [0.1, 0.2]}
    ).to_csv(tables_dir / "starting_population_divergence.csv", index=False)
    pd.DataFrame(
        {
            "arm": ["A", "D"],
            "hERG": [0.1, 0.2],
            "hERG_independent": [0.15, 0.25],
            "Solubility_AqSolDB": [-2.0, -3.0],
            "Solubility_AqSolDB_independent": [-2.1, -2.9],
        }
    ).to_csv(tables_dir / "topk_detail.csv", index=False)


def test_generate_results_md_has_no_figures_when_none_exist(tmp_path):
    _write_tables(tmp_path / "tables")

    out_path = generate_results_md(tmp_path)

    content = out_path.read_text()
    assert content.startswith("# RESULTS")
    assert "## Per-arm summary" in content
    assert "starting population" in content
    assert "optimized top-k" in content
    assert "fig1_structure_grid" not in content


def test_generate_results_md_embeds_existing_figures(tmp_path):
    _write_tables(tmp_path / "tables")
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    (figures_dir / "fig1_structure_grid.png").write_bytes(b"fake-png")

    out_path = generate_results_md(tmp_path)

    content = out_path.read_text()
    assert "![fig1_structure_grid](figures/fig1_structure_grid.png)" in content


def test_generate_results_md_has_no_prose_only_headers_and_tables(tmp_path):
    _write_tables(tmp_path / "tables")

    content = generate_results_md(tmp_path).read_text()

    for line in content.splitlines():
        assert line == "" or line.startswith("#") or line.startswith("|") or line.startswith("!["), line

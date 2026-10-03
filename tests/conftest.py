"""Shared synthetic fixtures for the analysis-layer tests.

Pure pandas frames matching the schema `rhmoo.runner.run_experiment` writes, so
the metrics / figure / report tests stay in the smoke budget: they exercise no
ADMET-AI inference, no GA and no LightGBM.
"""

from __future__ import annotations

import pandas as pd
import pytest

VARIANTS = ("model_control", "data_control_half", "data_control_test")
HEADLINE_VARIANT = "data_control_half"

SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCCCCCCCCCCCCCCC",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "c1ccc2ccccc2c1",
]


def make_detail() -> pd.DataFrame:
    """Per-molecule top-k detail, two optimized arms plus a sample and a
    reference arm -- enough to catch an arm filter that selects on name.
    """
    rows = []
    specs = [
        ("A", "optimize", 0.4),
        ("B_theta0.40", "optimize", 0.4),
        ("D", "sample", None),
        ("R", "reference", None),
    ]
    for arm, kind, theta in specs:
        for i, smiles in enumerate(SMILES):
            rows.append(
                {
                    "arm": arm,
                    "arm_kind": kind,
                    "theta": theta,
                    "seed": None if kind == "reference" else 0,
                    "smiles": smiles,
                    "objective_score": 0.9 - 0.05 * i if kind == "optimize" else 0.4 - 0.03 * i,
                    "fitness": 0.9 - 0.05 * i,
                    "ad_similarity_vs_reference": 0.2 + 0.1 * i,
                    "ad_similarity_vs_starting": 0.3 + 0.05 * i,
                    "arm_r_percentile": 0.95 - 0.05 * i,
                    "sa_score": 2.0 + 0.2 * i,
                    "implausible": i >= 4,
                    "murcko_scaffold": "<acyclic>" if i in (0, 3) else "c1ccccc1",
                    "dominant_component": "hERG" if i % 2 == 0 else "QED",
                    **{f"hERG_independent__{v}": 0.1 + 0.05 * i for v in VARIANTS},
                    **{f"Solubility_AqSolDB_independent__{v}": -2.0 + 0.3 * i for v in VARIANTS},
                    "hERG": 0.2 + 0.04 * i,
                    "Solubility_AqSolDB": -1.5 + 0.2 * i,
                }
            )
    return pd.DataFrame(rows)


def make_divergence_long() -> pd.DataFrame:
    rows = []
    for population, arm, kind in (
        ("starting", "starting_population", "baseline"),
        ("arm_r", "R", "reference"),
        ("optimized", "A", "optimize"),
    ):
        for variant in VARIANTS:
            for prop, base in (("hERG", 0.2), ("Solubility_AqSolDB", 0.6)):
                rows.append(
                    {
                        "arm": arm,
                        "arm_kind": kind,
                        "theta": None,
                        "seed": None,
                        "population": population,
                        "variant": variant,
                        "is_headline_variant": variant == HEADLINE_VARIANT,
                        "property": prop,
                        "n": 25,
                        "spearman_r": 0.5,
                        "mean_abs_disagreement": base + (0.3 if population == "optimized" else 0.0),
                    }
                )
    return pd.DataFrame(rows)


def make_summary_by_arm() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"arm": "A", "arm_kind": "optimize", "theta": None, "n_seeds_mean": 2,
             "objective_score_mean_mean": 0.80, "objective_score_mean_std": 0.02,
             "implausibility_rate_mean": 0.70},
            {"arm": "B_theta0.30", "arm_kind": "optimize", "theta": 0.3, "n_seeds_mean": 2,
             "objective_score_mean_mean": 0.75, "objective_score_mean_std": 0.03,
             "implausibility_rate_mean": 0.50},
            {"arm": "B_theta0.40", "arm_kind": "optimize", "theta": 0.4, "n_seeds_mean": 2,
             "objective_score_mean_mean": 0.70, "objective_score_mean_std": 0.03,
             "implausibility_rate_mean": 0.35},
            {"arm": "B_theta0.50", "arm_kind": "optimize", "theta": 0.5, "n_seeds_mean": 2,
             "objective_score_mean_mean": 0.64, "objective_score_mean_std": 0.04,
             "implausibility_rate_mean": 0.20},
            {"arm": "R", "arm_kind": "reference", "theta": None, "n_seeds_mean": 1,
             "objective_score_mean_mean": 0.42, "objective_score_mean_std": 0.0,
             "implausibility_rate_mean": 0.04},
        ]
    )


def make_decoupling_curves() -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        for seed in (0, 1):
            for generation in range(8):
                rows.append(
                    {
                        "arm": "A",
                        "seed": seed,
                        "generation": generation,
                        "objective_restricted": 0.4 + 0.05 * generation,
                        "independent_composite": 0.4 + 0.005 * generation,
                        "gap": 0.045 * generation,
                        "gap_delta": 0.045 * generation,
                        "variant": variant,
                    }
                )
    return pd.DataFrame(rows)


def make_decoupling_per_seed() -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        for seed in (0, 1):
            for margin in (0.05, 0.10, 0.15):
                rows.append(
                    {
                        "arm": "A",
                        "seed": seed,
                        "theta": None,
                        "variant": variant,
                        "margin": margin,
                        "persistence": 3,
                        "is_primary_margin": margin == 0.10,
                        "decoupling_generation": 3,
                        "decoupled": True,
                        "final_gap_delta": 0.3,
                        "max_gap_delta": 0.3,
                    }
                )
    return pd.DataFrame(rows)


def make_scaffold_clusters() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"arm": "A", "grouping": "murcko_scaffold", "cluster": "<acyclic>", "n_molecules": 4,
             "fraction": 0.67, "objective_score_mean": 0.85, "objective_score_max": 0.90,
             "representative_smiles": "CCCCCCCCCCCCCCCC", "representative_objective_score": 0.90},
            {"arm": "A", "grouping": "murcko_scaffold", "cluster": "c1ccccc1", "n_molecules": 2,
             "fraction": 0.33, "objective_score_mean": 0.70, "objective_score_max": 0.75,
             "representative_smiles": "c1ccccc1", "representative_objective_score": 0.75},
            {"arm": "A", "grouping": "dominant_component", "cluster": "hERG", "n_molecules": 3,
             "fraction": 0.5, "objective_score_mean": 0.80, "objective_score_max": 0.90,
             "representative_smiles": "CCO", "representative_objective_score": 0.90},
        ]
    )


@pytest.fixture
def results_dir(tmp_path):
    """A complete results directory as the runner would write it."""
    tables = tmp_path / "tables"
    tables.mkdir()
    make_detail().to_csv(tables / "topk_detail.csv", index=False)
    make_divergence_long().to_csv(tables / "divergence_long.csv", index=False)
    make_summary_by_arm().to_csv(tables / "summary_by_arm.csv", index=False)
    make_decoupling_curves().to_csv(tables / "decoupling_curves.csv", index=False)
    make_decoupling_per_seed().to_csv(tables / "decoupling_per_seed.csv", index=False)
    make_scaffold_clusters().to_csv(tables / "scaffold_clusters.csv", index=False)

    raw = tmp_path / "raw"
    raw.mkdir()
    for seed in (0, 1):
        pd.DataFrame(
            {
                "generation": [0, 0, 1, 1, 2, 2],
                "fitness": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "smiles": SMILES,
            }
        ).to_parquet(raw / f"A_seed{seed}.parquet")
        pd.DataFrame({"generation": [0] * 6, "rank": range(6), "smiles": SMILES,
                      "fitness": [0.6] * 6}).to_parquet(raw / f"A_seed{seed}_elites.parquet")
    # A sample arm has a single generation and must not be plotted as a trajectory.
    pd.DataFrame({"generation": [0, 0], "fitness": [0.1, 0.2], "smiles": SMILES[:2]}).to_parquet(
        raw / "D_seed0.parquet"
    )
    return tmp_path

"""Figure generation (brief SS5, PLAN Phase 10): reads the tables/raw
artifacts `rhmoo.runner.run_experiment` writes and renders the five
brief-mandated figures. Purely descriptive -- no interpretive text.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Draw  # noqa: E402

from rhmoo.metrics import divergence_headline, independent_divergence, objective_trajectory  # noqa: E402

INDEPENDENT_PAIRS = {"hERG": "hERG_independent", "Solubility_AqSolDB": "Solubility_AqSolDB_independent"}


def _representative_b_label(detail: pd.DataFrame) -> str | None:
    """Same fixed choice as arm C's AD theta (PLAN Phase 9): the middle of the
    swept theta values, for the single arm-A-vs-arm-B comparison figure 1.
    """
    labels = sorted(a for a in detail["arm"].unique() if a.startswith("B_theta"))
    return labels[len(labels) // 2] if labels else None


def fig1_structure_grid(detail: pd.DataFrame, out_path: str | Path, n: int = 20) -> Path:
    """Top-n by objective score, unmodified, arm A vs the representative arm B
    (brief SS5/SS8: not a hand-picked selection).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    b_label = _representative_b_label(detail)
    labels = [label for label in ("A", b_label) if label is not None]

    mols, legends = [], []
    for label in labels:
        subset = detail[detail["arm"] == label].sort_values("objective_score", ascending=False).head(n)
        for row in subset.itertuples():
            mol = Chem.MolFromSmiles(row.smiles)
            if mol is None:
                continue
            mols.append(mol)
            indep = getattr(row, "hERG_independent", float("nan"))
            legends.append(f"{label} obj={row.objective_score:.2f} hERG_indep={indep:.2f}")

    image = Draw.MolsToGridImage(mols, molsPerRow=n, legends=legends, subImgSize=(220, 220))
    image.save(str(out_path))
    return out_path


def fig2_score_vs_nn_distance(detail: pd.DataFrame, out_path: str | Path) -> Path:
    """Scatter of objective score vs nearest-neighbour distance to the
    predictor reference set, colored by arm (brief SS5, H2).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    for arm, group in detail.groupby("arm"):
        distance = 1.0 - group["ad_similarity_vs_reference"]
        ax.scatter(distance, group["objective_score"], label=arm, alpha=0.6, s=15)
    ax.set_xlabel("1 - max ECFP4 Tanimoto to predictor reference set")
    ax.set_ylabel("objective score")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig3_objective_vs_independent(
    detail: pd.DataFrame, starting_divergence: pd.DataFrame, out_path: str | Path
) -> Path:
    """H3: objective-vs-independent-scorer disagreement, starting population
    vs optimized top-k (pooled over all non-control arms) -- the headline
    number of the whole project (brief SS4).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    optimized_rows = detail[detail["arm"] != "D"]
    optimized_divergence = independent_divergence(optimized_rows, optimized_rows, INDEPENDENT_PAIRS)
    headline = divergence_headline(starting_divergence, optimized_divergence)

    properties = headline["property"].tolist()
    x = np.arange(len(properties))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(x - width / 2, headline["mean_abs_disagreement_starting"], width, label="starting population")
    ax.bar(x + width / 2, headline["mean_abs_disagreement_optimized"], width, label="optimized top-k")
    ax.set_xticks(x)
    ax.set_xticklabels(properties)
    ax.set_ylabel("mean absolute disagreement (objective vs independent scorer)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig4_ad_tradeoff(summary_by_arm: pd.DataFrame, out_path: str | Path) -> Path:
    """Achieved objective score vs AD theta, with implausibility rate (brief
    SS5, H4 trade-off curve).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    b_rows = summary_by_arm[summary_by_arm["arm"].str.startswith("B_theta")].sort_values("theta")

    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax1.errorbar(
        b_rows["theta"],
        b_rows["objective_score_mean_mean"],
        yerr=b_rows["objective_score_mean_std"],
        marker="o",
        color="tab:blue",
        label="objective score",
    )
    ax1.set_xlabel("applicability-domain theta")
    ax1.set_ylabel("objective score (mean +/- sd across seeds)", color="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(b_rows["theta"], b_rows["implausibility_rate_mean"], marker="s", color="tab:red", label="implausibility rate")
    ax2.set_ylabel("implausibility rate", color="tab:red")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def fig5_trajectories(raw_dir: str | Path, out_path: str | Path) -> Path:
    """Objective score per generation, seed spread shaded, one line per
    non-control arm (brief SS5).
    """
    raw_dir = Path(raw_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = sorted({p.stem.rsplit("_seed", 1)[0] for p in raw_dir.glob("*.parquet")} - {"D"})

    fig, ax = plt.subplots(figsize=(7, 5))
    for label in labels:
        seed_curves = []
        for path in sorted(raw_dir.glob(f"{label}_seed*.parquet")):
            trajectory = pd.read_parquet(path)
            if "generation" not in trajectory.columns:
                continue
            per_generation = objective_trajectory(trajectory)
            seed_curves.append(per_generation.set_index("generation")["mean"])
        if not seed_curves:
            continue
        combined = pd.concat(seed_curves, axis=1)
        mean_curve = combined.mean(axis=1)
        ax.plot(mean_curve.index, mean_curve.to_numpy(), label=label)
        if combined.shape[1] > 1:
            ax.fill_between(mean_curve.index, combined.min(axis=1).to_numpy(), combined.max(axis=1).to_numpy(), alpha=0.2)

    ax.set_xlabel("generation")
    ax.set_ylabel("mean fitness across the population")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_all_figures(results_dir: str | Path) -> list[Path]:
    results_dir = Path(results_dir)
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"

    detail = pd.read_csv(tables_dir / "topk_detail.csv")
    summary_by_arm = pd.read_csv(tables_dir / "summary_by_arm.csv")
    starting_divergence = pd.read_csv(tables_dir / "starting_population_divergence.csv")

    return [
        fig1_structure_grid(detail, figures_dir / "fig1_structure_grid.png"),
        fig2_score_vs_nn_distance(detail, figures_dir / "fig2_score_vs_nn_distance.png"),
        fig3_objective_vs_independent(detail, starting_divergence, figures_dir / "fig3_objective_vs_independent.png"),
        fig4_ad_tradeoff(summary_by_arm, figures_dir / "fig4_ad_tradeoff.png"),
        fig5_trajectories(results_dir / "raw", figures_dir / "fig5_trajectories.png"),
    ]

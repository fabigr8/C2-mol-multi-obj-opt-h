"""Figure generation (brief SS5): reads the tables/raw artifacts
`rhmoo.runner.run_experiment` writes and renders the seven brief-mandated
figures. Purely descriptive -- no interpretive text.

Arms are selected by `arm_kind`, never by excluding the control's name. An
`arm != "D"` filter silently came to mean "optimized or reference" once arm R
entered the tables, which would have pooled 2,845 approved drugs into the H3
headline number with no error and no visible symptom.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Draw  # noqa: E402

from rhmoo.independent_scorer import INDEPENDENT_ENDPOINTS, independent_column  # noqa: E402
from rhmoo.metrics import objective_trajectory  # noqa: E402

#: Populations in reporting order (brief SS4.2 wants all three side by side).
POPULATION_ORDER = ("starting", "arm_r", "optimized")

#: Figures that are matplotlib-rendered and so can also be written as SVG.
_VECTOR_FIGURES = (
    "fig2_score_vs_nn_distance",
    "fig3_objective_vs_independent",
    "fig4_ad_tradeoff",
    "fig5_trajectories",
    "fig6_decoupling",
)


def _save(fig, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if out_path.stem in _VECTOR_FIGURES:
        fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    return out_path


def optimized(detail: pd.DataFrame) -> pd.DataFrame:
    return detail[detail["arm_kind"] == "optimize"]


def _representative_b_label(detail: pd.DataFrame) -> str | None:
    """Same fixed choice as the AD+synth arm's theta: the middle of the swept
    theta values, for the single arm-A-vs-arm-B comparison in figure 1.
    """
    labels = sorted(a for a in detail["arm"].unique() if "_theta" in str(a))
    return labels[len(labels) // 2] if labels else None


def _headline_variant(detail_columns, variants: list[str] | None = None) -> str:
    if variants:
        return variants[0]
    for column in detail_columns:
        if column.startswith("hERG_independent__"):
            return column.split("__", 1)[1]
    return "model_control"


def fig1_structure_grid(
    detail: pd.DataFrame, out_path: str | Path, n: int = 20, headline_variant: str | None = None
) -> Path:
    """Top-n by objective score, unmodified, arm A vs the representative arm B
    (brief SS5/SS9: the top-20 by score, not a hand-picked selection).

    Laid out 5 per row rather than n per row: at n=20 a single row is ~4,400px
    wide and illegible at blog width, which brief SS5 explicitly requires.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arms = optimized(detail)
    variant = headline_variant or _headline_variant(detail.columns)
    herg_column = independent_column("hERG", variant)

    b_label = _representative_b_label(arms)
    labels = [label for label in ("A", b_label) if label is not None and label in set(arms["arm"])]

    panels = []
    for label in labels:
        subset = arms[arms["arm"] == label].sort_values("objective_score", ascending=False).head(n)
        mols, legends = [], []
        for row in subset.itertuples():
            mol = Chem.MolFromSmiles(row.smiles) if isinstance(row.smiles, str) else None
            if mol is None:
                continue
            mols.append(mol)
            independent = getattr(row, herg_column, float("nan"))
            percentile = getattr(row, "arm_r_percentile", float("nan"))
            legends.append(
                f"obj={row.objective_score:.2f}  hERG_ind={independent:.2f}  armR_pct={percentile:.2f}"
            )
        if mols:
            panels.append((label, mols, legends))

    if not panels:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        return _save(fig, out_path)

    # Height follows the real row count, so a short panel is not padded out
    # with whitespace and a full 20-molecule panel is not squashed.
    rows_per_panel = [max(1, -(-len(mols) // 5)) for _, mols, _ in panels]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(10, 2.4 * sum(rows_per_panel) + 0.8 * len(panels)), squeeze=False
    )
    for ax, (label, mols, legends) in zip(axes.ravel(), panels):
        image = Draw.MolsToGridImage(
            mols, molsPerRow=5, legends=legends, subImgSize=(300, 300), returnPNG=True
        )
        png = image.data if hasattr(image, "data") else image
        ax.imshow(plt.imread(io.BytesIO(png)))
        ax.set_title(f"arm {label} -- top {len(mols)} by objective score", fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, out_path)


def fig2_score_vs_nn_distance(detail: pd.DataFrame, out_path: str | Path) -> Path:
    """Objective score vs nearest-neighbour distance to the predictor reference
    set, colored by arm (brief SS5, H2).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for arm, group in detail.groupby("arm"):
        ax.scatter(
            1.0 - group["ad_similarity_vs_reference"],
            group["objective_score"],
            label=arm,
            alpha=0.6,
            s=15,
        )
    ax.set_xlabel("1 - max ECFP4 Tanimoto to predictor reference set")
    ax.set_ylabel("objective score")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return _save(fig, Path(out_path))


def fig3_objective_vs_independent(divergence_long: pd.DataFrame, out_path: str | Path) -> Path:
    """H3: objective-vs-independent disagreement for the starting population,
    arm R and the optimized top-k, one panel per scorer variant (brief SS4.2).

    The difference between the starting population and the optimized top-k is
    the headline number of the project; the three variants are shown separately
    and never averaged.
    """
    variants = [v for v in divergence_long["variant"].dropna().unique()]
    properties = [p for p in INDEPENDENT_ENDPOINTS if p in set(divergence_long["property"])]

    fig, axes = plt.subplots(1, max(len(variants), 1), figsize=(5 * max(len(variants), 1), 5), squeeze=False)
    for ax, variant in zip(axes.ravel(), variants):
        subset = divergence_long[divergence_long["variant"] == variant]
        x = np.arange(len(properties))
        populations = [p for p in POPULATION_ORDER if p in set(subset["population"])]
        width = 0.8 / max(len(populations), 1)
        for i, population in enumerate(populations):
            values = [
                subset[(subset["population"] == population) & (subset["property"] == prop)][
                    "mean_abs_disagreement"
                ].mean()
                for prop in properties
            ]
            ax.bar(x + i * width - 0.4 + width / 2, values, width, label=population)
        ax.set_xticks(x)
        ax.set_xticklabels(properties, fontsize=8, rotation=15)
        ax.set_title(variant, fontsize=10)
        ax.set_ylabel("mean absolute disagreement")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, Path(out_path))


def fig4_ad_tradeoff(summary_by_arm: pd.DataFrame, out_path: str | Path) -> Path:
    """Achieved objective score vs AD theta, with implausibility rate (brief
    SS5, H4 trade-off curve).
    """
    rows = summary_by_arm[summary_by_arm["arm"].astype(str).str.contains("_theta")].sort_values("theta")

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.errorbar(
        rows["theta"],
        rows["objective_score_mean_mean"],
        yerr=rows["objective_score_mean_std"],
        marker="o",
        color="tab:blue",
        label="objective score",
    )
    ax1.set_xlabel("applicability-domain theta")
    ax1.set_ylabel("objective score (mean +/- sd across seeds)", color="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(
        rows["theta"],
        rows["implausibility_rate_mean"],
        marker="s",
        color="tab:red",
        label="implausibility rate",
    )
    ax2.set_ylabel("implausibility rate", color="tab:red")
    fig.tight_layout()
    return _save(fig, Path(out_path))


def fig5_trajectories(raw_dir: str | Path, out_path: str | Path) -> Path:
    """Objective score per generation, seed spread shaded, one line per
    optimized arm (brief SS5).
    """
    raw_dir = Path(raw_dir)
    candidates = {
        p.stem.rsplit("_seed", 1)[0]
        for p in raw_dir.glob("*.parquet")
        if not p.stem.endswith("_elites")
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    for label in sorted(candidates):
        seed_curves = []
        for path in sorted(raw_dir.glob(f"{label}_seed*.parquet")):
            if path.stem.endswith("_elites"):
                continue
            trajectory = pd.read_parquet(path)
            # Only an optimized arm has a multi-generation trajectory; the
            # sample and reference arms are a single scored set.
            if "generation" not in trajectory.columns or trajectory["generation"].nunique() < 2:
                continue
            seed_curves.append(objective_trajectory(trajectory).set_index("generation")["mean"])
        if not seed_curves:
            continue
        combined = pd.concat(seed_curves, axis=1)
        mean_curve = combined.mean(axis=1)
        ax.plot(mean_curve.index, mean_curve.to_numpy(), label=label)
        if combined.shape[1] > 1:
            ax.fill_between(
                mean_curve.index,
                combined.min(axis=1).to_numpy(),
                combined.max(axis=1).to_numpy(),
                alpha=0.2,
            )

    ax.set_xlabel("generation")
    ax.set_ylabel("mean fitness across the population")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return _save(fig, Path(out_path))


def fig6_decoupling(
    curves: pd.DataFrame,
    out_path: str | Path,
    decoupling: pd.DataFrame | None = None,
    headline_variant: str | None = None,
) -> Path:
    """H3b (brief SS4.3): objective score and independent-scorer estimate on one
    axis, per generation, for each optimized arm.

    Both series are on the same [0, 1] scale because both are pushed through the
    same normalization and aggregation, restricted to the endpoints an
    independent scorer covers (see rhmoo.decoupling).
    """
    curves = curves.copy()
    variant = headline_variant or (
        curves["variant"].iloc[0] if "variant" in curves and len(curves) else "model_control"
    )
    subset = curves[curves["variant"] == variant]
    arms = sorted(subset["arm"].unique())

    fig, axes = plt.subplots(1, max(len(arms), 1), figsize=(5 * max(len(arms), 1), 4.5), squeeze=False, sharey=True)
    for ax, arm in zip(axes.ravel(), arms):
        arm_rows = subset[subset["arm"] == arm]
        for column, label, color in (
            ("objective_restricted", "objective (restricted)", "tab:blue"),
            ("independent_composite", f"independent ({variant})", "tab:orange"),
        ):
            per_seed = arm_rows.pivot_table(index="generation", columns="seed", values=column)
            mean_curve = per_seed.mean(axis=1)
            ax.plot(mean_curve.index, mean_curve.to_numpy(), label=label, color=color)
            if per_seed.shape[1] > 1:
                ax.fill_between(
                    mean_curve.index,
                    per_seed.min(axis=1).to_numpy(),
                    per_seed.max(axis=1).to_numpy(),
                    alpha=0.2,
                    color=color,
                )

        if decoupling is not None and len(decoupling):
            primary = decoupling[
                (decoupling["arm"] == arm)
                & (decoupling["variant"] == variant)
                & decoupling["is_primary_margin"].astype(bool)
            ]
            generations = primary["decoupling_generation"].dropna()
            if len(generations):
                ax.axvline(float(generations.mean()), linestyle="--", color="grey")

        ax.set_title(arm, fontsize=10)
        ax.set_xlabel("generation")
    axes.ravel()[0].set_ylabel("composite score (covered endpoints)")
    axes.ravel()[0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    return _save(fig, Path(out_path))


def fig7_exploit_clusters(
    clusters: pd.DataFrame, out_path: str | Path, top_n: int = 8
) -> Path:
    """brief SS5.1: scaffold clusters of the optimized top-k, with counts and a
    representative structure per cluster.

    Turns a scalar implausibility rate into a structural claim about what the
    search converged on.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scaffolds = clusters[clusters["grouping"] == "murcko_scaffold"]
    arms = sorted(scaffolds["arm"].unique())
    if not arms:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        return _save(fig, out_path)

    arm = "A" if "A" in arms else arms[0]
    subset = scaffolds[scaffolds["arm"] == arm].head(top_n)

    fig, (ax_bar, ax_grid) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.4]})

    order = subset.iloc[::-1]
    ax_bar.barh(range(len(order)), order["n_molecules"], color="tab:blue")
    ax_bar.set_yticks(range(len(order)))
    ax_bar.set_yticklabels(
        [s if len(str(s)) <= 28 else str(s)[:25] + "..." for s in order["cluster"]], fontsize=7
    )
    ax_bar.set_xlabel(f"molecules in arm {arm} top-k")

    mols, legends = [], []
    for row in subset.itertuples():
        mol = Chem.MolFromSmiles(row.representative_smiles) if isinstance(row.representative_smiles, str) else None
        if mol is None:
            continue
        mols.append(mol)
        legends.append(f"n={row.n_molecules}  obj={row.representative_objective_score:.2f}")

    if mols:
        image = Draw.MolsToGridImage(
            mols, molsPerRow=4, legends=legends, subImgSize=(250, 250), returnPNG=True
        )
        png = image.data if hasattr(image, "data") else image
        ax_grid.imshow(plt.imread(io.BytesIO(png)))
    ax_grid.set_title(f"cluster representatives (highest objective score), arm {arm}", fontsize=10)
    ax_grid.axis("off")
    fig.tight_layout()
    return _save(fig, out_path)


def generate_all_figures(results_dir: str | Path, headline_variant: str | None = None) -> list[Path]:
    results_dir = Path(results_dir)
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"

    detail = pd.read_csv(tables_dir / "topk_detail.csv")
    summary_by_arm = pd.read_csv(tables_dir / "summary_by_arm.csv")
    divergence_long = pd.read_csv(tables_dir / "divergence_long.csv")

    if headline_variant is None:
        headline = divergence_long[divergence_long.get("is_headline_variant", False).astype(bool)]
        headline_variant = headline["variant"].iloc[0] if len(headline) else _headline_variant(detail.columns)

    paths = [
        fig1_structure_grid(
            detail, figures_dir / "fig1_structure_grid.png", headline_variant=headline_variant
        ),
        fig2_score_vs_nn_distance(detail, figures_dir / "fig2_score_vs_nn_distance.png"),
        fig3_objective_vs_independent(divergence_long, figures_dir / "fig3_objective_vs_independent.png"),
        fig4_ad_tradeoff(summary_by_arm, figures_dir / "fig4_ad_tradeoff.png"),
        fig5_trajectories(results_dir / "raw", figures_dir / "fig5_trajectories.png"),
    ]

    curves_path = tables_dir / "decoupling_curves.csv"
    decoupling_path = tables_dir / "decoupling_per_seed.csv"
    curves = pd.read_csv(curves_path) if curves_path.exists() else pd.DataFrame()
    decoupling = pd.read_csv(decoupling_path) if decoupling_path.exists() else None
    paths.append(
        fig6_decoupling(
            curves, figures_dir / "fig6_decoupling.png", decoupling, headline_variant=headline_variant
        )
        if len(curves)
        else _empty(figures_dir / "fig6_decoupling.png")
    )

    clusters_path = tables_dir / "scaffold_clusters.csv"
    clusters = pd.read_csv(clusters_path) if clusters_path.exists() else pd.DataFrame(columns=["grouping", "arm"])
    paths.append(fig7_exploit_clusters(clusters, figures_dir / "fig7_exploit_clusters.png"))
    return paths


def _empty(out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")
    return _save(fig, out_path)

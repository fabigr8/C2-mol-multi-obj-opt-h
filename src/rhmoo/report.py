"""RESULTS.md generation (brief SS5/SS8): tables and figure embeds only, no
interpretive prose ("RESULTS.md is a machine-generated artifact... numbers,
tables, figure embeds. The agent must not write conclusions, narrative, or
claims into it.").
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rhmoo.metrics import independent_divergence

INDEPENDENT_PAIRS = {"hERG": "hERG_independent", "Solubility_AqSolDB": "Solubility_AqSolDB_independent"}

FIGURE_NAMES = (
    "fig1_structure_grid",
    "fig2_score_vs_nn_distance",
    "fig3_objective_vs_independent",
    "fig4_ad_tradeoff",
    "fig5_trajectories",
)


def _markdown_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    for col in formatted.select_dtypes(include="number").columns:
        formatted[col] = formatted[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    header = "| " + " | ".join(str(c) for c in formatted.columns) + " |"
    separator = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in formatted.itertuples(index=False)]
    return "\n".join([header, separator, *rows])


def generate_results_md(results_dir: str | Path) -> Path:
    results_dir = Path(results_dir)
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"

    summary_by_arm = pd.read_csv(tables_dir / "summary_by_arm.csv")
    starting_divergence = pd.read_csv(tables_dir / "starting_population_divergence.csv")
    detail = pd.read_csv(tables_dir / "topk_detail.csv")

    optimized_rows = detail[detail["arm"] != "D"]
    optimized_divergence = independent_divergence(optimized_rows, optimized_rows, INDEPENDENT_PAIRS)

    lines = ["# RESULTS", ""]

    lines += ["## Per-arm summary (mean +/- sd across seeds)", "", _markdown_table(summary_by_arm), ""]
    lines += [
        "## H3: objective-vs-independent divergence -- starting population",
        "",
        _markdown_table(starting_divergence),
        "",
    ]
    lines += [
        "## H3: objective-vs-independent divergence -- optimized top-k",
        "",
        _markdown_table(optimized_divergence),
        "",
    ]

    for name in FIGURE_NAMES:
        for ext in ("png", "svg"):
            path = figures_dir / f"{name}.{ext}"
            if path.exists():
                lines += [f"## {name}", "", f"![{name}]({path.relative_to(results_dir).as_posix()})", ""]
                break

    out_path = results_dir / "RESULTS.md"
    out_path.write_text("\n".join(lines))
    return out_path

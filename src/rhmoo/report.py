"""RESULTS.md generation (brief SS5): tables and figure embeds only, no
interpretive prose ("RESULTS.md is a machine-generated artifact... numbers,
tables, figure embeds. The agent must not write conclusions, narrative, or
claims into it.").

`tests/test_report.py` enforces that every non-blank line is a heading, a table
row or an image embed, so units and qualifiers belong in column headers rather
than in sentences.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FIGURE_NAMES = (
    "fig1_structure_grid",
    "fig2_score_vs_nn_distance",
    "fig3_objective_vs_independent",
    "fig4_ad_tradeoff",
    "fig5_trajectories",
    "fig6_decoupling",
    "fig7_exploit_clusters",
)


def _markdown_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    for col in formatted.select_dtypes(include="number").columns:
        formatted[col] = formatted[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    header = "| " + " | ".join(str(c) for c in formatted.columns) + " |"
    separator = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in formatted.itertuples(index=False)]
    return "\n".join([header, separator, *rows])


def _section(lines: list[str], title: str, frame: pd.DataFrame | None) -> None:
    if frame is None or frame.empty:
        return
    lines += [f"## {title}", "", _markdown_table(frame), ""]


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def generate_results_md(results_dir: str | Path) -> Path:
    results_dir = Path(results_dir)
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"

    summary_by_arm = _read(tables_dir / "summary_by_arm.csv")
    divergence_long = _read(tables_dir / "divergence_long.csv")
    decoupling = _read(tables_dir / "decoupling_per_seed.csv")
    clusters = _read(tables_dir / "scaffold_clusters.csv")
    calibration = _read(tables_dir / "battery_calibration.csv")

    lines = ["# RESULTS", ""]

    _section(lines, "Per-arm summary (mean +/- sd across seeds)", summary_by_arm)

    if divergence_long is not None and not divergence_long.empty:
        columns = [
            c
            for c in ("population", "arm", "variant", "property", "n", "spearman_r", "mean_abs_disagreement")
            if c in divergence_long.columns
        ]
        headline = divergence_long[divergence_long["is_headline_variant"].astype(bool)]
        _section(
            lines,
            "H3: objective-vs-independent divergence -- headline scorer variant",
            headline[columns].sort_values(["population", "arm", "property"]),
        )
        _section(
            lines,
            "H3: objective-vs-independent divergence -- all scorer variants",
            divergence_long[columns].sort_values(["population", "arm", "variant", "property"]),
        )

    if decoupling is not None and not decoupling.empty:
        primary = decoupling[decoupling["is_primary_margin"].astype(bool)]
        _section(lines, "H3b: decoupling generation (primary pre-declared margin)", primary)
        _section(lines, "H3b: decoupling generation (declared margin sweep)", decoupling)

    _section(lines, "Exploit clusters (optimized arms)", clusters)
    _section(lines, "Implausibility battery calibration on arm R", calibration)

    for name in FIGURE_NAMES:
        for ext in ("png", "svg"):
            path = figures_dir / f"{name}.{ext}"
            if path.exists():
                lines += [f"## {name}", "", f"![{name}]({path.relative_to(results_dir).as_posix()})", ""]
                break

    out_path = results_dir / "RESULTS.md"
    out_path.write_text("\n".join(lines))
    return out_path

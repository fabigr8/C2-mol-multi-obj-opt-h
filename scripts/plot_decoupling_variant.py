"""Render fig6 for a non-headline scorer variant.

`generate_all_figures` draws fig6 for the pre-declared headline oracle only
(`data_control_half`), which is the right default: the headline figure should
show the pre-registered instrument. But the H3b result turns on the comparison
between variants -- only `data_control_test` is disjoint from ADMET-AI's
training data, and only it detects decoupling -- so the write-up needs the same
figure for the other variants. Same plotting code, different variant.

    uv run python scripts/plot_decoupling_variant.py data_control_test
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from rhmoo.figures import fig6_decoupling

ROOT = Path(__file__).resolve().parents[1]


def main(variant: str, results_dir: Path) -> Path:
    tables = results_dir / "tables"
    curves = pd.read_csv(tables / "decoupling_curves.csv")
    decoupling = pd.read_csv(tables / "decoupling_per_seed.csv")
    out_path = results_dir / "figures" / f"fig6b_decoupling_{variant}.png"
    return fig6_decoupling(curves, out_path, decoupling, headline_variant=variant)


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "data_control_test"
    results_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "main"
    print(main(variant, results_dir))

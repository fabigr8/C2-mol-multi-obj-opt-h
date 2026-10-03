"""Determinism verification (brief SS10): re-run one arm with the same seed and
assert an identical top-k.

Checks the objective scores as well as the SMILES, because a matching set of
molecules with different scores would still mean the pipeline is not
reproducible. Depends on the GA's stable argsort: unstable tie-breaking is
platform- and build-dependent, which is exactly what this check has to rule out
if the full run is to be reproducible on a different machine.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from rhmoo.config import load_config
from rhmoo.runner import run_experiment


def _topk(results_dir: Path, arm: str) -> pd.DataFrame:
    detail = pd.read_csv(results_dir / "tables" / "topk_detail.csv")
    subset = detail[detail["arm"] == arm]
    return subset.sort_values("smiles")[["smiles", "objective_score"]].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--arm", default="A")
    args = parser.parse_args()

    config = load_config(args.config)

    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        a = run_experiment(args.config, first)
        b = run_experiment(args.config, second)
        left, right = _topk(a, args.arm), _topk(b, args.arm)

    if left.empty:
        raise SystemExit(f"arm {args.arm!r} produced no reported molecules in {args.config}")

    pd.testing.assert_frame_equal(left, right)
    print(
        f"determinism OK: {args.config} arm {args.arm} reproduced an identical top-k "
        f"({len(left)} molecules, scores matched) across two independent runs"
    )


if __name__ == "__main__":
    main()

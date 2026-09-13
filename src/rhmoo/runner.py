"""Experiment entry point: loads config, sets up the output directory and run
log. The arms x seeds optimization loop (data layer, objective, GA, constraints,
metrics) is wired in here as those modules land in later phases.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rhmoo.config import config_hash, load_config
from rhmoo.run_log import run_log


def run_experiment(config_path: str | Path, results_root: str | Path | None = None) -> Path:
    config = load_config(config_path)
    out_dir = Path(results_root) if results_root else Path(config.results_dir)
    out_dir = out_dir / config.name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out_dir / "config.yaml")

    chash = config_hash(config)
    with run_log(out_dir / "run_log.jsonl", chash, config.name) as state:
        # TODO(phase 9): orchestrate arms x seeds and populate molecules_evaluated.
        state["molecules_evaluated"] = 0

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an rhmoo experiment from a config file.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root")
    args = parser.parse_args()
    out_dir = run_experiment(args.config, args.results_root)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

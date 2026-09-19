"""CLI: render all brief-mandated figures (PLAN Phase 10) from an experiment's
results directory (as written by `rhmoo.runner.run_experiment`).
"""

import argparse

from rhmoo.figures import generate_all_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures from an rhmoo results directory.")
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    paths = generate_all_figures(args.results_dir)
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

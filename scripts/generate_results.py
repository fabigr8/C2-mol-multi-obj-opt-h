"""CLI: generate RESULTS.md (PLAN Phase 10) from an experiment's results
directory (as written by `rhmoo.runner.run_experiment`).
"""

import argparse

from rhmoo.report import generate_results_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RESULTS.md from an rhmoo results directory.")
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    out_path = generate_results_md(args.results_dir)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

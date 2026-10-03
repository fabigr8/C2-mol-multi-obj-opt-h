"""Calibrates and freezes the implausibility battery against arm R (brief SS4.5).

Blocking: this must run, and its output must be committed, before any arm A
molecule is generated. The runner refuses to start unless the active config's
`implausibility_battery` block matches the frozen output written here.

Supersedes the Phase 8 "pre-registration freeze", which set these thresholds by
declaration rather than by calibration. Those values flagged 15.4% of approved
drugs -- three times brief SS4.5's ceiling -- driven almost entirely by
requiring zero Brenk alerts, which roughly half of all approved drugs violate.
See docs/PREREGISTRATION.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from rhmoo import calibration
from rhmoo.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    artifact = REPO_ROOT / config.reference_arm.artifact_path
    if not artifact.exists():
        raise SystemExit(f"arm R artifact not found at {artifact}; run `make arm-r` first")

    reference = pd.read_parquet(artifact)
    result = calibration.calibrate(reference)

    out_dir = artifact.parent
    criteria = calibration.per_criterion_table(reference, result)
    criteria.to_csv(out_dir / "battery_calibration.csv", index=False)

    histogram = calibration.fail_count_histogram(reference, result["thresholds"])
    histogram.to_csv(out_dir / "battery_fail_count_histogram.csv", index=False)

    controls = calibration.score_positive_controls(result["thresholds"])
    controls.to_csv(out_dir / "battery_positive_controls.csv", index=False)

    frozen = {
        "implausibility_battery": result["thresholds"],
        "alpha": result["alpha"],
        "target_flag_rate": result["target_flag_rate"],
        "joint_flag_rate_on_arm_r": result["joint_flag_rate"],
        "n_reference": result["n_reference"],
        "alpha_grid": list(calibration.ALPHA_GRID),
        "alpha_scan": result["scan"],
        "reference_artifact": str(config.reference_arm.artifact_path),
        "positive_controls_detectable_all_flagged": bool(
            controls[controls["group"] == "detectable"]["implausible"].all()
        ),
        "positive_controls_blind_spot_flagged": controls[controls["group"] == "blind_spot"][
            "implausible"
        ].tolist(),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "battery_frozen.json").write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")

    print(f"alpha* = {result['alpha']}  joint flag rate on arm R = {result['joint_flag_rate']:.4f} "
          f"(target <= {result['target_flag_rate']})")
    print(criteria.to_string(index=False))
    print()
    print(controls[["control", "group", "fail_count", "implausible"]].to_string(index=False))
    print()
    print("Paste into every config's implausibility_battery block:")
    for key, value in result["thresholds"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

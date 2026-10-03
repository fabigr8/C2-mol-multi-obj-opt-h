"""Calibration is tested on a synthetic reference panel rather than on the real
2,845-molecule arm R set: the full battery panel costs ~17s of RDKit, a third of
the smoke budget. The real calibration is exercised by scripts/calibrate_battery.py
and its committed output.
"""

import numpy as np
import pandas as pd
import pytest

from rhmoo import calibration

pytestmark = pytest.mark.smoke


def _reference(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "molecular_weight": rng.normal(350, 90, n),
            "clogp": rng.normal(2.5, 1.2, n),
            "tpsa": rng.normal(80, 30, n),
            "rotatable_bonds": rng.integers(0, 12, n).astype(float),
            "sa_score": rng.normal(3.0, 0.8, n),
            "pains_alerts": rng.binomial(1, 0.05, n).astype(float),
            "brenk_alerts": rng.binomial(3, 0.2, n).astype(float),
        }
    )


def test_joint_flag_rate_is_monotone_in_alpha():
    """The declared search assumes monotonicity -- a larger tail mass cannot
    flag fewer molecules -- which is what makes the chosen alpha unique.
    """
    reference = _reference()
    rates = [
        calibration.joint_flag_rate(reference, calibration.thresholds_for_alpha(reference, alpha))
        for alpha in sorted(calibration.ALPHA_GRID)
    ]
    assert rates == sorted(rates)


def test_calibrate_picks_the_largest_alpha_within_target():
    reference = _reference()
    result = calibration.calibrate(reference, target=0.05)

    assert result["joint_flag_rate"] <= 0.05
    # Most sensitive battery consistent with the constraint: every larger alpha
    # on the grid must exceed the target.
    larger = [s for s in result["scan"] if s["alpha"] > result["alpha"]]
    assert all(not s["within_target"] for s in larger)


def test_calibrate_raises_rather_than_silently_widening():
    """If no alpha on the declared grid meets the target, that is a decision for
    a human, not something to paper over.
    """
    reference = _reference()
    # A grid whose every alpha overshoots the target: the rule has no solution,
    # which is a decision for a human rather than something to widen silently.
    with pytest.raises(ValueError):
        calibration.calibrate(reference, target=0.001, alpha_grid=(0.10, 0.09))


def test_thresholds_are_rounded_up_onto_the_declared_grid():
    reference = _reference()
    thresholds = calibration.thresholds_for_alpha(reference, 0.05)

    assert thresholds["mw_max"] % 10 == pytest.approx(0.0)
    assert (thresholds["clogp_max"] * 2) % 1 == pytest.approx(0.0)
    assert float(thresholds["rotatable_bonds_max"]).is_integer()
    assert thresholds["fail_threshold_count"] == calibration.FAIL_THRESHOLD_COUNT
    # Counts are integers, so a fractional alert cutoff would be meaningless.
    assert isinstance(thresholds["pains_alerts_max"], int)
    assert isinstance(thresholds["brenk_alerts_max"], int)


def test_rounding_up_never_tightens_a_threshold():
    """Rounding up keeps the flag rate at or below alpha; rounding down could
    push an individual criterion over its declared tail mass.
    """
    reference = _reference()
    alpha = 0.05
    thresholds = calibration.thresholds_for_alpha(reference, alpha)
    for key, (column, _) in calibration.CONTINUOUS_CRITERIA.items():
        exact = float(np.quantile(reference[column].dropna(), 1 - alpha))
        assert thresholds[key] >= exact


def test_per_criterion_table_includes_the_joint_row():
    reference = _reference()
    result = calibration.calibrate(reference)
    table = calibration.per_criterion_table(reference, result)

    assert len(table) == len(calibration.CRITERION_CHECK_COLUMN) + 1
    joint = table[table["criterion"].str.startswith("JOINT")]
    assert len(joint) == 1
    assert joint["arm_r_flag_rate"].iloc[0] == pytest.approx(result["joint_flag_rate"])


def test_positive_controls_split_into_detectable_and_declared_blind_spot():
    """brief SS4.5's <=5% ceiling buys loose thresholds, so sensitivity has to be
    demonstrated rather than assumed -- and the blind spot declared rather than
    discovered later.
    """
    from rhmoo.config import load_config

    thresholds = load_config("configs/main.yaml").implausibility_battery
    controls = calibration.score_positive_controls(thresholds)

    detectable = controls[controls["group"] == "detectable"]
    assert len(detectable) == len(calibration.POSITIVE_CONTROLS_DETECTABLE)
    assert detectable["implausible"].all(), "the calibrated battery lost its teeth"

    blind = controls[controls["group"] == "blind_spot"]
    assert len(blind) == len(calibration.POSITIVE_CONTROLS_BLIND_SPOT)
    # Documented in docs/LIMITATIONS.md: the frozen criteria contain no ring
    # topology check, so these are expected NOT to be flagged. If this ever
    # starts passing, the limitation text needs updating.
    assert not blind["implausible"].any()


def test_committed_calibration_matches_the_frozen_thresholds():
    import json
    from pathlib import Path

    frozen = json.loads(Path("calibration/battery_frozen.json").read_text())
    assert frozen["joint_flag_rate_on_arm_r"] <= frozen["target_flag_rate"]
    assert frozen["positive_controls_detectable_all_flagged"] is True
    assert set(frozen["implausibility_battery"]) == set(
        list(calibration.CONTINUOUS_CRITERIA) + list(calibration.COUNT_CRITERIA) + ["fail_threshold_count"]
    )

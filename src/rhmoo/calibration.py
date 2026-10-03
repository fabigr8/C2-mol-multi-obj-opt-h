"""Implausibility-battery calibration against arm R (brief SS4.5 -- blocking).

The problem this solves: a pre-declared battery that most approved drugs also
fail makes every implausibility number downstream uninterpretable. That is the
mirror-image failure of post-hoc threshold picking, and brief SS4.5 therefore
requires thresholds chosen so that no more than 5% of the approved-drug
reference set is flagged.

The scheme is one scalar knob and one declared monotone search:

1. The criteria set is fixed as the seven pre-declared checks. Only thresholds
   move. (Adding criteria at a fixed `fail_threshold_count` *loosens* every
   threshold, because each must sit further into the approved-drug tail to hold
   the joint rate -- so the set is not up for negotiation here.)
2. Each threshold is a function of a single tail mass `alpha`:
   - continuous upper-bounded criteria take arm R's `(1 - alpha)` empirical
     quantile, rounded *up* to a declared grid so the frozen value is a
     readable number rather than a six-decimal quantile;
   - count criteria take the smallest integer `c` with
     `P_armR(count > c) <= alpha`. This is what turns a zero-tolerance alert
     count into a calibrated one, and it is where the real problem lives:
     roughly half of all approved drugs carry at least one Brenk alert.
3. `alpha` is scanned over a declared grid, and the *largest* alpha whose joint
   "fails >= threshold" rate on arm R is <= the target is chosen -- i.e. the
   most sensitive battery consistent with the constraint. The joint rate is
   monotone non-decreasing in alpha, so the solution is unique.

Why this is pre-registration rather than post-hoc fitting, mechanically:
the only data input is arm R; no arm A molecule exists when it runs; the rule
has a single scalar degree of freedom solved by a declared scan, which removes
the per-criterion freedom that post-hoc picking exploits; and the runner
refuses to start unless the active config's battery byte-matches the frozen
output (see `rhmoo.preflight`).

Sensitivity is demonstrated separately, without looking at arm A, by the
pre-declared positive controls in `POSITIVE_CONTROLS`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from rhmoo.metrics import battery_from_detail, physchem_panel, sa_scores, structural_alerts

#: Target: no more than this fraction of approved drugs flagged (brief SS4.5).
TARGET_FLAG_RATE = 0.05

#: Declared alpha grid, searched from loosest to tightest. The grid resolution
#: is deliberate: it makes the outcome a rule with a coarse knob rather than a
#: continuous fit to the target.
ALPHA_GRID = tuple(round(0.10 - 0.01 * i, 2) for i in range(10))

#: Continuous criteria: config key -> (per-molecule column, rounding grid).
CONTINUOUS_CRITERIA = {
    "mw_max": ("molecular_weight", 10.0),
    "clogp_max": ("clogp", 0.5),
    "tpsa_max": ("tpsa", 10.0),
    "rotatable_bonds_max": ("rotatable_bonds", 1.0),
    "sa_score_max": ("sa_score", 0.5),
}

#: Count criteria: config key -> per-molecule column.
COUNT_CRITERIA = {
    "pains_alerts_max": "pains_alerts",
    "brenk_alerts_max": "brenk_alerts",
}

#: Which battery check each config key produces, for the per-criterion table.
CRITERION_CHECK_COLUMN = {
    "mw_max": "fails_mw",
    "clogp_max": "fails_clogp",
    "tpsa_max": "fails_tpsa",
    "rotatable_bonds_max": "fails_rotatable_bonds",
    "sa_score_max": "fails_sa",
    "pains_alerts_max": "fails_pains",
    "brenk_alerts_max": "fails_brenk",
}

#: Pre-declared number of criteria a molecule must fail to count as implausible.
FAIL_THRESHOLD_COUNT = 2

#: Pre-declared positive controls: hand-written degenerate structures of the
#: kind a naive optimizer is expected to produce. Committed as part of the
#: pre-registration so that the calibrated battery's sensitivity is a measured,
#: recorded property rather than a hope -- and measurable without looking at a
#: single arm A molecule.
#:
#: Split into two declared groups, because the measurement showed the battery
#: is not uniformly sensitive:
#:
#: - `DETECTABLE` degeneracy is driven by size, lipophilicity, flexibility or
#:   structural alerts, which the seven frozen criteria do measure.
#: - `BLIND_SPOT` degeneracy is driven by ring topology -- an enormous
#:   macrocycle, or a heavily fused polyaromatic. The frozen criteria contain
#:   no ring-count, ring-size or aromaticity check, so such molecules fail at
#:   most one criterion and are *not* flagged, even though they are plainly
#:   implausible.
#:
#: The blind spot is declared here rather than patched away: brief SS4.5 fixes
#: the criteria set, and adding ring criteria at a fixed fail_threshold_count
#: would loosen every other threshold (each would have to sit further into the
#: approved-drug tail to hold the <=5% joint rate). So the correct output is to
#: report the limitation -- see docs/LIMITATIONS.md -- and to read the
#: implausibility rate as a lower bound on implausibility, never as a complete
#: account of it.
POSITIVE_CONTROLS_DETECTABLE = {
    "long_alkane_chain": "C" * 60,
    "polyhalogenated_chain": "C(F)(F)" * 20,
    "high_mw_repeat": "c1ccccc1" + "CCO" * 40,
}

POSITIVE_CONTROLS_BLIND_SPOT = {
    "large_macrocycle": "C1" + "C" * 40 + "1",
    "fused_polyaromatic": "c1ccc2cc3cc4cc5cc6ccccc6cc5cc4cc3cc2c1",
}

POSITIVE_CONTROLS = {**POSITIVE_CONTROLS_DETECTABLE, **POSITIVE_CONTROLS_BLIND_SPOT}


def _round_up(value: float, grid: float) -> float:
    rounded = math.ceil(value / grid) * grid
    # Avoid 879.9999999 style artifacts from binary floating point.
    return round(rounded, 6)


def thresholds_for_alpha(reference: pd.DataFrame, alpha: float) -> dict[str, float]:
    """Battery thresholds implied by tail mass `alpha` on the reference set."""
    thresholds: dict[str, float] = {}

    for key, (column, grid) in CONTINUOUS_CRITERIA.items():
        values = reference[column].dropna().to_numpy(dtype=float)
        quantile = float(np.quantile(values, 1.0 - alpha))
        thresholds[key] = _round_up(quantile, grid)

    for key, column in COUNT_CRITERIA.items():
        counts = reference[column].dropna().to_numpy()
        # Smallest integer cutoff whose exceedance rate is within alpha.
        cutoff = int(counts.max()) if len(counts) else 0
        for candidate in range(0, int(counts.max()) + 1 if len(counts) else 1):
            if float((counts > candidate).mean()) <= alpha:
                cutoff = candidate
                break
        thresholds[key] = cutoff

    thresholds["fail_threshold_count"] = FAIL_THRESHOLD_COUNT
    return thresholds


def joint_flag_rate(reference: pd.DataFrame, thresholds: dict[str, float]) -> float:
    return float(battery_from_detail(reference, thresholds)["implausible"].mean())


def calibrate(
    reference: pd.DataFrame,
    target: float = TARGET_FLAG_RATE,
    alpha_grid: tuple[float, ...] = ALPHA_GRID,
) -> dict:
    """Run the declared search. Returns the chosen alpha, the frozen thresholds,
    and the full scan so the decision is auditable rather than asserted.
    """
    scan = []
    chosen = None
    for alpha in alpha_grid:
        thresholds = thresholds_for_alpha(reference, alpha)
        rate = joint_flag_rate(reference, thresholds)
        scan.append({"alpha": alpha, "joint_flag_rate": rate, "within_target": rate <= target})
        if chosen is None and rate <= target:
            chosen = (alpha, thresholds, rate)

    if chosen is None:
        raise ValueError(
            f"no alpha in {alpha_grid} achieves a joint flag rate <= {target} on the reference set; "
            "the criteria set or the target needs to be revisited explicitly, not silently widened"
        )

    alpha, thresholds, rate = chosen
    return {
        "alpha": alpha,
        "target_flag_rate": target,
        "joint_flag_rate": rate,
        "thresholds": thresholds,
        "scan": scan,
        "n_reference": int(len(reference)),
    }


def per_criterion_table(reference: pd.DataFrame, result: dict) -> pd.DataFrame:
    """brief SS4.5: flag rate of each individual criterion on arm R.

    This is the table that shows which criteria actually do the work, and is
    the honest counterweight to a loose-looking individual threshold.
    """
    thresholds = result["thresholds"]
    checks = battery_from_detail(reference, thresholds)
    alpha = result["alpha"]

    rows = []
    for key, check_column in CRITERION_CHECK_COLUMN.items():
        if key in CONTINUOUS_CRITERIA:
            column, grid = CONTINUOUS_CRITERIA[key]
            rule = f"ceil(quantile(arm_R[{column}], 1-alpha) / {grid}) * {grid}"
        else:
            column = COUNT_CRITERIA[key]
            rule = f"min integer c with P(arm_R[{column}] > c) <= alpha"
        rows.append(
            {
                "criterion": key,
                "column": column,
                "threshold": thresholds[key],
                "arm_r_flag_rate": float(checks[check_column].mean()),
                "alpha": alpha,
                "rule": rule,
            }
        )

    rows.append(
        {
            "criterion": f"JOINT_fails_ge_{thresholds['fail_threshold_count']}",
            "column": "implausible",
            "threshold": thresholds["fail_threshold_count"],
            "arm_r_flag_rate": float(checks["implausible"].mean()),
            "alpha": alpha,
            "rule": f"fail_count >= {thresholds['fail_threshold_count']}",
        }
    )
    return pd.DataFrame(rows)


def fail_count_histogram(reference: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    checks = battery_from_detail(reference, thresholds)
    counts = checks["fail_count"].value_counts().sort_index()
    return pd.DataFrame({"fail_count": counts.index, "n_approved_drugs": counts.to_numpy()})


def score_positive_controls(thresholds: dict[str, float]) -> pd.DataFrame:
    """Apply the calibrated battery to the pre-declared degenerate structures.

    The `group` column records which declared group each control belongs to, so
    the committed table shows both what the battery catches and what it
    structurally cannot.
    """
    names = list(POSITIVE_CONTROLS)
    smiles = [POSITIVE_CONTROLS[n] for n in names]
    groups = ["detectable" if n in POSITIVE_CONTROLS_DETECTABLE else "blind_spot" for n in names]

    detail = pd.concat([physchem_panel(smiles), structural_alerts(smiles)], axis=1)
    detail["sa_score"] = sa_scores(smiles)
    checks = battery_from_detail(detail, thresholds)

    out = pd.DataFrame({"control": names, "group": groups, "smiles": smiles})
    return pd.concat([out, checks.reset_index(drop=True)], axis=1)

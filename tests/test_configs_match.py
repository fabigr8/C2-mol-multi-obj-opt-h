"""The dress rehearsal must differ from the main config only in budget.

Without this, the rehearsal silently drifts from the thing it rehearses -- which
is the exact failure it exists to prevent, since its whole purpose is to prove
the main run's code path works before anyone spends hours on it.
"""

from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

pytestmark = pytest.mark.smoke

#: The only keys allowed to differ. Everything else -- arms, seeds, theta
#: sweep, objective, battery, scorer variants, decoupling -- must match.
ALLOWED_DIFFERENCES = {
    ("name",),
    ("k_top",),
    ("starting_population_baseline_size",),
    ("ga", "population_size"),
    ("ga", "n_generations"),
}


def _flatten(node, prefix=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _flatten(value, prefix + (key,))
    else:
        yield prefix, node


def test_dress_rehearsal_differs_from_main_only_in_budget():
    main = yaml.safe_load((CONFIG_DIR / "main.yaml").read_text())
    rehearsal = yaml.safe_load((CONFIG_DIR / "dress_rehearsal.yaml").read_text())

    main_flat = dict(_flatten(main))
    rehearsal_flat = dict(_flatten(rehearsal))

    assert set(main_flat) == set(rehearsal_flat), "the two configs have different keys"

    differing = {k for k in main_flat if main_flat[k] != rehearsal_flat[k]}
    unexpected = differing - ALLOWED_DIFFERENCES

    assert unexpected == set(), f"dress_rehearsal.yaml drifted from main.yaml in {sorted(unexpected)}"
    # And the budget keys really are smaller, so this is a rehearsal and not a copy.
    assert rehearsal_flat[("ga", "population_size")] < main_flat[("ga", "population_size")]
    assert rehearsal_flat[("ga", "n_generations")] < main_flat[("ga", "n_generations")]


def test_dress_rehearsal_keeps_the_full_arm_and_seed_structure():
    main = yaml.safe_load((CONFIG_DIR / "main.yaml").read_text())
    rehearsal = yaml.safe_load((CONFIG_DIR / "dress_rehearsal.yaml").read_text())

    assert rehearsal["arms"] == main["arms"]
    assert rehearsal["seeds"] == main["seeds"]
    assert rehearsal["applicability_domain"]["theta_values"] == main["applicability_domain"]["theta_values"]
    assert rehearsal["implausibility_battery"] == main["implausibility_battery"]
    assert rehearsal["independent_scorers"] == main["independent_scorers"]
    assert rehearsal["decoupling"] == main["decoupling"]


def test_every_config_uses_the_frozen_battery():
    """brief SS9: a pre-declared threshold is only pre-declared if every config
    that could be run carries the same frozen values.
    """
    import json

    frozen = json.loads((CONFIG_DIR.parent / "calibration" / "battery_frozen.json").read_text())
    expected = frozen["implausibility_battery"]

    for path in CONFIG_DIR.glob("*.yaml"):
        battery = yaml.safe_load(path.read_text())["implausibility_battery"]
        assert {k: float(v) for k, v in battery.items()} == {
            k: float(v) for k, v in expected.items()
        }, f"{path.name} does not carry the frozen calibration"

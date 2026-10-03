from pathlib import Path

import pytest

from rhmoo.config import ArmConfig, ObjectiveComponent, config_hash, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

pytestmark = pytest.mark.smoke


def test_load_smoke_config():
    config = load_config(CONFIG_DIR / "smoke.yaml")
    assert config.ga.population_size == 20
    assert config.ga.n_generations == 5
    assert len(config.arms) == 5
    assert {arm.name for arm in config.arms} == {"D", "R", "A", "B", "C"}
    assert {arm.name: arm.kind for arm in config.arms} == {
        "D": "sample",
        "R": "reference",
        "A": "optimize",
        "B": "optimize",
        "C": "optimize",
    }


def test_load_main_config():
    config = load_config(CONFIG_DIR / "main.yaml")
    assert config.ga.population_size == 200
    assert config.ga.n_generations == 100
    assert len(config.seeds) == 5


def test_config_hash_is_stable_and_content_sensitive():
    a = load_config(CONFIG_DIR / "smoke.yaml")
    b = load_config(CONFIG_DIR / "smoke.yaml")
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(load_config(CONFIG_DIR / "main.yaml"))


def test_invalid_direction_rejected():
    with pytest.raises(ValueError):
        ObjectiveComponent(name="x", source="rdkit", direction="up", weight=1.0)


def test_negative_weight_rejected():
    with pytest.raises(ValueError):
        ObjectiveComponent(name="x", source="rdkit", direction="max", weight=-1.0)


def test_arm_name_is_validated_by_shape_not_a_whitelist():
    """A closed whitelist already blocked arm R and would equally block the
    brief SS6.1 iterated-patching rounds; what matters is that the name is safe
    as a filename and a column value.
    """
    ArmConfig(name="patch_round_1", description="x", kind="optimize",
              use_ad_constraint=False, use_synth_constraint=False)
    for bad in ("1A", "a b", "a/b", ""):
        with pytest.raises(ValueError):
            ArmConfig(name=bad, description="x", kind="optimize",
                      use_ad_constraint=False, use_synth_constraint=False)


def test_invalid_arm_kind_rejected():
    with pytest.raises(ValueError):
        ArmConfig(name="A", description="x", kind="wander",
                  use_ad_constraint=False, use_synth_constraint=False)


def test_constraints_require_an_optimized_arm():
    """A constraint on a non-optimized arm is silently meaningless, so it is an
    error rather than a no-op.
    """
    with pytest.raises(ValueError):
        ArmConfig(name="R", description="x", kind="reference",
                  use_ad_constraint=True, use_synth_constraint=False)


def test_headline_scorer_variant_must_be_among_variants():
    from rhmoo.config import IndependentScorersConfig

    with pytest.raises(ValueError):
        IndependentScorersConfig(variants=("model_control",), headline_variant="data_control_half",
                                 cache_path="x.parquet")


def test_decoupling_primary_margin_must_be_in_the_sweep():
    from rhmoo.config import DecouplingConfig

    DecouplingConfig(margin=0.1, persistence=3, margin_sweep=(0.05, 0.1, 0.15))
    with pytest.raises(ValueError):
        DecouplingConfig(margin=0.2, persistence=3, margin_sweep=(0.05, 0.1, 0.15))


def test_n_elite_is_defined_once():
    """The GA and the SS4.3 elite-set analysis must not be able to disagree."""
    config = load_config(CONFIG_DIR / "main.yaml")
    assert config.ga.n_elite == max(1, int(config.ga.population_size * config.ga.elitism_fraction))

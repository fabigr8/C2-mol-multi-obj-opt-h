from pathlib import Path

import pytest

from rhmoo.config import ObjectiveComponent, config_hash, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

pytestmark = pytest.mark.smoke


def test_load_smoke_config():
    config = load_config(CONFIG_DIR / "smoke.yaml")
    assert config.ga.population_size == 20
    assert config.ga.n_generations == 5
    assert len(config.arms) == 4
    assert {arm.name for arm in config.arms} == {"D", "A", "B", "C"}


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

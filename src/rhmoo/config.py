"""Experiment config schema.

Loaded from a single YAML file per run (brief SS6: config-driven, no hardcoded
constants in logic). `config_hash` gives a stable fingerprint for provenance
logging (brief SS6: structured run logs record the config hash).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

_VALID_DIRECTIONS = {"max", "min"}
_VALID_ARM_NAMES = {"D", "A", "B", "C"}


@dataclass(frozen=True)
class ObjectiveComponent:
    name: str
    source: str  # "admet-ai" or "rdkit"
    direction: str  # "max" or "min"
    weight: float

    def __post_init__(self) -> None:
        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(f"component {self.name!r}: direction must be one of {_VALID_DIRECTIONS}")
        if self.weight < 0:
            raise ValueError(f"component {self.name!r}: weight must be >= 0")


@dataclass(frozen=True)
class ObjectiveConfig:
    components: tuple[ObjectiveComponent, ...]
    normalization: str
    aggregation: str
    cache_path: str


@dataclass(frozen=True)
class GAConfig:
    population_size: int
    n_generations: int
    elitism_fraction: float
    tournament_k: int
    crossover_enabled: bool
    crossover_rate: float
    mutation_rate: float


@dataclass(frozen=True)
class ApplicabilityDomainConfig:
    theta_values: tuple[float, ...]
    fingerprint_radius: int
    fingerprint_bits: int
    fingerprint_cache_path: str


@dataclass(frozen=True)
class SynthesizabilityConfig:
    sa_score_threshold: float


@dataclass(frozen=True)
class ArmConfig:
    name: str
    description: str
    use_ad_constraint: bool
    use_synth_constraint: bool

    def __post_init__(self) -> None:
        if self.name not in _VALID_ARM_NAMES:
            raise ValueError(f"arm name must be one of {_VALID_ARM_NAMES}, got {self.name!r}")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seeds: tuple[int, ...]
    k_top: int
    starting_population_path: str
    reference_set_path: str
    results_dir: str
    objective: ObjectiveConfig
    ga: GAConfig
    applicability_domain: ApplicabilityDomainConfig
    synthesizability: SynthesizabilityConfig
    arms: tuple[ArmConfig, ...]
    implausibility_battery: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("at least one arm must be configured")


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())

    objective_raw = raw["objective"]
    components = tuple(ObjectiveComponent(**c) for c in objective_raw["components"])
    objective = ObjectiveConfig(
        components=components,
        normalization=objective_raw["normalization"],
        aggregation=objective_raw["aggregation"],
        cache_path=objective_raw["cache_path"],
    )

    ga = GAConfig(**raw["ga"])

    ad_raw = raw["applicability_domain"]
    applicability_domain = ApplicabilityDomainConfig(
        theta_values=tuple(ad_raw["theta_values"]),
        fingerprint_radius=ad_raw["fingerprint_radius"],
        fingerprint_bits=ad_raw["fingerprint_bits"],
        fingerprint_cache_path=ad_raw["fingerprint_cache_path"],
    )

    synthesizability = SynthesizabilityConfig(**raw["synthesizability"])
    arms = tuple(ArmConfig(**a) for a in raw["arms"])

    return ExperimentConfig(
        name=raw["name"],
        seeds=tuple(raw["seeds"]),
        k_top=raw["k_top"],
        starting_population_path=raw["starting_population_path"],
        reference_set_path=raw["reference_set_path"],
        results_dir=raw["results_dir"],
        objective=objective,
        ga=ga,
        applicability_domain=applicability_domain,
        synthesizability=synthesizability,
        arms=arms,
        implausibility_battery=raw["implausibility_battery"],
    )


def config_hash(config: ExperimentConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

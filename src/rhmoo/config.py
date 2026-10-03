"""Experiment config schema.

Loaded from a single YAML file per run (brief SS7: config-driven, no hardcoded
constants in logic). `config_hash` gives a stable fingerprint for provenance
logging (brief SS7: structured run logs record the config hash).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

_VALID_DIRECTIONS = {"max", "min"}

#: Arm names are validated by shape, not by a closed whitelist. A whitelist
#: would have to be edited for every new arm -- it already blocked arm R, and
#: would equally block the brief SS6.1 iterated-patching rounds. What actually
#: matters is that the name is a safe filename/column value.
_ARM_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

#: What the runner does with an arm, replacing the arm-name string matching
#: that used to be spread across runner.py:
#: - "optimize": run the GA (arms A/B/C)
#: - "sample": draw a random sample from the starting library, no optimization (arm D)
#: - "reference": load a pre-scored frozen set of molecules, no optimization,
#:   seed-invariant (arm R, the approved drugs)
_VALID_ARM_KINDS = {"optimize", "sample", "reference"}


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
    # Extra parameters for the chosen aggregation (e.g. soft_min_tau). Empty
    # for the default weighted geometric mean.
    aggregation_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GAConfig:
    population_size: int
    n_generations: int
    elitism_fraction: float
    tournament_k: int
    crossover_enabled: bool
    crossover_rate: float
    mutation_rate: float

    @property
    def n_elite(self) -> int:
        """Elite count, defined once so the GA and the SS4.3 elite-set analysis
        cannot disagree about it.
        """
        return max(1, int(self.population_size * self.elitism_fraction))


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
class IndependentScorersConfig:
    """brief SS4.2: several independent scorers, reported separately."""

    variants: tuple[str, ...]
    # The variant whose divergence is the pre-declared headline H3 number.
    # Brief SS4.2: "The data-control is the load-bearing scorer."
    headline_variant: str
    cache_path: str

    def __post_init__(self) -> None:
        if self.headline_variant not in self.variants:
            raise ValueError(
                f"headline_variant {self.headline_variant!r} is not among variants {self.variants}"
            )


@dataclass(frozen=True)
class DecouplingConfig:
    """brief SS4.3 / H3b: pre-declared detector for the generation at which the
    objective score and the independent-scorer estimate separate.
    """

    margin: float
    persistence: int
    margin_sweep: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.persistence < 1:
            raise ValueError("decoupling.persistence must be >= 1")
        if self.margin not in self.margin_sweep:
            raise ValueError("decoupling.margin must appear in margin_sweep so the primary is in the sweep")


@dataclass(frozen=True)
class ReferenceArmConfig:
    """brief SS3.3 arm R: the approved-drug set, scored but never optimized.

    Scored once before the run by scripts/score_arm_r.py, because brief SS4.5
    requires the battery to be calibrated against it *before* any arm A
    molecule exists. The runner loads that frozen artifact rather than
    recomputing, so the arm R reported in RESULTS.md is literally the set the
    thresholds were calibrated on.
    """

    artifact_path: str
    # None = use the whole approved-drug set (the default; a top-k subset would
    # bias both the percentile scale and the calibration set).
    sample_size: int | None


@dataclass(frozen=True)
class ArmConfig:
    name: str
    description: str
    kind: str
    use_ad_constraint: bool
    use_synth_constraint: bool

    def __post_init__(self) -> None:
        if not _ARM_NAME_PATTERN.match(self.name):
            raise ValueError(
                f"arm name must match {_ARM_NAME_PATTERN.pattern} (safe as a filename and a column "
                f"value), got {self.name!r}"
            )
        if self.kind not in _VALID_ARM_KINDS:
            raise ValueError(f"arm {self.name!r}: kind must be one of {_VALID_ARM_KINDS}, got {self.kind!r}")
        if self.kind != "optimize" and (self.use_ad_constraint or self.use_synth_constraint):
            raise ValueError(
                f"arm {self.name!r}: constraints only apply to an optimized arm, but kind={self.kind!r}"
            )


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seeds: tuple[int, ...]
    k_top: int
    starting_population_path: str
    reference_set_path: str
    results_dir: str
    # H3 starting-population divergence baseline (brief SS4.2) sample size; None
    # means use the whole starting population.
    starting_population_baseline_size: int | None
    objective: ObjectiveConfig
    ga: GAConfig
    applicability_domain: ApplicabilityDomainConfig
    synthesizability: SynthesizabilityConfig
    independent_scorers: IndependentScorersConfig
    decoupling: DecouplingConfig
    reference_arm: ReferenceArmConfig
    arms: tuple[ArmConfig, ...]
    implausibility_battery: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("at least one arm must be configured")
        names = [a.name for a in self.arms]
        if len(set(names)) != len(names):
            raise ValueError(f"arm names must be unique, got {names}")

    def arms_of_kind(self, kind: str) -> tuple[ArmConfig, ...]:
        return tuple(a for a in self.arms if a.kind == kind)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())

    objective_raw = raw["objective"]
    components = tuple(ObjectiveComponent(**c) for c in objective_raw["components"])
    objective = ObjectiveConfig(
        components=components,
        normalization=objective_raw["normalization"],
        aggregation=objective_raw["aggregation"],
        cache_path=objective_raw["cache_path"],
        aggregation_params=dict(objective_raw.get("aggregation_params") or {}),
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

    scorers_raw = raw["independent_scorers"]
    independent_scorers = IndependentScorersConfig(
        variants=tuple(scorers_raw["variants"]),
        headline_variant=scorers_raw["headline_variant"],
        cache_path=scorers_raw["cache_path"],
    )

    decoupling_raw = raw["decoupling"]
    decoupling = DecouplingConfig(
        margin=decoupling_raw["margin"],
        persistence=decoupling_raw["persistence"],
        margin_sweep=tuple(decoupling_raw["margin_sweep"]),
    )

    reference_raw = raw["reference_arm"]
    reference_arm = ReferenceArmConfig(
        artifact_path=reference_raw["artifact_path"],
        sample_size=reference_raw.get("sample_size"),
    )

    arms = tuple(ArmConfig(**a) for a in raw["arms"])

    return ExperimentConfig(
        name=raw["name"],
        seeds=tuple(raw["seeds"]),
        k_top=raw["k_top"],
        starting_population_path=raw["starting_population_path"],
        reference_set_path=raw["reference_set_path"],
        results_dir=raw["results_dir"],
        starting_population_baseline_size=raw.get("starting_population_baseline_size"),
        objective=objective,
        ga=ga,
        applicability_domain=applicability_domain,
        synthesizability=synthesizability,
        independent_scorers=independent_scorers,
        decoupling=decoupling,
        reference_arm=reference_arm,
        arms=arms,
        implausibility_battery=raw["implausibility_battery"],
    )


def config_hash(config: ExperimentConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

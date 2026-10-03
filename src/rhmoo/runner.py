"""Experiment runner: orchestrates arms x seeds (x theta) using the shared
objective/constraint/independent-scorer/metrics modules, and writes one raw
parquet of every evaluated molecule plus summary tables under `results/`.

Arms are dispatched on `ArmConfig.kind`, not on the arm's name:

- `sample` (arm D): draws a `k_top`-sized random sample from the starting
  library per seed; metrics cover the whole sample (no GA, so nothing to rank).
- `reference` (arm R): loads the frozen, pre-scored approved-drug artifact and
  verifies its hash. Seed-invariant, so it runs once rather than once per seed.
  It is loaded rather than rescored because brief SS4.5 requires the battery to
  be calibrated against arm R *before* the run, and the arm R reported here
  must be the same molecules the thresholds were calibrated on.
- `optimize` (arms A/B/C): runs the GA per seed; the top `k_top` of the *final*
  population by fitness is the metrics set (brief SS4).

Which arm sweeps theta is derived from its constraints rather than its name: an
AD-constrained arm without the synthesizability constraint is the one that
produces H4's trade-off curve, so it sweeps the full theta range. An arm with
both constraints needs only one AD setting alongside SA, and takes the middle
of the declared range.

The H3 "starting population" divergence baseline is computed once per
experiment, since it depends on neither arm nor seed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from rhmoo.applicability_domain import ReferenceFingerprintIndex, soft_penalty_above_threshold_is_bad
from rhmoo.config import ArmConfig, ExperimentConfig, config_hash, load_config
from rhmoo.decoupling import decoupling_curves, decoupling_summary
from rhmoo.ga import SelfiesGA
from rhmoo.independent_scorer import IndependentScorerSet, independent_pairs
from rhmoo.metrics import (
    arm_r_percentile,
    battery_from_detail,
    component_breakdown,
    component_log_contributions,
    dominant_component,
    generic_scaffold,
    independent_divergence,
    leave_one_out_deltas,
    murcko_scaffold,
    sa_scores,
    scaffold_clusters,
)
from rhmoo.normalization import DrugBankPercentileNormalizer
from rhmoo.objective import aggregate, normalize_components
from rhmoo.predictors import ADMETPredictor, canonicalize
from rhmoo.preflight import check_experiment
from rhmoo.run_log import run_log
from rhmoo.scoring import score_molecule_set
from rhmoo.synthesizability import sa_penalty


class ObjectivePipeline:
    """Composite objective (+ optional AD/SA constraints) as a GA `fitness_fn`.

    Returns one row per input SMILES with `fitness`, `valid`, every raw and
    normalized objective component, and (if enabled) the AD similarity / SA
    score used for the constraint penalty -- all propagated into the GA's
    full trajectory log (brief SS3.2/SS9: every evaluated molecule is data).
    """

    def __init__(
        self,
        config: ExperimentConfig,
        predictor: ADMETPredictor,
        normalizer: DrugBankPercentileNormalizer,
        reference_index: ReferenceFingerprintIndex | None = None,
        theta: float | None = None,
        use_synth_constraint: bool = False,
    ):
        self.config = config
        self.predictor = predictor
        self.normalizer = normalizer
        self.reference_index = reference_index
        self.theta = theta
        self.use_synth_constraint = use_synth_constraint
        self.component_names = [c.name for c in config.objective.components]

    def __call__(self, smiles: list[str]) -> pd.DataFrame:
        raw = self.predictor.predict(smiles)
        normalized = normalize_components(self.config.objective, self.normalizer, raw[self.component_names])
        composite = aggregate(self.config.objective, normalized)

        result = pd.DataFrame(index=raw.index)
        result["smiles"] = smiles
        for name in self.component_names:
            result[name] = raw[name]
            result[f"{name}_normalized"] = normalized[name]
        # Representation-matched reliability signal (brief SS3.3): the ECFP4
        # applicability domain does not describe where a Chemprop GNN is
        # competent, but the spread across its own ensemble members does.
        for column in raw.columns:
            if column.endswith("_ensemble_std"):
                result[column] = raw[column]
        result["objective_score"] = composite
        result["valid"] = raw["valid"]
        fitness = composite.to_numpy(dtype=float)

        if self.reference_index is not None:
            result["ad_max_tanimoto"] = self.reference_index.max_tanimoto(smiles)
            fitness = fitness * soft_penalty_above_threshold_is_bad(
                result["ad_max_tanimoto"].to_numpy(), self.theta
            )

        if self.use_synth_constraint:
            result["sa_score"] = sa_scores(smiles)
            fitness = fitness * sa_penalty(
                result["sa_score"].to_numpy(), self.config.synthesizability.sa_score_threshold
            )

        # ADMET-AI / normalization failures (rare, brief SS9: log, don't silently
        # drop): score 0 rather than corrupt GA selection with a NaN fitness.
        result["fitness"] = np.nan_to_num(fitness, nan=0.0)
        return result


def _sample_population(pool: list[str], n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    n = min(n, len(pool))
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in idx]


def best_unique_molecules(trajectory: pd.DataFrame, k: int) -> tuple[list[str], int]:
    """The k highest-fitness distinct molecules the run evaluated.

    brief SS4 asks for "the top-k molecules per arm per seed"; it does not say
    the final population, and drawing from the final population is fragile.
    An elitist GA converges, so a final population of 200 slots routinely holds
    only a handful of distinct structures -- and the highest-scoring molecules
    the optimizer found, which is exactly where the exploits are, may have been
    discovered and then crowded out generations earlier.

    Keyed on canonical SMILES, because SELFIES decoding emits whichever
    spelling the token sequence produces and the same molecule otherwise
    appears several times. Returns the selection and the total number of
    distinct molecules evaluated.

    Ties are broken by SMILES (groupby sorts by key, then a stable sort by
    fitness preserves that order), so the selection does not depend on
    evaluation order.
    """
    valid = trajectory[trajectory["smiles"].notna() & trajectory["fitness"].notna()]
    if valid.empty:
        return [], 0
    frame = pd.DataFrame(
        {
            "smiles": [canonicalize(s) or s for s in valid["smiles"]],
            "fitness": valid["fitness"].to_numpy(dtype=float),
        }
    )
    best = frame.groupby("smiles", as_index=False)["fitness"].max()
    best = best.sort_values("fitness", ascending=False, kind="stable")
    return best["smiles"].head(k).tolist(), len(best)


def sweeps_theta(arm: ArmConfig) -> bool:
    """An AD-constrained arm without the synthesizability constraint is the one
    whose theta sweep produces H4's trade-off curve (brief SS3.3).
    """
    return arm.kind == "optimize" and arm.use_ad_constraint and not arm.use_synth_constraint


def _arm_theta(config: ExperimentConfig, arm: ArmConfig) -> float | None:
    if not arm.use_ad_constraint:
        return None
    thetas = config.applicability_domain.theta_values
    return thetas[len(thetas) // 2]


def arm_plan(config: ExperimentConfig) -> list[dict]:
    """Every (arm, label, theta, seed) cell the experiment will run.

    Shared by `run_experiment` and `--dry-run`, so the ETA is computed from the
    same plan that executes.
    """
    cells = []
    for arm in config.arms:
        thetas = config.applicability_domain.theta_values if sweeps_theta(arm) else [_arm_theta(config, arm)]
        for theta in thetas:
            label = f"{arm.name}_theta{theta:.2f}" if sweeps_theta(arm) else arm.name
            # Arm R is seed-invariant: the approved drugs do not depend on a
            # seed, so running it per seed would emit identical duplicate rows.
            seeds = [None] if arm.kind == "reference" else list(config.seeds)
            for seed in seeds:
                cells.append({"arm": arm, "label": label, "theta": theta, "seed": seed})
    return cells


def _load_arm_r(config: ExperimentConfig) -> pd.DataFrame:
    artifact = Path(config.reference_arm.artifact_path)
    if not artifact.is_absolute():
        artifact = Path(__file__).resolve().parents[2] / artifact
    detail = pd.read_parquet(artifact)
    if config.reference_arm.sample_size is not None:
        detail = detail.head(config.reference_arm.sample_size).reset_index(drop=True)
    return detail


def _run_one(
    config: ExperimentConfig,
    label: str,
    arm: ArmConfig,
    seed: int | None,
    theta: float | None,
    starting_pool: list[str],
    predictor: ADMETPredictor,
    normalizer: DrugBankPercentileNormalizer,
    predictor_reference_index: ReferenceFingerprintIndex,
    starting_population_index: ReferenceFingerprintIndex,
    scorers: IndependentScorerSet,
    arm_r_scores: np.ndarray | None,
) -> dict:
    """Runs one (arm, seed[, theta]) cell end to end."""
    elites = None
    n_unique_evaluated = final_unique = None
    curve_rows: list[pd.DataFrame] = []
    decoupling_rows: list[dict] = []

    if arm.kind == "reference":
        detail = _load_arm_r(config)
        trajectory = detail.copy()
        trajectory["generation"] = 0
        top_smiles = detail["smiles"].astype(str).tolist()

    elif arm.kind == "sample":
        top_smiles = _sample_population(starting_pool, config.k_top, seed)
        pipeline = ObjectivePipeline(config, predictor, normalizer)
        detail = score_molecule_set(
            top_smiles, pipeline, scorers, predictor_reference_index, starting_population_index
        )
        trajectory = detail.copy()
        trajectory["generation"] = 0

    else:  # optimize
        pipeline = ObjectivePipeline(
            config,
            predictor,
            normalizer,
            reference_index=predictor_reference_index if arm.use_ad_constraint else None,
            theta=theta,
            use_synth_constraint=arm.use_synth_constraint,
        )
        initial_population = _sample_population(starting_pool, config.ga.population_size, seed)
        ga = SelfiesGA(config.ga, pipeline, seed=seed)
        final_population, final_fitness, trajectory = ga.run(initial_population)
        top_smiles, n_unique_evaluated = best_unique_molecules(trajectory, config.k_top)
        # Population collapse is a real property of the search, so it is
        # measured and reported rather than being allowed to silently thin
        # every downstream statistic.
        final_unique = len({canonicalize(s) or s for s in final_population})
        # Re-score the selection explicitly (cheap: hits the persistent
        # predictor cache) rather than slicing the trajectory, so the
        # reported rows are unambiguous for a molecule seen in several
        # generations.
        detail = score_molecule_set(
            top_smiles, pipeline, scorers, predictor_reference_index, starting_population_index
        )

        # H3b (brief SS4.3): score the per-generation elite sets with every
        # scorer variant. Cheap only because the elite set is small and the
        # scorer cache is shared -- see rhmoo.decoupling.
        elites = ga.elite_log()
        elite_smiles = elites["smiles"].drop_duplicates().tolist()
        elite_scores = pipeline(elite_smiles).reset_index(drop=True)
        elite_scores = pd.concat([elite_scores, scorers.score(elite_smiles)], axis=1)
        for variant in config.independent_scorers.variants:
            curve = decoupling_curves(elites, elite_scores, config, normalizer, variant)
            curve.insert(0, "arm", label)
            curve.insert(1, "seed", seed)
            curve_rows.append(curve)
            for row in decoupling_summary(curve, config, variant):
                decoupling_rows.append({"arm": label, "seed": seed, "theta": theta, **row})

    component_names = [c.name for c in config.objective.components]
    weights = {c.name: c.weight for c in config.objective.components}

    battery = battery_from_detail(detail, config.implausibility_battery)
    normalized = detail[[f"{n}_normalized" for n in component_names]].copy()
    normalized.columns = component_names
    contributions = component_log_contributions(normalized, weights)
    loo = leave_one_out_deltas(normalized, weights)

    detail = detail.reset_index(drop=True)
    detail["arm"] = label
    detail["arm_kind"] = arm.kind
    detail["theta"] = theta
    detail["seed"] = seed
    detail["murcko_scaffold"] = [murcko_scaffold(s) for s in detail["smiles"]]
    detail["generic_scaffold"] = [generic_scaffold(s) for s in detail["smiles"]]
    detail["dominant_component"] = dominant_component(contributions, component_names).to_numpy()
    if arm_r_scores is not None:
        detail["arm_r_percentile"] = arm_r_percentile(
            detail["objective_score"].to_numpy(), arm_r_scores
        )
    detail = pd.concat([detail, battery.reset_index(drop=True), contributions.reset_index(drop=True),
                        loo.reset_index(drop=True)], axis=1)

    components = component_breakdown(detail, component_names)

    summary = {
        "arm": label,
        "arm_kind": arm.kind,
        "seed": seed,
        "theta": theta,
        "n_seeds": 1,
        "n_molecules_evaluated": len(trajectory),
        "n_molecules_reported": len(detail),
        "n_unique_molecules_evaluated": n_unique_evaluated,
        "topk_unique_molecules": None if n_unique_evaluated is None else len(detail),
        # Convergence diagnostic (brief SS9: report it, do not hide it). A
        # low fraction means the final population collapsed onto a few
        # structures -- informative about the search, and no longer able to
        # thin the reported top-k now that it is drawn from the whole run.
        "final_population_unique_molecules": final_unique,
        "final_population_unique_fraction": (
            None if final_unique is None else final_unique / config.ga.population_size
        ),
        "objective_score_mean": detail["objective_score"].mean(),
        "objective_score_std": detail["objective_score"].std(),
        "objective_score_max": detail["objective_score"].max(),
        "fitness_mean": detail["fitness"].mean() if "fitness" in detail else np.nan,
        "implausibility_rate": battery["implausible"].mean(),
        "ad_similarity_vs_reference_mean": float(detail["ad_similarity_vs_reference"].mean()),
        "ad_similarity_vs_starting_mean": float(detail["ad_similarity_vs_starting"].mean()),
        "sa_score_mean": float(np.nanmean(detail["sa_score"].to_numpy())),
        "n_scaffold_clusters": int(detail["murcko_scaffold"].nunique()),
    }
    if arm_r_scores is not None:
        summary["arm_r_percentile_mean"] = float(detail["arm_r_percentile"].mean())
        summary["arm_r_percentile_median"] = float(detail["arm_r_percentile"].median())
        # The headline number brief SS4.1 asks for: if this is 1.0, every
        # reported molecule outscores every approved drug.
        summary["arm_r_percentile_min"] = float(detail["arm_r_percentile"].min())
    for column in detail.columns:
        if column.endswith("_ensemble_std"):
            summary[f"{column}_mean"] = float(detail[column].mean())
    for _, row in components.iterrows():
        summary[f"component_{row['component']}_mean"] = row["mean"]
    for name in component_names:
        summary[f"component_{name}_log_share_mean"] = float(detail[f"{name}_log_share"].mean())

    # Tidy divergence rows, one per (variant, property) -- brief SS4.2 requires
    # this for three populations x three variants, which a wide summary table
    # cannot carry without synthesizing column names.
    population = {"reference": "arm_r", "sample": "starting_sample", "optimize": "optimized"}[arm.kind]
    divergence_rows = []
    for variant in config.independent_scorers.variants:
        divergence = independent_divergence(detail, detail, independent_pairs(variant))
        for _, row in divergence.iterrows():
            divergence_rows.append(
                {
                    "arm": label,
                    "arm_kind": arm.kind,
                    "theta": theta,
                    "seed": seed,
                    "population": population,
                    "variant": variant,
                    "is_headline_variant": variant == config.independent_scorers.headline_variant,
                    **row.to_dict(),
                }
            )

    return {
        "trajectory": trajectory,
        "summary": summary,
        "detail": detail,
        "elites": elites,
        "divergence": divergence_rows,
        "decoupling": decoupling_rows,
        "curves": curve_rows,
    }


def _starting_population_divergence(
    config: ExperimentConfig,
    starting_pool: list[str],
    predictor: ADMETPredictor,
    normalizer: DrugBankPercentileNormalizer,
    scorers: IndependentScorerSet,
) -> list[dict]:
    """H3 baseline (brief SS4.2): objective-vs-independent divergence computed
    once over the starting population, for comparison against each arm/seed's
    optimized-top-k divergence. The *difference* between the two is the
    headline number of the project.
    """
    sample_size = config.starting_population_baseline_size
    pool = starting_pool if sample_size is None else _sample_population(starting_pool, sample_size, seed=0)
    pipeline = ObjectivePipeline(config, predictor, normalizer)
    scored = pipeline(pool).reset_index(drop=True)
    scored = pd.concat([scored, scorers.score(pool)], axis=1)

    rows = []
    for variant in config.independent_scorers.variants:
        divergence = independent_divergence(scored, scored, independent_pairs(variant))
        for _, row in divergence.iterrows():
            rows.append(
                {
                    "arm": "starting_population",
                    "arm_kind": "baseline",
                    "theta": None,
                    "seed": None,
                    "population": "starting",
                    "variant": variant,
                    "is_headline_variant": variant == config.independent_scorers.headline_variant,
                    **row.to_dict(),
                }
            )
    return rows


def run_experiment(config_path: str | Path, results_root: str | Path | None = None) -> Path:
    config = load_config(config_path)
    # brief SS4.5/SS9: refuse to run against an uncalibrated or edited battery,
    # a missing scorer variant, or an arm R artifact that does not match its
    # manifest. Process claims are worth nothing unless something enforces them.
    check_experiment(config)

    out_dir = Path(results_root) if results_root else Path(config.results_dir)
    out_dir = out_dir / config.name
    raw_dir = out_dir / "raw"
    tables_dir = out_dir / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out_dir / "config.yaml")

    starting_pool = pd.read_csv(config.starting_population_path)["smiles"].astype(str).tolist()

    predictor = ADMETPredictor(cache_path=config.objective.cache_path)
    normalizer = DrugBankPercentileNormalizer.from_admet_ai_drugbank()
    predictor_reference_index = ReferenceFingerprintIndex.from_csv(
        config.reference_set_path,
        cache_path=config.applicability_domain.fingerprint_cache_path,
        radius=config.applicability_domain.fingerprint_radius,
        n_bits=config.applicability_domain.fingerprint_bits,
    )
    starting_population_index = ReferenceFingerprintIndex.from_csv(
        config.starting_population_path,
        radius=config.applicability_domain.fingerprint_radius,
        n_bits=config.applicability_domain.fingerprint_bits,
    )
    scorers = IndependentScorerSet(
        variants=config.independent_scorers.variants,
        cache_path=config.independent_scorers.cache_path,
    )

    # Arm R's composite distribution is the scale every other arm's percentile
    # is expressed on, so it is loaded before any arm runs.
    arm_r_scores = None
    if config.arms_of_kind("reference"):
        arm_r_scores = _load_arm_r(config)["objective_score"].to_numpy(dtype=float)

    chash = config_hash(config)
    summaries: list[dict] = []
    details: list[pd.DataFrame] = []
    divergence: list[dict] = []
    decoupling: list[dict] = []
    curves: list[pd.DataFrame] = []
    total_evaluated = 0

    with run_log(out_dir / "run_log.jsonl", chash, config.name) as state:
        for cell in arm_plan(config):
            result = _run_one(
                config,
                cell["label"],
                cell["arm"],
                cell["seed"],
                cell["theta"],
                starting_pool,
                predictor,
                normalizer,
                predictor_reference_index,
                starting_population_index,
                scorers,
                arm_r_scores,
            )
            suffix = "" if cell["seed"] is None else f"_seed{cell['seed']}"
            stem = f"{cell['label']}{suffix}"
            result["trajectory"].to_parquet(raw_dir / f"{stem}.parquet", index=False)
            if result["elites"] is not None:
                result["elites"].to_parquet(raw_dir / f"{stem}_elites.parquet", index=False)

            summaries.append(result["summary"])
            details.append(result["detail"])
            divergence.extend(result["divergence"])
            decoupling.extend(result["decoupling"])
            curves.extend(result["curves"])
            total_evaluated += len(result["trajectory"])

        divergence.extend(
            _starting_population_divergence(config, starting_pool, predictor, normalizer, scorers)
        )

        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(tables_dir / "summary_per_seed.csv", index=False)

        label_columns = ["arm", "arm_kind", "seed", "theta"]
        numeric_cols = [c for c in summary_df.columns if c not in label_columns]
        grouped = summary_df.groupby(["arm", "arm_kind", "theta"], dropna=False)[numeric_cols]
        summary_by_arm = pd.concat(
            [grouped.mean().add_suffix("_mean"), grouped.std().add_suffix("_std")], axis=1
        )
        # A seed-invariant arm has structurally zero spread, not missing spread
        # (brief SS3.5 wants a mean and a spread for every reported number).
        summary_by_arm = summary_by_arm.reset_index()
        single = summary_by_arm["n_seeds_mean"] == 1
        for column in [c for c in summary_by_arm.columns if c.endswith("_std")]:
            summary_by_arm.loc[single & summary_by_arm[column].isna(), column] = 0.0
        summary_by_arm.to_csv(tables_dir / "summary_by_arm.csv", index=False)

        detail_df = pd.concat(details, ignore_index=True)
        detail_df.to_csv(tables_dir / "topk_detail.csv", index=False)

        pd.DataFrame(divergence).to_csv(tables_dir / "divergence_long.csv", index=False)

        if decoupling:
            pd.DataFrame(decoupling).to_csv(tables_dir / "decoupling_per_seed.csv", index=False)
        if curves:
            pd.concat(curves, ignore_index=True).to_csv(tables_dir / "decoupling_curves.csv", index=False)

        # brief SS5.1: exploit characterization over the optimized arms only.
        optimized = detail_df[detail_df["arm_kind"] == "optimize"]
        if not optimized.empty:
            clusters = []
            for (arm_label,), group in optimized.groupby(["arm"]):
                for grouping in ("murcko_scaffold", "dominant_component"):
                    table = scaffold_clusters(group, grouping)
                    table.insert(0, "arm", arm_label)
                    table.insert(1, "grouping", grouping)
                    table = table.rename(columns={grouping: "cluster"})
                    clusters.append(table)
            pd.concat(clusters, ignore_index=True).to_csv(tables_dir / "scaffold_clusters.csv", index=False)

            contribution_columns = [c for c in optimized.columns if c.endswith(("_log_share", "_log_contribution", "_leave_one_out_delta"))]
            optimized.groupby(["arm", "arm_kind"])[contribution_columns].agg(["mean", "std"]).to_csv(
                tables_dir / "component_contributions.csv"
            )

        # brief SS4.5 output layout: the canonical calibration table is committed
        # under calibration/; copy it in so results/ is self-contained.
        frozen_table = Path(__file__).resolve().parents[2] / "calibration" / "battery_calibration.csv"
        if frozen_table.exists():
            shutil.copy(frozen_table, tables_dir / "battery_calibration.csv")

        state["molecules_evaluated"] = total_evaluated
        state["ensemble_variance"] = (
            "available" if predictor.ensemble_variance_available else "unavailable"
        )

    return out_dir


def dry_run(config_path: str | Path, probe_size: int = 200) -> dict:
    """Count planned evaluations and estimate wall time on *this* machine.

    Throughput varies by almost an order of magnitude across machines (this
    project has measured ~56 mol/s on a 2-vCPU container and ~390 mol/s on a
    14-core laptop), so the only useful ETA is a measured one. Lets the full
    budget be sized before committing hours to it.
    """
    import time

    config = load_config(config_path)
    cells = arm_plan(config)

    per_cell = {}
    for cell in cells:
        kind = cell["arm"].kind
        if kind == "optimize":
            population, generations = config.ga.population_size, config.ga.n_generations
            per_cell[cell["label"], cell["seed"]] = population + generations * (
                population - config.ga.n_elite
            )
        elif kind == "sample":
            per_cell[cell["label"], cell["seed"]] = config.k_top
        else:
            per_cell[cell["label"], cell["seed"]] = 0  # arm R is loaded, not scored

    total = sum(per_cell.values())

    starting_pool = pd.read_csv(config.starting_population_path)["smiles"].astype(str).tolist()
    probe = starting_pool[:probe_size]
    predictor = ADMETPredictor(cache_path=None)  # uncached, so the probe measures real inference
    started = time.monotonic()
    predictor.predict(probe)
    elapsed = time.monotonic() - started
    rate = len(probe) / elapsed if elapsed > 0 else float("nan")

    return {
        "config": str(config_path),
        "cells": len(cells),
        "evaluations_before_cache_dedup": total,
        "measured_mol_per_s": round(rate, 1),
        "estimated_hours_upper_bound": round(total / rate / 3600, 2) if rate else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an rhmoo experiment from a config file.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count planned evaluations and estimate wall time on this machine, then exit",
    )
    args = parser.parse_args()

    if args.dry_run:
        import json

        print(json.dumps(dry_run(args.config), indent=2))
        return

    out_dir = run_experiment(args.config, args.results_root)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

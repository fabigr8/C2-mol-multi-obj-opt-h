"""Experiment runner (PLAN Phase 9): orchestrates arms x seeds (x theta for arm
B) using the shared objective/constraint/independent-scorer/metrics modules,
and writes one raw parquet of every evaluated molecule plus summary tables
under `results/`.

Design decisions not fully pinned down by the config schema:
- Arm D (control) draws a `k_top`-sized random sample per seed and reports
  metrics on the whole sample (no GA, so there is nothing to rank/select).
- Arms A/B/C run the full GA per seed; the top `k_top` of the *final*
  population by fitness is the metrics set (brief SS4).
- Arm C fixes its AD-constraint theta at the middle of
  `applicability_domain.theta_values` (the sweep is arm B's H4 trade-off
  curve; arm C only needs one setting alongside the SA constraint).
- The H3 "starting population" divergence baseline (brief SS4) is computed
  once per experiment (not per arm/seed) over the full starting population,
  since it does not depend on the arm or seed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from rhmoo.applicability_domain import ReferenceFingerprintIndex, soft_penalty_above_threshold_is_bad
from rhmoo.config import ArmConfig, ExperimentConfig, config_hash, load_config
from rhmoo.ga import SelfiesGA
from rhmoo.independent_scorer import IndependentScorer
from rhmoo.metrics import (
    component_breakdown,
    implausibility_battery,
    independent_divergence,
    nearest_neighbor_similarity,
    physchem_panel,
    sa_scores,
    select_top_k,
    structural_alerts,
)
from rhmoo.normalization import DrugBankPercentileNormalizer
from rhmoo.objective import aggregate, normalize_components
from rhmoo.predictors import ADMETPredictor
from rhmoo.run_log import run_log
from rhmoo.synthesizability import sa_penalty

# Objective-component -> independent-scorer column (brief SS4, H3): only the
# two endpoints the in-repo independent scorer covers (PLAN Phase 6).
INDEPENDENT_PAIRS = {"hERG": "hERG_independent", "Solubility_AqSolDB": "Solubility_AqSolDB_independent"}


class ObjectivePipeline:
    """Composite objective (+ optional AD/SA constraints) as a GA `fitness_fn`.

    Returns one row per input SMILES with `fitness`, `valid`, every raw and
    normalized objective component, and (if enabled) the AD similarity / SA
    score used for the constraint penalty -- all propagated into the GA's
    full trajectory log (brief SS3.2/SS8: every evaluated molecule is data).
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
        result["objective_score"] = composite
        result["valid"] = raw["valid"]
        fitness = composite.to_numpy(dtype=float)

        if self.reference_index is not None:
            result["ad_max_tanimoto"] = self.reference_index.max_tanimoto(smiles)
            fitness = fitness * soft_penalty_above_threshold_is_bad(result["ad_max_tanimoto"].to_numpy(), self.theta)

        if self.use_synth_constraint:
            result["sa_score"] = sa_scores(smiles)
            fitness = fitness * sa_penalty(result["sa_score"].to_numpy(), self.config.synthesizability.sa_score_threshold)

        # ADMET-AI / normalization failures (rare, brief SS8: log, don't silently
        # drop): score 0 rather than corrupt GA selection with a NaN fitness.
        result["fitness"] = np.nan_to_num(fitness, nan=0.0)
        return result


def _sample_population(pool: list[str], n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    n = min(n, len(pool))
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in idx]


def _arm_theta(config: ExperimentConfig, arm: ArmConfig) -> float | None:
    """Fixed AD theta for arm C (see module docstring); arm B is swept by the
    caller and passes its own theta explicitly.
    """
    if arm.name != "C":
        return None
    thetas = config.applicability_domain.theta_values
    return thetas[len(thetas) // 2]


def _run_one(
    config: ExperimentConfig,
    label: str,
    arm: ArmConfig,
    seed: int,
    theta: float | None,
    starting_pool: list[str],
    predictor: ADMETPredictor,
    normalizer: DrugBankPercentileNormalizer,
    predictor_reference_index: ReferenceFingerprintIndex,
    starting_population_index: ReferenceFingerprintIndex,
    independent_scorer: IndependentScorer,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Runs one (arm, seed[, theta]) combination end to end. Returns the full
    raw trajectory/sample DataFrame, a one-row summary dict, and a per-molecule
    top-k detail DataFrame (figures/RESULTS.md read the latter two).
    """
    if arm.name == "D":
        sample = _sample_population(starting_pool, config.k_top, seed)
        pipeline = ObjectivePipeline(config, predictor, normalizer)
        top_raw = pipeline(sample)
        top_raw["generation"] = 0
        trajectory = top_raw
        top_smiles = sample
    else:
        reference_index = predictor_reference_index if arm.use_ad_constraint else None
        pipeline = ObjectivePipeline(
            config,
            predictor,
            normalizer,
            reference_index=reference_index,
            theta=theta,
            use_synth_constraint=arm.use_synth_constraint,
        )
        initial_population = _sample_population(starting_pool, config.ga.population_size, seed)
        ga = SelfiesGA(config.ga, pipeline, seed=seed)
        final_population, final_fitness, trajectory = ga.run(initial_population)
        top_smiles, _ = select_top_k(final_population, final_fitness, config.k_top)
        # Recompute the top-k rows explicitly (cheap: hits the persistent
        # predictor cache) rather than filtering the trajectory, since a
        # SMILES can recur across generations and top-k must be unambiguous.
        top_raw = pipeline(top_smiles)

    panel = physchem_panel(top_smiles)
    alerts = structural_alerts(top_smiles)
    sa = sa_scores(top_smiles)
    battery = implausibility_battery(panel, alerts, sa, config.implausibility_battery)
    ad_vs_reference = nearest_neighbor_similarity(top_smiles, predictor_reference_index)
    ad_vs_starting = nearest_neighbor_similarity(top_smiles, starting_population_index)
    independent_top = independent_scorer.score(top_smiles)
    divergence = independent_divergence(top_raw, independent_top, INDEPENDENT_PAIRS)
    components = component_breakdown(top_raw, [c.name for c in config.objective.components])

    summary = {
        "arm": label,
        "seed": seed,
        "theta": theta,
        "n_molecules_evaluated": len(trajectory),
        "objective_score_mean": top_raw["objective_score"].mean(),
        "objective_score_std": top_raw["objective_score"].std(),
        "fitness_mean": top_raw["fitness"].mean(),
        "implausibility_rate": battery["implausible"].mean(),
        "ad_similarity_vs_reference_mean": float(np.mean(ad_vs_reference)),
        "ad_similarity_vs_starting_mean": float(np.mean(ad_vs_starting)),
        "sa_score_mean": float(np.nanmean(sa)),
    }
    for _, row in components.iterrows():
        summary[f"component_{row['component']}_mean"] = row["mean"]
    for _, row in divergence.iterrows():
        summary[f"{row['property']}_spearman_r_optimized"] = row["spearman_r"]
        summary[f"{row['property']}_mean_abs_disagreement_optimized"] = row["mean_abs_disagreement"]

    detail = top_raw.copy()
    detail["arm"] = label
    detail["theta"] = theta
    detail["seed"] = seed
    detail["ad_similarity_vs_reference"] = ad_vs_reference
    detail["ad_similarity_vs_starting"] = ad_vs_starting
    detail["sa_score"] = sa
    detail = pd.concat([detail.reset_index(drop=True), panel, alerts, battery, independent_top], axis=1)

    return trajectory, summary, detail


def _starting_population_divergence(
    config: ExperimentConfig,
    starting_pool: list[str],
    predictor: ADMETPredictor,
    normalizer: DrugBankPercentileNormalizer,
    independent_scorer: IndependentScorer,
) -> pd.DataFrame:
    """H3 baseline (brief SS4): objective-vs-independent divergence computed
    once over the starting population, for comparison against each arm/seed's
    optimized-top-k divergence. `starting_population_baseline_size` (config,
    None = use all) bounds the ADMET-AI inference cost of this one-off call.
    """
    sample_size = config.starting_population_baseline_size
    pool = starting_pool if sample_size is None else _sample_population(starting_pool, sample_size, seed=0)
    pipeline = ObjectivePipeline(config, predictor, normalizer)
    scored = pipeline(pool)
    independent = independent_scorer.score(pool)
    return independent_divergence(scored, independent, INDEPENDENT_PAIRS)


def run_experiment(config_path: str | Path, results_root: str | Path | None = None) -> Path:
    config = load_config(config_path)
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
    independent_scorer = IndependentScorer()

    chash = config_hash(config)
    all_summaries: list[dict] = []
    all_details: list[pd.DataFrame] = []
    total_evaluated = 0

    with run_log(out_dir / "run_log.jsonl", chash, config.name) as state:
        for arm in config.arms:
            theta_sweep = (
                config.applicability_domain.theta_values if arm.name == "B" else [_arm_theta(config, arm)]
            )
            for theta in theta_sweep:
                label = arm.name if arm.name != "B" else f"B_theta{theta:.2f}"
                for seed in config.seeds:
                    trajectory, summary, detail = _run_one(
                        config,
                        label,
                        arm,
                        seed,
                        theta,
                        starting_pool,
                        predictor,
                        normalizer,
                        predictor_reference_index,
                        starting_population_index,
                        independent_scorer,
                    )
                    trajectory.to_parquet(raw_dir / f"{label}_seed{seed}.parquet", index=False)
                    all_summaries.append(summary)
                    all_details.append(detail)
                    total_evaluated += len(trajectory)

        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(tables_dir / "summary_per_seed.csv", index=False)

        numeric_cols = [c for c in summary_df.columns if c not in ("arm", "seed", "theta")]
        grouped = summary_df.groupby(["arm", "theta"], dropna=False)[numeric_cols]
        summary_by_arm = pd.concat([grouped.mean().add_suffix("_mean"), grouped.std().add_suffix("_std")], axis=1)
        summary_by_arm.reset_index().to_csv(tables_dir / "summary_by_arm.csv", index=False)

        pd.concat(all_details, ignore_index=True).to_csv(tables_dir / "topk_detail.csv", index=False)

        baseline_divergence = _starting_population_divergence(
            config, starting_pool, predictor, normalizer, independent_scorer
        )
        baseline_divergence.to_csv(tables_dir / "starting_population_divergence.csv", index=False)

        state["molecules_evaluated"] = total_evaluated

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an rhmoo experiment from a config file.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root")
    args = parser.parse_args()
    out_dir = run_experiment(args.config, args.results_root)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

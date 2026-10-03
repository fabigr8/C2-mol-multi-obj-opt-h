"""Scores arm R -- the approved-drug reference set (brief SS3.3, SS3.4) -- and
freezes it as a committed artifact.

Arm R is the DrugBank set shipped with ADMET-AI: 2,845 approved drugs, scored
with the composite objective and every independent scorer but never optimized.
It does two jobs:

1. It gives the composite score an interpretable scale, so the headline becomes
   "arm A's molecules score higher on this objective than any approved drug
   does" rather than an absolute composite value (brief SS4.1).
2. It is the calibration set for the implausibility battery (brief SS4.5),
   which must be calibrated *before* any arm A molecule is generated.

Because of (2) this runs before the experiment, and because the arm R reported
in RESULTS.md must be the same set the thresholds were calibrated on, the
runner loads this artifact and verifies its hash rather than recomputing it.

The predictions are recomputed from the live model rather than lifted from the
precomputed columns in ADMET-AI's shipped CSV: the ~50s saved is not worth
weakening the provenance claim that every number in this repo came out of the
pinned pipeline. `tests/test_arm_r.py` asserts the two agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from rhmoo.applicability_domain import ReferenceFingerprintIndex
from rhmoo.config import config_hash, load_config
from rhmoo.independent_scorer import IndependentScorerSet
from rhmoo.normalization import DrugBankPercentileNormalizer
from rhmoo.predictors import ADMETPredictor
from rhmoo.scoring import score_molecule_set

REPO_ROOT = Path(__file__).resolve().parents[1]


def approved_drug_path() -> Path:
    from admet_ai.constants import DEFAULT_DRUGBANK_PATH

    return Path(DEFAULT_DRUGBANK_PATH)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N (for testing)")
    args = parser.parse_args()

    config = load_config(args.config)
    source = approved_drug_path()
    approved = pd.read_csv(source)
    smiles = approved["smiles"].astype(str).tolist()
    if args.limit:
        smiles = smiles[: args.limit]

    predictor = ADMETPredictor(cache_path=config.objective.cache_path)
    normalizer = DrugBankPercentileNormalizer.from_admet_ai_drugbank()
    scorers = IndependentScorerSet(
        variants=config.independent_scorers.variants,
        cache_path=config.independent_scorers.cache_path,
    )
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

    # No AD/SA penalty: arm R is scored, never optimized, so fitness == objective.
    from rhmoo.runner import ObjectivePipeline

    pipeline = ObjectivePipeline(config, predictor, normalizer)
    detail = score_molecule_set(
        smiles,
        pipeline,
        scorers,
        predictor_reference_index=predictor_reference_index,
        starting_population_index=starting_population_index,
    )

    out_path = REPO_ROOT / config.reference_arm.artifact_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    detail.to_parquet(out_path, index=False)

    manifest = {
        "description": "Arm R: approved drugs (DrugBank set shipped with ADMET-AI), scored not optimized",
        "source_file": str(source),
        "source_sha256": sha256_file(source),
        "n_molecules": int(len(detail)),
        "n_source_rows": int(len(approved)),
        "columns": list(detail.columns),
        "independent_scorer_variants": list(config.independent_scorers.variants),
        "admet_ai_ensemble_variance_available": predictor.ensemble_variance_available,
        "config": args.config,
        "config_hash": config_hash(config),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": sha256_file(out_path),
    }
    manifest_path = out_path.parent / "arm_r_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"wrote {out_path} ({len(detail)} molecules)")
    print(f"wrote {manifest_path}")
    print(f"objective_score: mean={detail['objective_score'].mean():.4f} max={detail['objective_score'].max():.4f}")


if __name__ == "__main__":
    main()

"""Trains the independent scorers' frozen LightGBM models (brief SS4.2):
RDKit descriptors + MACCS keys for hERG (binary classification) and
Solubility_AqSolDB (regression). Output artifacts are committed to the repo and
never retrained during an experiment run.

Three variants of one recipe, following Renz et al. (2019)'s
optimization/model-control/data-control design:

| variant             | trained on              | evaluated on | disjoint from ADMET-AI's training data? |
|---------------------|-------------------------|--------------|------------------------------------------|
| `model_control`     | full TDC `train_val`    | TDC `test`   | no -- same data, different architecture  |
| `data_control_half` | half B of `train_val`   | TDC `test`   | no -- see caveat                         |
| `data_control_test` | TDC `test`              | `train_val`  | yes                                      |

Caveat (brief SS4.2 is internally inconsistent here, see docs/DEVIATIONS.md):
ADMET-AI is a fixed pretrained artifact fit on the *whole* `train_val` split,
and brief SS2 forbids retraining the objective's predictors. So no half of
`train_val` can be disjoint from the objective's training data.
`data_control_half` is Renz-shaped but is a data control only relative to a
hypothetical half-trained objective; `data_control_test` is the only variant
that is genuinely disjoint, at the cost of a much smaller training set.

The competence gate below is declared here, in version control, *before* the
variants are trained. Both hERG data controls are trained on a few hundred
molecules and may fail it. Declaring the bar in advance is what makes
"reported but excluded from the headline" a pre-registered decision rather
than a post-hoc one; every variant is reported either way (brief SS9).
"""

from __future__ import annotations

import argparse
import json
from importlib import metadata
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import rdBase
from sklearn.metrics import mean_absolute_error, roc_auc_score

from rhmoo.independent_scorer import (
    FEATURE_COLUMNS,
    HALF_SPLIT_SEED,
    VARIANTS,
    featurize,
    half_mask,
    model_dir,
)

SEED = 0
N_BOOST_ROUND = 300
LGBM_PARAMS = {"num_leaves": 31, "learning_rate": 0.05, "verbosity": -1, "seed": SEED}

#: Pre-declared minimum competence for a scorer to be eligible as the headline
#: H3 oracle. Not a filter -- every variant is trained and reported.
COMPETENCE_GATE = {
    "hERG": {"metric": "roc_auc", "min": 0.65},
    "Solubility_AqSolDB": {"metric": "mae", "max": 1.2},
}

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "tdc" / "admet_group"

#: TDC directory name and prediction task per objective endpoint.
ENDPOINTS = {
    "hERG": {"tdc": "herg", "objective": "binary", "metric": "roc_auc"},
    "Solubility_AqSolDB": {"tdc": "solubility_aqsoldb", "objective": "regression", "metric": "mae"},
}

#: Which TDC split each variant trains and evaluates on, and which half of the
#: training split it keeps (None = all of it).
VARIANT_SPECS = {
    "model_control": {"train": "train_val", "eval": "test", "half": None},
    "data_control_half": {"train": "train_val", "eval": "test", "half": "B"},
    "data_control_test": {"train": "test", "eval": "train_val", "half": None},
}

_METRIC_FNS = {"roc_auc": roc_auc_score, "mae": mean_absolute_error}
_HIGHER_IS_BETTER = {"roc_auc": True, "mae": False}


def _load_split(tdc_name: str, split: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / tdc_name / f"{split}.csv")


def _passes_gate(endpoint: str, score: float) -> bool:
    gate = COMPETENCE_GATE[endpoint]
    if "min" in gate:
        return score >= gate["min"]
    return score <= gate["max"]


def _train_and_evaluate(endpoint: str, variant: str) -> dict:
    spec = VARIANT_SPECS[variant]
    endpoint_spec = ENDPOINTS[endpoint]
    metric = endpoint_spec["metric"]

    train = _load_split(endpoint_spec["tdc"], spec["train"])
    evaluation = _load_split(endpoint_spec["tdc"], spec["eval"])

    if spec["half"] is not None:
        keep = half_mask(train["Drug"].tolist(), spec["half"], HALF_SPLIT_SEED)
        train = train[keep].reset_index(drop=True)

    X_train = featurize(train["Drug"].tolist())
    X_eval = featurize(evaluation["Drug"].tolist())

    # featurize() logs unparseable SMILES (brief SS9: log, don't silently drop);
    # drop their all-NaN rows here rather than training on a label with no features.
    train_mask = X_train.notna().any(axis=1)
    eval_mask = X_eval.notna().any(axis=1)

    booster = lgb.train(
        {**LGBM_PARAMS, "objective": endpoint_spec["objective"]},
        lgb.Dataset(X_train[train_mask], label=train["Y"].to_numpy()[train_mask]),
        num_boost_round=N_BOOST_ROUND,
    )

    predictions = booster.predict(X_eval[eval_mask])
    score = float(_METRIC_FNS[metric](evaluation["Y"].to_numpy()[eval_mask], predictions))

    return {
        "booster": booster,
        "metrics": {
            "n_train": int(train_mask.sum()),
            "n_eval": int(eval_mask.sum()),
            "train_split": spec["train"] + (f" (half {spec['half']})" if spec["half"] else ""),
            "eval_split": spec["eval"],
            "metric": metric,
            "held_out_score": score,
            "higher_is_better": _HIGHER_IS_BETTER[metric],
            "competence_gate": COMPETENCE_GATE[endpoint],
            "passes_competence_gate": _passes_gate(endpoint, score),
        },
    }


def train_variant(variant: str) -> dict:
    directory = model_dir(variant)
    directory.mkdir(parents=True, exist_ok=True)

    results = {}
    for endpoint in ENDPOINTS:
        result = _train_and_evaluate(endpoint, variant)
        filename = {"hERG": "herg_lightgbm.txt", "Solubility_AqSolDB": "solubility_aqsoldb_lightgbm.txt"}[
            endpoint
        ]
        result["booster"].save_model(str(directory / filename))
        results[endpoint] = result["metrics"]

    spec = VARIANT_SPECS[variant]
    metadata_out = {
        "variant": variant,
        "feature_columns": list(FEATURE_COLUMNS),
        "lightgbm_version": lgb.__version__,
        "rdkit_version": rdBase.rdkitVersion,
        "scikit_learn_version": metadata.version("scikit-learn"),
        "params": {**LGBM_PARAMS, "num_boost_round": N_BOOST_ROUND},
        "seed": SEED,
        "half_split": (
            None
            if spec["half"] is None
            else {
                "half": spec["half"],
                "seed": HALF_SPLIT_SEED,
                "rule": "sha256(f'{seed}:{canonical_smiles}').digest()[0] % 2 -> 0=A, 1=B",
            }
        ),
        "disjoint_from_admet_ai_training_data": spec["train"] == "test",
        "held_out_metrics": results,
        "training_data": {
            endpoint: (
                f"data/raw/tdc/admet_group/{ENDPOINTS[endpoint]['tdc']}/{spec['train']}.csv"
                f"{' (half ' + spec['half'] + ')' if spec['half'] else ''}"
                f" (TDC ADMET_Group official split); evaluated on {spec['eval']}.csv"
            )
            for endpoint in ENDPOINTS
        },
    }
    (directory / "metadata.json").write_text(json.dumps(metadata_out, indent=2) + "\n")
    return metadata_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANT_SPECS), action="append", help="default: all")
    args = parser.parse_args()

    for variant in args.variant or list(VARIANTS):
        written = train_variant(variant)
        print(f"=== {variant} ===")
        print(json.dumps(written["held_out_metrics"], indent=2))


if __name__ == "__main__":
    main()

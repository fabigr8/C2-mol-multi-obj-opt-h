"""Trains the independent scorer's frozen LightGBM models (brief SS4/SS6,
Phase 6): RDKit descriptors + MACCS keys, fit on the official TDC ADMET_Group
train_val split for hERG (binary classification) and Solubility_AqSolDB
(regression), evaluated on the official held-out test split so the scorer's
own competence is documented rather than assumed. Output artifacts are
committed to the repo and never retrained during an experiment run.
"""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from rdkit import rdBase
from sklearn.metrics import mean_absolute_error, roc_auc_score

from rhmoo.independent_scorer import FEATURE_COLUMNS, featurize

SEED = 0
N_BOOST_ROUND = 300
LGBM_PARAMS = {"num_leaves": 31, "learning_rate": 0.05, "verbosity": -1, "seed": SEED}

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "tdc" / "admet_group"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "independent_scorer"


def _load_split(endpoint: str, split: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / endpoint / f"{split}.csv")


def _train_and_evaluate(endpoint: str, objective: str, metric_fn, higher_is_better: bool) -> dict:
    train_val = _load_split(endpoint, "train_val")
    test = _load_split(endpoint, "test")

    X_train = featurize(train_val["Drug"].tolist())
    X_test = featurize(test["Drug"].tolist())

    # featurize() logs unparseable SMILES (brief §8); drop their all-NaN rows
    # here rather than training on a label with no features.
    train_mask = X_train.notna().any(axis=1)
    test_mask = X_test.notna().any(axis=1)

    booster = lgb.train(
        {**LGBM_PARAMS, "objective": objective},
        lgb.Dataset(X_train[train_mask], label=train_val["Y"].to_numpy()[train_mask]),
        num_boost_round=N_BOOST_ROUND,
    )

    predictions = booster.predict(X_test[test_mask])
    score = metric_fn(test["Y"].to_numpy()[test_mask], predictions)
    return {
        "booster": booster,
        "metrics": {
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "held_out_score": float(score),
            "higher_is_better": higher_is_better,
        },
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    herg = _train_and_evaluate(
        "herg", objective="binary", metric_fn=roc_auc_score, higher_is_better=True
    )
    herg["booster"].save_model(str(MODEL_DIR / "herg_lightgbm.txt"))
    herg["metrics"]["metric"] = "roc_auc"

    solubility = _train_and_evaluate(
        "solubility_aqsoldb", objective="regression", metric_fn=mean_absolute_error, higher_is_better=False
    )
    solubility["booster"].save_model(str(MODEL_DIR / "solubility_aqsoldb_lightgbm.txt"))
    solubility["metrics"]["metric"] = "mae"

    metadata_out = {
        "feature_columns": list(FEATURE_COLUMNS),
        "lightgbm_version": lgb.__version__,
        "rdkit_version": rdBase.rdkitVersion,
        "scikit_learn_version": metadata.version("scikit-learn"),
        "params": {**LGBM_PARAMS, "num_boost_round": N_BOOST_ROUND},
        "seed": SEED,
        "held_out_metrics": {"hERG": herg["metrics"], "Solubility_AqSolDB": solubility["metrics"]},
        "training_data": {
            "hERG": "data/raw/tdc/admet_group/herg/{train_val,test}.csv (TDC ADMET_Group official split)",
            "Solubility_AqSolDB": (
                "data/raw/tdc/admet_group/solubility_aqsoldb/{train_val,test}.csv "
                "(TDC ADMET_Group official split)"
            ),
        },
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata_out, indent=2))
    print(json.dumps(metadata_out["held_out_metrics"], indent=2))


if __name__ == "__main__":
    main()

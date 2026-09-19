"""Independent scorer (brief SS4, H3 -- the load-bearing evidence for the whole
project): a deliberately distinct model (RDKit physchem descriptors + MACCS
keys, LightGBM) for two of the objective's ADMET-AI-sourced endpoints (hERG,
aqueous solubility). Used to check whether the optimizer exploits the
objective's own predictors rather than the properties they estimate.

Frozen artifacts under models/independent_scorer/ are committed and never
retrained during an experiment run -- see scripts/train_independent_scorer.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, Descriptors, MACCSkeys

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "independent_scorer"
N_MACCS_BITS = 167

logger = logging.getLogger(__name__)

_DESCRIPTOR_NAMES = tuple(name for name, _ in Descriptors._descList)
FEATURE_COLUMNS = tuple(_DESCRIPTOR_NAMES) + tuple(f"maccs_{i}" for i in range(N_MACCS_BITS))


def featurize(smiles_list: list[str]) -> pd.DataFrame:
    """One row per input SMILES (same order); rows for invalid SMILES are all-NaN."""
    rows = []
    n_invalid = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_invalid += 1
            rows.append({col: np.nan for col in FEATURE_COLUMNS})
            continue
        descriptors = Descriptors.CalcMolDescriptors(mol)
        maccs_bits = np.zeros((N_MACCS_BITS,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), maccs_bits)
        rows.append({**descriptors, **{f"maccs_{i}": int(b) for i, b in enumerate(maccs_bits)}})
    if n_invalid:
        logger.warning("independent scorer: %d unparseable SMILES out of %d", n_invalid, len(smiles_list))
    return pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))


class IndependentScorer:
    """Loads the frozen hERG classifier + solubility regressor and scores SMILES."""

    def __init__(self, model_dir: str | Path = MODEL_DIR):
        model_dir = Path(model_dir)
        self.metadata = json.loads((model_dir / "metadata.json").read_text())
        self._herg_model = lgb.Booster(model_file=str(model_dir / "herg_lightgbm.txt"))
        self._solubility_model = lgb.Booster(model_file=str(model_dir / "solubility_aqsoldb_lightgbm.txt"))

    def score(self, smiles_list: list[str]) -> pd.DataFrame:
        features = featurize(smiles_list)
        herg_proba = self._herg_model.predict(features)
        solubility = self._solubility_model.predict(features)
        valid = features.notna().any(axis=1)
        return pd.DataFrame(
            {
                "hERG_independent": np.where(valid, herg_proba, np.nan),
                "Solubility_AqSolDB_independent": np.where(valid, solubility, np.nan),
            }
        )

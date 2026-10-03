"""Independent scorers (brief SS4.2, H3 -- the load-bearing evidence for the
whole project): deliberately distinct models (RDKit physchem descriptors +
MACCS keys, LightGBM) for two of the objective's ADMET-AI-sourced endpoints
(hERG, aqueous solubility). Used to check whether the optimizer exploits the
objective's own predictors rather than the properties they estimate.

Following Renz et al. (2019), the brief requires more than one kind of control,
and this module carries three *variants* of the same recipe:

- `model_control` -- trained on the full TDC `train_val` split, i.e. the same
  data ADMET-AI was fit on. Differs in architecture only, so any bias in the
  dataset is invisible to it.
- `data_control_half` -- the same recipe on a deterministic disjoint half of
  `train_val`. Renz-shaped, but see the caveat below.
- `data_control_test` -- the same recipe trained on the TDC `test` split, which
  ADMET-AI was *not* fit on. The only variant genuinely disjoint from the
  objective predictor's training data, at the cost of a much smaller n.

Caveat that must not be lost (docs/LIMITATIONS.md): ADMET-AI is a fixed
pretrained artifact fit on the *whole* `train_val` split, so no half of
`train_val` is disjoint from the objective's training data.
`data_control_half` is a data control only relative to a hypothetical
half-trained objective, and must never be described as disjoint from
ADMET-AI's training set.

Scores from different variants are reported separately and never averaged
(brief SS4.2: "if they diverge from each other, say so rather than averaging
them").

Frozen artifacts under models/independent_scorer/<variant>/ are committed and
never retrained during an experiment run -- see
scripts/train_independent_scorer.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, Descriptors, MACCSkeys

from rhmoo.cache import SmilesKeyedCache

MODEL_ROOT = Path(__file__).resolve().parents[2] / "models" / "independent_scorer"
N_MACCS_BITS = 167

#: The endpoints an independent scorer covers. Deliberately 2 of the
#: objective's 5 components -- Caco2_Wang and CYP3A4_Veith are uncovered, so
#: exploitation routed through them is invisible to H3 (brief SS4.2).
INDEPENDENT_ENDPOINTS = ("hERG", "Solubility_AqSolDB")

#: Variant name -> model filename per endpoint. One recipe, three training sets.
_MODEL_FILENAMES = {
    "hERG": "herg_lightgbm.txt",
    "Solubility_AqSolDB": "solubility_aqsoldb_lightgbm.txt",
}

#: All variants the repo ships. Order is the reporting order.
VARIANTS = ("model_control", "data_control_half", "data_control_test")

#: Separator between endpoint and variant in a score column name. Two
#: underscores, because `Solubility_AqSolDB` already contains single ones.
_VARIANT_SEP = "__"

logger = logging.getLogger(__name__)

_DESCRIPTOR_NAMES = tuple(name for name, _ in Descriptors._descList)
FEATURE_COLUMNS = tuple(_DESCRIPTOR_NAMES) + tuple(f"maccs_{i}" for i in range(N_MACCS_BITS))


def model_dir(variant: str) -> Path:
    return MODEL_ROOT / variant


def independent_column(endpoint: str, variant: str) -> str:
    """Score column name for one (endpoint, variant) pair."""
    return f"{endpoint}_independent{_VARIANT_SEP}{variant}"


def split_independent_column(column: str) -> tuple[str, str]:
    """Inverse of `independent_column`."""
    head, _, variant = column.rpartition(_VARIANT_SEP)
    if not variant or not head.endswith("_independent"):
        raise ValueError(f"not an independent-score column: {column!r}")
    return head[: -len("_independent")], variant


def independent_pairs(variant: str) -> dict[str, str]:
    """Objective column -> independent column, for `metrics.independent_divergence`.

    Single source of truth: this previously existed as the same literal dict
    copied into runner.py, figures.py and report.py.
    """
    return {endpoint: independent_column(endpoint, variant) for endpoint in INDEPENDENT_ENDPOINTS}


#: Deterministic half-split salt for `data_control_half`. Recorded in the
#: variant's metadata.json and re-derivable by the overlap tests.
HALF_SPLIT_SEED = 0


def half_assignment(smiles: str, seed: int = HALF_SPLIT_SEED) -> str:
    """Assign one molecule to half "A" or "B", deterministically.

    Keyed on a hash of the *canonical* SMILES rather than on a shuffled row
    index: a hash assignment is stable under row reordering, duplicate rows and
    any change in dataset size, so the split is reproducible from the CSV alone
    with no stored index. Grouping on the canonical form also guarantees the
    same molecule never lands in both halves under two different spellings,
    which is exactly the leak the zero-overlap test exists to rule out.
    """
    from rhmoo.predictors import canonicalize

    canonical = canonicalize(smiles)
    key = canonical if canonical is not None else smiles
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return "A" if digest[0] % 2 == 0 else "B"


def half_mask(smiles_list: list[str], half: str, seed: int = HALF_SPLIT_SEED) -> np.ndarray:
    """Boolean mask selecting the molecules assigned to `half`."""
    return np.array([half_assignment(s, seed) == half for s in smiles_list], dtype=bool)


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
    """One variant: the frozen hERG classifier + solubility regressor."""

    def __init__(self, variant: str = "model_control", model_dir_override: str | Path | None = None):
        self.variant = variant
        directory = Path(model_dir_override) if model_dir_override else model_dir(variant)
        self.model_dir = directory
        self.metadata = json.loads((directory / "metadata.json").read_text())
        self._models = {
            endpoint: lgb.Booster(model_file=str(directory / filename))
            for endpoint, filename in _MODEL_FILENAMES.items()
        }

    @property
    def columns(self) -> list[str]:
        return [independent_column(endpoint, self.variant) for endpoint in INDEPENDENT_ENDPOINTS]

    def predict_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Score a pre-computed feature matrix. Separated from `score` so a set of
        variants can share one featurization pass -- featurization is ~500x
        slower than the boosters, so sharing it is what makes three variants
        cost about the same as one.
        """
        valid = features.notna().any(axis=1)
        out = {}
        for endpoint in INDEPENDENT_ENDPOINTS:
            predicted = self._models[endpoint].predict(features)
            out[independent_column(endpoint, self.variant)] = np.where(valid, predicted, np.nan)
        return pd.DataFrame(out, index=features.index)

    def score(self, smiles_list: list[str]) -> pd.DataFrame:
        return self.predict_features(featurize(smiles_list)).reset_index(drop=True)


class IndependentScorerSet:
    """Several variants scored together over a single shared featurization pass,
    with an optional persistent cache.

    Brief SS4.3 (per-generation divergence) calls per-generation rescoring
    "cheap", which holds only because featurization is shared and cached: the
    boosters run at ~45,000 mol/s but featurization at ~80 mol/s, so the
    featurize call is the entire cost and must not be repeated per variant.
    """

    def __init__(self, variants: list[str] | tuple[str, ...] = VARIANTS, cache_path: str | Path | None = None):
        self.scorers = [IndependentScorer(variant) for variant in variants]
        if not self.scorers:
            raise ValueError("IndependentScorerSet needs at least one variant")

        # A variant whose features differ is not "the same recipe on different
        # data", which is the entire premise of the model-vs-data control
        # comparison. Fail loudly rather than silently comparing apples.
        for scorer in self.scorers:
            declared = scorer.metadata.get("feature_columns")
            if declared != list(FEATURE_COLUMNS):
                raise ValueError(
                    f"variant {scorer.variant!r} declares a different feature set than the "
                    "current featurizer; retrain it or pin the matching rhmoo version"
                )

        self.variants = [scorer.variant for scorer in self.scorers]
        self.columns = [column for scorer in self.scorers for column in scorer.columns]
        self.cache = SmilesKeyedCache(
            columns=self.columns,
            path=cache_path,
            provenance={"producer": "independent_scorer", "variants": ",".join(self.variants)},
        )

    def score(self, smiles_list: list[str]) -> pd.DataFrame:
        """Wide frame, one row per input SMILES in order, one column per
        (endpoint, variant). Invalid SMILES yield NaN rather than being dropped.
        """
        to_score = self.cache.missing(smiles_list)
        if to_score:
            features = featurize(to_score)
            scored = pd.concat(
                [scorer.predict_features(features) for scorer in self.scorers], axis=1
            )
            scored.index = pd.Index(to_score, name="smiles")
            self.cache.add(scored)
            self.cache.save()
        return self.cache.lookup(smiles_list).reset_index(drop=True)

    def pairs(self, variant: str) -> dict[str, str]:
        return independent_pairs(variant)

    def metadata(self) -> dict[str, dict]:
        return {scorer.variant: scorer.metadata for scorer in self.scorers}

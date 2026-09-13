"""ADMET-AI + QED prediction, with a persistent cache keyed by canonical SMILES
(brief SS6: the GA will re-evaluate duplicates constantly; this is the main
lever on runtime cost).

`ADMETModel.predict()` silently drops invalid SMILES (verified in Phase 0:
4 in, 2 out, no exception) and can reorder/dedupe its output, so results are
always realigned onto the requested SMILES list rather than trusted as-is.
"""

from __future__ import annotations

import contextlib
import io
import logging
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED

ADMET_COMPONENTS = ("Solubility_AqSolDB", "hERG", "CYP3A4_Veith", "Caco2_Wang")
OUTPUT_COLUMNS = (*ADMET_COMPONENTS, "QED")

logger = logging.getLogger(__name__)


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def compute_qed(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot compute QED for invalid SMILES: {smiles!r}")
    return QED.qed(mol)


@contextlib.contextmanager
def _quiet_admet_ai():
    """Lightning hardcodes a stdout progress bar; suppress it and its loggers."""
    warnings.filterwarnings("ignore")
    for name in ("lightning", "lightning.pytorch", "lightning.pytorch.utilities"):
        logging.getLogger(name).setLevel(logging.ERROR)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


class PredictionCache:
    """In-memory cache of predictor outputs keyed by canonical SMILES, optionally
    persisted to a parquet file across runs.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        if self.path is not None and self.path.exists():
            self._df = pd.read_parquet(self.path).set_index("smiles")
        else:
            self._df = pd.DataFrame(columns=list(OUTPUT_COLUMNS)).rename_axis("smiles")

    def __contains__(self, smiles: str) -> bool:
        return smiles in self._df.index

    def missing(self, canonical_smiles: Iterable[str]) -> list[str]:
        return [s for s in dict.fromkeys(canonical_smiles) if s not in self]

    def lookup(self, canonical_smiles: Iterable[str]) -> pd.DataFrame:
        return self._df.reindex(list(canonical_smiles))

    def add(self, rows: pd.DataFrame) -> None:
        rows = rows.rename_axis("smiles")
        new = rows[~rows.index.isin(self._df.index)]
        self._df = pd.concat([self._df, new])

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._df.reset_index().to_parquet(self.path, index=False)


class ADMETPredictor:
    """Batched ADMET-AI + RDKit-QED prediction with an optional persistent cache."""

    def __init__(self, cache_path: str | Path | None = None):
        self.cache = PredictionCache(cache_path)
        self._model = None

    def _load_model(self):
        if self._model is None:
            from admet_ai import ADMETModel

            with _quiet_admet_ai():
                self._model = ADMETModel(include_physchem=False, drugbank_path=None)
        return self._model

    def predict(self, smiles: list[str]) -> pd.DataFrame:
        canonical = [canonicalize(s) for s in smiles]
        n_invalid = sum(c is None for c in canonical)
        if n_invalid:
            logger.warning("dropped %d unparseable SMILES out of %d requested", n_invalid, len(smiles))

        valid_canonical = [c for c in canonical if c is not None]
        to_query = self.cache.missing(valid_canonical)

        if to_query:
            model = self._load_model()
            with _quiet_admet_ai():
                raw = model.predict(smiles=to_query)
            raw = raw.reindex(to_query)  # ADMETModel.predict can drop/reorder rows
            dropped_by_model = raw[list(ADMET_COMPONENTS)].isna().all(axis=1)
            if dropped_by_model.any():
                logger.warning(
                    "ADMET-AI dropped %d of %d queried SMILES", int(dropped_by_model.sum()), len(to_query)
                )
            rows = raw[list(ADMET_COMPONENTS)].copy()
            rows["QED"] = [compute_qed(s) for s in to_query]
            self.cache.add(rows)
            self.cache.save()

        result = self.cache.lookup(canonical).reset_index(drop=True)
        result["input_smiles"] = smiles
        result["canonical_smiles"] = canonical
        result["valid"] = [c is not None for c in canonical]
        return result

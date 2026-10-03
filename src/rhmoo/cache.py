"""Canonical-SMILES-keyed prediction cache, shared by the ADMET-AI predictor and
the independent scorers (brief SS7: "Cache ADMET-AI predictions keyed by
canonical SMILES -- the GA will re-evaluate duplicates constantly and this is
the main runtime cost. SS4.3 relies on this cache making per-generation
rescoring cheap.").

The independent scorers need the same behaviour for a different column set, so
this is the one implementation both use rather than two copies that drift.

A persisted cache carries a sidecar `<name>.meta.json` recording the schema
version, column list, and producing library versions. A mismatch means the
cached numbers were produced by a different pipeline, so the cache is rebuilt
loudly rather than silently mixing schemas -- the file is regenerable and
gitignored, so discarding it is always safe.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

SCHEMA_VERSION = 2

logger = logging.getLogger(__name__)


class SmilesKeyedCache:
    """In-memory frame of per-molecule values keyed by canonical SMILES,
    optionally persisted to parquet across runs.
    """

    def __init__(
        self,
        columns: Iterable[str],
        path: str | Path | None = None,
        provenance: Mapping[str, str] | None = None,
    ):
        self.columns = list(columns)
        self.path = Path(path) if path else None
        self.provenance = dict(provenance or {})
        self._df = self._load()

    @property
    def meta_path(self) -> Path | None:
        if self.path is None:
            return None
        return self.path.with_suffix(".meta.json")

    def _expected_meta(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "columns": self.columns,
            "provenance": self.provenance,
        }

    def _empty(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self.columns).rename_axis("smiles")

    def _load(self) -> pd.DataFrame:
        if self.path is None or not self.path.exists():
            return self._empty()

        expected = self._expected_meta()
        meta_path = self.meta_path
        found = None
        if meta_path is not None and meta_path.exists():
            try:
                found = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                found = None
        if found != expected:
            logger.warning(
                "cache %s has schema %s but %s is expected -- discarding and rebuilding",
                self.path,
                found,
                expected,
            )
            return self._empty()

        df = pd.read_parquet(self.path).set_index("smiles")
        missing_columns = [c for c in self.columns if c not in df.columns]
        if missing_columns:
            logger.warning(
                "cache %s is missing columns %s -- discarding and rebuilding", self.path, missing_columns
            )
            return self._empty()
        return df

    def __contains__(self, smiles: str) -> bool:
        return smiles in self._df.index

    def __len__(self) -> int:
        return len(self._df)

    def missing(self, canonical_smiles: Iterable[str]) -> list[str]:
        """Deduplicated, order-preserving list of keys not yet cached."""
        return [s for s in dict.fromkeys(canonical_smiles) if s not in self]

    def lookup(self, canonical_smiles: Iterable[str]) -> pd.DataFrame:
        return self._df.reindex(list(canonical_smiles))

    def add(self, rows: pd.DataFrame) -> None:
        rows = rows.rename_axis("smiles")
        new = rows[~rows.index.isin(self._df.index)]
        if new.empty:
            return
        self._df = new if self._df.empty else pd.concat([self._df, new])

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._df.reset_index().to_parquet(self.path, index=False)
        self.meta_path.write_text(json.dumps(self._expected_meta(), indent=2, sort_keys=True) + "\n")

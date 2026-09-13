"""Applicability-domain constraint (brief SS3.3/SS4, arm B/C): max ECFP4
Tanimoto similarity to a reference set, applied as a continuous multiplicative
soft penalty below theta -- never a hard filter. Bulk Tanimoto is vectorized
via a single 0/1 fingerprint matrix multiply (a dot product of 0/1 vectors
equals the AND-popcount), since this runs on every GA candidate, every
generation, against the whole reference set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

DEFAULT_RADIUS = 2
DEFAULT_N_BITS = 2048


def _fingerprint_generator(radius: int, n_bits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def bulk_ecfp4_bits(
    smiles_list: list[str], radius: int = DEFAULT_RADIUS, n_bits: int = DEFAULT_N_BITS
) -> np.ndarray:
    """(n, n_bits) uint8 matrix of ECFP4 bits; rows for invalid SMILES are all-zero."""
    gen = _fingerprint_generator(radius, n_bits)
    matrix = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), matrix[i])
    return matrix


def bulk_max_tanimoto(query_bits: np.ndarray, reference_bits: np.ndarray) -> np.ndarray:
    """Max Tanimoto similarity of each query fingerprint against every reference
    fingerprint. Similarity between two all-zero fingerprints is defined as 0
    (not 1) to avoid rewarding invalid/empty candidates.
    """
    query = np.atleast_2d(query_bits).astype(np.float32)
    reference = np.atleast_2d(reference_bits).astype(np.float32)
    intersection = query @ reference.T
    query_popcount = query.sum(axis=1, keepdims=True)
    reference_popcount = reference.sum(axis=1, keepdims=True).T
    union = query_popcount + reference_popcount - intersection
    similarity = np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0
    )
    return similarity.max(axis=1)


def soft_penalty_above_threshold_is_bad(values: np.ndarray, theta: float) -> np.ndarray:
    """1.0 at/above theta, continuous multiplicative decay below it."""
    return np.minimum(1.0, np.asarray(values, dtype=float) / theta)


class ReferenceFingerprintIndex:
    """ECFP4 fingerprints of a reference set, optionally cached to a .npz file
    keyed by (smiles list, radius, n_bits) via the cache path itself.
    """

    def __init__(self, smiles: list[str], bits: np.ndarray):
        self.smiles = list(smiles)
        self.bits = bits

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        smiles_column: str = "smiles",
        cache_path: str | Path | None = None,
        radius: int = DEFAULT_RADIUS,
        n_bits: int = DEFAULT_N_BITS,
    ) -> "ReferenceFingerprintIndex":
        cache_path = Path(cache_path) if cache_path else None
        if cache_path is not None and cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            return cls(list(data["smiles"]), data["bits"])

        smiles = pd.read_csv(path)[smiles_column].astype(str).tolist()
        bits = bulk_ecfp4_bits(smiles, radius=radius, n_bits=n_bits)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, smiles=np.array(smiles, dtype=object), bits=bits)
        return cls(smiles, bits)

    def max_tanimoto(
        self, query_smiles: list[str], radius: int = DEFAULT_RADIUS, n_bits: int = DEFAULT_N_BITS
    ) -> np.ndarray:
        query_bits = bulk_ecfp4_bits(query_smiles, radius=radius, n_bits=n_bits)
        return bulk_max_tanimoto(query_bits, self.bits)

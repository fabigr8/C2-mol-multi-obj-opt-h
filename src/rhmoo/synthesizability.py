"""Synthetic accessibility (brief SS3.3, arm C): wraps RDKit's bundled
`sascorer` contrib script (not part of the core package's public API).
Applied as a continuous multiplicative soft penalty above the threshold, for
the same reason the applicability-domain constraint is a soft penalty rather
than a hard filter (brief SS3.3): it preserves gradient signal for the GA
instead of a cliff.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import RDConfig

_SA_SCORE_DIR = os.path.join(RDConfig.RDContribDir, "SA_Score")
if _SA_SCORE_DIR not in sys.path:
    sys.path.append(_SA_SCORE_DIR)

import sascorer  # noqa: E402


def sa_score(smiles: str) -> float | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return sascorer.calculateScore(mol)


def sa_penalty(scores: np.ndarray, threshold: float) -> np.ndarray:
    """1.0 at/below threshold, continuous multiplicative decay above it."""
    scores = np.asarray(scores, dtype=float)
    return np.minimum(1.0, threshold / scores)

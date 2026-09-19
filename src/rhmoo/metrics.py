"""Metrics module (brief SS4), computed per arm x seed for the top-k molecules
and for the control arm:

- objective trajectory / final top-k distribution, per-component breakdown
- distribution shift (H2): AD nearest-neighbour Tanimoto similarity
- independent-scorer divergence (H3, highest priority): rank correlation and
  mean absolute disagreement between the objective's own predictors and the
  independent scorer
- chemical plausibility: physchem panel, SA score, PAINS/Brenk alerts, and the
  pre-declared implausibility battery (fails >= fail_threshold_count)

This module only consumes precomputed score DataFrames (from `predictors`,
`independent_scorer`, `applicability_domain`, `synthesizability`); it has no
ADMET-AI dependency of its own, so it stays fast to unit test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from rhmoo.synthesizability import sa_score

PHYSCHEM_COLUMNS = (
    "molecular_weight",
    "clogp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "ring_count",
    "largest_ring_size",
    "fraction_csp3",
    "formal_charge",
    "longest_aliphatic_chain",
)


def _catalog(name: str) -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, name))
    return FilterCatalog(params)


_PAINS_CATALOG = _catalog("PAINS")
_BRENK_CATALOG = _catalog("BRENK")


def longest_aliphatic_chain(mol: Chem.Mol) -> int:
    """Length (atom count) of the longest chain of non-ring carbons.

    The induced subgraph of non-ring atoms cannot contain a cycle (any cycle
    is by definition a ring, so its atoms are flagged `IsInRing`), so each
    connected component is a tree/forest and the longest path within it can be
    found with the standard two-pass BFS ("double sweep") algorithm.
    """
    chain_atoms = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C" and not atom.GetIsAromatic() and not atom.IsInRing()
    }
    if not chain_atoms:
        return 0

    adjacency: dict[int, list[int]] = {idx: [] for idx in chain_atoms}
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in chain_atoms and b in chain_atoms:
            adjacency[a].append(b)
            adjacency[b].append(a)

    def _farthest(start: int) -> tuple[int, int]:
        visited = {start}
        frontier = [start]
        depth = 0
        farthest_node = start
        while frontier:
            next_frontier = []
            for node in frontier:
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
                        farthest_node = neighbor
            if next_frontier:
                depth += 1
            frontier = next_frontier
        return farthest_node, depth

    longest = 0
    seen: set[int] = set()
    for atom_idx in chain_atoms:
        if atom_idx in seen:
            continue
        far_node, _ = _farthest(atom_idx)
        _, hops = _farthest(far_node)
        component = set()
        stack = [far_node]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node])
        seen |= component
        longest = max(longest, hops + 1)
    return longest


def physchem_panel(smiles_list: list[str]) -> pd.DataFrame:
    """One row per input SMILES (same order); rows for invalid SMILES are all-NaN."""
    rows = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rows.append({col: np.nan for col in PHYSCHEM_COLUMNS})
            continue
        ring_sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings()]
        rows.append(
            {
                "molecular_weight": Descriptors.MolWt(mol),
                "clogp": Descriptors.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol),
                "hbd": Descriptors.NumHDonors(mol),
                "hba": Descriptors.NumHAcceptors(mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "ring_count": rdMolDescriptors.CalcNumRings(mol),
                "largest_ring_size": max(ring_sizes, default=0),
                "fraction_csp3": Descriptors.FractionCSP3(mol),
                "formal_charge": Chem.GetFormalCharge(mol),
                "longest_aliphatic_chain": longest_aliphatic_chain(mol),
            }
        )
    return pd.DataFrame(rows, columns=list(PHYSCHEM_COLUMNS))


def structural_alerts(smiles_list: list[str]) -> pd.DataFrame:
    """One row per input SMILES: PAINS and Brenk alert counts (NaN if invalid)."""
    rows = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            rows.append({"pains_alerts": np.nan, "brenk_alerts": np.nan})
            continue
        rows.append(
            {
                "pains_alerts": len(_PAINS_CATALOG.GetMatches(mol)),
                "brenk_alerts": len(_BRENK_CATALOG.GetMatches(mol)),
            }
        )
    return pd.DataFrame(rows, columns=["pains_alerts", "brenk_alerts"])


def sa_scores(smiles_list: list[str]) -> np.ndarray:
    """SA score per input SMILES (NaN if invalid)."""
    return np.array([np.nan if (s := sa_score(smiles)) is None else s for smiles in smiles_list])


def implausibility_battery(
    panel: pd.DataFrame, alerts: pd.DataFrame, sa: np.ndarray, battery: dict
) -> pd.DataFrame:
    """Pre-declared implausibility battery (brief SS4/SS8): one boolean column per
    check, `fail_count`, and `implausible` (fails >= `fail_threshold_count`).
    A molecule with any NaN input (invalid SMILES) fails every check it can't
    be evaluated on, since an unparseable molecule cannot be judged plausible.
    """
    sa_series = pd.Series(sa, index=panel.index)

    def _check(values: pd.Series, max_value: float) -> pd.Series:
        # NaN (unparseable molecule) fails the check rather than comparing False.
        return (values > max_value) | values.isna()

    checks = pd.DataFrame(index=panel.index)
    checks["fails_mw"] = _check(panel["molecular_weight"], battery["mw_max"])
    checks["fails_clogp"] = _check(panel["clogp"], battery["clogp_max"])
    checks["fails_tpsa"] = _check(panel["tpsa"], battery["tpsa_max"])
    checks["fails_rotatable_bonds"] = _check(panel["rotatable_bonds"], battery["rotatable_bonds_max"])
    checks["fails_sa"] = _check(sa_series, battery["sa_score_max"])
    checks["fails_pains"] = _check(alerts["pains_alerts"], battery["pains_alerts_max"])
    checks["fails_brenk"] = _check(alerts["brenk_alerts"], battery["brenk_alerts_max"])
    checks["fail_count"] = checks.sum(axis=1)
    checks["implausible"] = checks["fail_count"] >= battery["fail_threshold_count"]
    return checks


def select_top_k(smiles: list[str], fitness: np.ndarray, k: int) -> tuple[list[str], np.ndarray]:
    """Indices of the `k` highest-fitness molecules, ties broken by original order."""
    k = min(k, len(smiles))
    order = np.argsort(-np.asarray(fitness, dtype=float), kind="stable")[:k]
    return [smiles[i] for i in order], np.asarray(fitness)[order]


def objective_trajectory(trajectory: pd.DataFrame) -> pd.DataFrame:
    """Per-generation summary (mean/max/min/std/count) of the `fitness` column,
    restricted to successfully evaluated (non-NaN fitness) molecules.
    """
    valid = trajectory[trajectory["fitness"].notna()]
    return valid.groupby("generation")["fitness"].agg(["mean", "max", "min", "std", "count"]).reset_index()


def component_breakdown(df: pd.DataFrame, component_columns: list[str]) -> pd.DataFrame:
    """Mean/std/min/max of each named column, one row per component."""
    summary = df[component_columns].agg(["mean", "std", "min", "max"]).T
    return summary.rename_axis("component").reset_index()


def nearest_neighbor_similarity(smiles_list: list[str], reference_index) -> np.ndarray:
    """Max ECFP4 Tanimoto similarity of each molecule to its nearest neighbour in
    `reference_index` (a `rhmoo.applicability_domain.ReferenceFingerprintIndex`).
    """
    return reference_index.max_tanimoto(smiles_list)


def independent_divergence(
    objective_scores: pd.DataFrame, independent_scores: pd.DataFrame, pairs: dict[str, str]
) -> pd.DataFrame:
    """Rank correlation and mean absolute disagreement between the objective's
    own raw predictor value and the independent scorer, for each (objective
    column -> independent column) pair. Rows with a NaN on either side are
    dropped before computing each pair's statistics.
    """
    rows = []
    for objective_col, independent_col in pairs.items():
        paired = pd.DataFrame(
            {
                "objective": objective_scores[objective_col].to_numpy(),
                "independent": independent_scores[independent_col].to_numpy(),
            }
        ).dropna()
        rows.append(
            {
                "property": objective_col,
                "n": len(paired),
                "spearman_r": (
                    paired["objective"].corr(paired["independent"], method="spearman")
                    if len(paired) > 1
                    else np.nan
                ),
                "mean_abs_disagreement": (
                    (paired["objective"] - paired["independent"]).abs().mean() if len(paired) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def divergence_headline(
    starting_divergence: pd.DataFrame, optimized_divergence: pd.DataFrame
) -> pd.DataFrame:
    """The project's headline number (brief SS4): per-property divergence on
    the starting population vs the optimized top-k, side by side.
    """
    merged = starting_divergence.merge(
        optimized_divergence, on="property", suffixes=("_starting", "_optimized")
    )
    merged["mean_abs_disagreement_delta"] = (
        merged["mean_abs_disagreement_optimized"] - merged["mean_abs_disagreement_starting"]
    )
    return merged

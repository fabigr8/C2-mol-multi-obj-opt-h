"""Per-molecule scoring shared by the runner and the pre-run arm R script.

Arm R (the approved drugs) has to be scored *before* the experiment, because
brief SS4.5 requires the implausibility battery to be calibrated against it
before any arm A molecule is generated -- and it also has to be *reported in*
the experiment. Scoring it twice would mean the arm R in RESULTS.md was not
necessarily the set the thresholds were calibrated on, so the scoring lives
here and both callers use it.

Deliberately excludes the implausibility battery: the battery's thresholds are
an *output* of calibrating against this frame, so applying them here would be
circular. Callers apply `metrics.implausibility_battery` to the panel/alerts/SA
columns once thresholds exist.
"""

from __future__ import annotations

import pandas as pd

from rhmoo.applicability_domain import ReferenceFingerprintIndex
from rhmoo.independent_scorer import IndependentScorerSet
from rhmoo.metrics import nearest_neighbor_similarity, physchem_panel, sa_scores, structural_alerts

#: Columns the battery needs, produced here so calibration and the runner both
#: read them from the same frame.
BATTERY_INPUT_COLUMNS = ("sa_score", "pains_alerts", "brenk_alerts")


def score_molecule_set(
    smiles: list[str],
    pipeline,
    scorers: IndependentScorerSet,
    predictor_reference_index: ReferenceFingerprintIndex | None = None,
    starting_population_index: ReferenceFingerprintIndex | None = None,
) -> pd.DataFrame:
    """One row per molecule: objective components (raw + normalized), composite
    score, ensemble spread, every independent scorer variant, the physchem
    panel, structural alerts, SA score and nearest-neighbour similarities.

    Order is preserved and invalid molecules are kept as NaN rows rather than
    dropped (brief SS9).
    """
    scored = pipeline(smiles).reset_index(drop=True)
    # The pipeline adds sa_score only for arms with the synthesizability
    # constraint; drop it so the concat below cannot produce a duplicate
    # column that later assignments would silently write to twice.
    scored = scored.drop(columns=[c for c in ("sa_score",) if c in scored.columns])
    panel = physchem_panel(smiles)
    alerts = structural_alerts(smiles)
    independent = scorers.score(smiles)

    detail = pd.concat([scored, panel, alerts, independent], axis=1)
    detail["sa_score"] = sa_scores(smiles)

    if predictor_reference_index is not None:
        detail["ad_similarity_vs_reference"] = nearest_neighbor_similarity(smiles, predictor_reference_index)
    if starting_population_index is not None:
        detail["ad_similarity_vs_starting"] = nearest_neighbor_similarity(smiles, starting_population_index)

    return detail

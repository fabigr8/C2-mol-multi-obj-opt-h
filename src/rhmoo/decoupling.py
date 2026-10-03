"""H3b: per-generation objective-vs-independent decoupling (brief SS4.3).

Brief SS4.3 asks for the H3 divergence computed per generation rather than only
on the final top-k, and calls it the one output of the project with direct
operational value: it says whether divergence onset is detectable mid-run, which
is what would make it usable as an early-stopping or monitoring signal.

Three design points, all pre-declared in `docs/PREREGISTRATION.md`:

1. **The elite set, not every logged molecule.** Brief SS4.3 calls this "cheap"
   because "the molecules are already logged and predictions are cached". That
   is only true for the elite set: the independent scorers' featurization runs
   at ~80 mol/s, so rescoring the ~455k logged molecules of a full run costs
   about as much as the entire experiment. The elite set is ~20 molecules per
   generation, heavily duplicated across generations, and cached -- which is
   genuinely cheap.

2. **Comparability by construction.** The composite is in [0, 1] while the
   scorers emit a probability and a log-solubility, so the two series are not
   directly comparable as they stand. But ADMET-AI and the independent scorers
   predict the *same quantities in the same units*, so both are pushed through
   the same DrugBank-percentile normalization and the same aggregation,
   restricted to the two covered endpoints with weights renormalised. That is
   pure reuse of `normalization` and `objective`, and it yields two [0, 1]
   series on one axis.

3. **The delta, not the raw gap.** The independent scorers are imperfect
   models, so a constant offset between the series is oracle error, not
   exploitation (brief SS9: "some measured divergence is oracle error").
   Detecting on `gap(g) - gap(0)` cancels it.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from rhmoo.config import ExperimentConfig, ObjectiveConfig
from rhmoo.independent_scorer import INDEPENDENT_ENDPOINTS, independent_column
from rhmoo.objective import aggregate, normalize_components


def restricted_objective(config: ExperimentConfig) -> ObjectiveConfig:
    """The objective restricted to the endpoints an independent scorer covers.

    Weights keep their configured values and are renormalised by `aggregate`,
    so the restricted composite is the original objective's own weighting of
    the covered subset rather than a new, arbitrary one.
    """
    components = tuple(c for c in config.objective.components if c.name in INDEPENDENT_ENDPOINTS)
    if not components:
        raise ValueError("no objective component is covered by an independent scorer")
    return dataclasses.replace(config.objective, components=components)


def _composite(objective: ObjectiveConfig, normalizer, values: pd.DataFrame) -> np.ndarray:
    normalized = normalize_components(objective, normalizer, values)
    return aggregate(objective, normalized).to_numpy(dtype=float)


def decoupling_curves(
    elites: pd.DataFrame,
    scores: pd.DataFrame,
    config: ExperimentConfig,
    normalizer,
    variant: str,
) -> pd.DataFrame:
    """One row per generation: the two comparable series and their gap.

    `elites` is the GA's elite log (generation, rank, smiles). `scores` must
    carry, per SMILES, the raw ADMET-AI values for the covered endpoints and
    the matching independent-scorer columns for `variant`.
    """
    objective = restricted_objective(config)
    names = [c.name for c in objective.components]

    scores = scores.drop_duplicates(subset="smiles").set_index("smiles")
    aligned = scores.reindex(elites["smiles"].tolist())

    objective_values = aligned[names].reset_index(drop=True)
    independent_values = aligned[[independent_column(n, variant) for n in names]].reset_index(drop=True)
    independent_values.columns = names

    frame = pd.DataFrame(
        {
            "generation": elites["generation"].to_numpy(),
            "objective_restricted": _composite(objective, normalizer, objective_values),
            "independent_composite": _composite(objective, normalizer, independent_values),
        }
    )

    curve = frame.groupby("generation", as_index=False).mean()
    curve["gap"] = curve["objective_restricted"] - curve["independent_composite"]
    baseline = curve.loc[curve["generation"].idxmin(), "gap"]
    curve["gap_delta"] = curve["gap"] - baseline
    curve["variant"] = variant
    return curve


def decoupling_generation(curve: pd.DataFrame, margin: float, persistence: int) -> int | None:
    """First generation at which `gap_delta` exceeds `margin` for `persistence`
    consecutive generations, or None if it never does.

    None is a legitimate, reportable outcome -- "did not decouple within
    budget" (brief SS9: report the null).
    """
    curve = curve.sort_values("generation")
    generations = curve["generation"].to_numpy()
    exceeded = (curve["gap_delta"].to_numpy() > margin).astype(int)

    run = 0
    for i, flag in enumerate(exceeded):
        run = run + 1 if flag else 0
        if run >= persistence:
            return int(generations[i - persistence + 1])
    return None


def decoupling_summary(curve: pd.DataFrame, config: ExperimentConfig, variant: str) -> list[dict]:
    """The primary pre-declared margin plus the declared sensitivity sweep.

    Reporting a sweep is not tuning: the primary margin is fixed in config and
    the sweep is declared alongside it (brief SS9).
    """
    rows = []
    for margin in config.decoupling.margin_sweep:
        generation = decoupling_generation(curve, margin, config.decoupling.persistence)
        rows.append(
            {
                "variant": variant,
                "margin": margin,
                "persistence": config.decoupling.persistence,
                "is_primary_margin": margin == config.decoupling.margin,
                "decoupling_generation": generation,
                "decoupled": generation is not None,
                "final_gap_delta": float(curve["gap_delta"].iloc[-1]),
                "max_gap_delta": float(curve["gap_delta"].max()),
            }
        )
    return rows

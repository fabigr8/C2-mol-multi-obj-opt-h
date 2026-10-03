# Deviations and brief corrections

Points where the implementation departs from `docs/BRIEFB-V2.md`, or where the
brief is incorrect about the repository or internally inconsistent. Recorded so
the audit trail is explicit rather than implied by the code.

## 1. QED was never dropped (brief SS3.1, SS10, SS12 are incorrect)

Brief v2 states that "the implementation dropped it" and makes restoring QED a
blocking item, with any removal requiring sign-off and an additional arm.

This is factually wrong about the repository. QED has been a first-class
objective component since Phase 3: declared in both `configs/main.yaml` and
`configs/smoke.yaml` with `source: rdkit`, `direction: max`, `weight: 0.2`, and
computed by `compute_qed` in `src/rhmoo/predictors.py`. The v1 brief listed QED
as adjustable and it was never adjusted.

**Resolution:** requirement already satisfied. No code change, no deviation, no
sign-off, no additional arm.

## 2. The data-control cannot be disjoint from the objective's training data (brief SS4.2)

Brief SS4.2 asks for a data-control "trained on a disjoint half of the TDC
`train_val` split, with the objective-side half excluded". ADMET-AI is a fixed
pretrained artifact fit on the *whole* `train_val` split, and brief SS2 forbids
retraining the objective's predictors, so no half of `train_val` is disjoint
from the objective's training data.

**Resolution:** three variants are shipped instead of two. `data_control_half`
implements the brief literally and is labelled honestly; `data_control_test`,
trained on the TDC `test` split, is the only genuinely disjoint control. Both
are reported, never averaged. See `docs/LIMITATIONS.md` section 1.

## 3. "Committed in one commit, with the git SHA recorded" is self-referential (brief SS4.5)

A commit cannot contain its own SHA.

**Resolution:** the calibration script, its outputs and the frozen configs are
committed in a single commit, and that commit's SHA is recorded out-of-band by
an immediately-created annotated tag `prereg/battery-<date>`. Every run
additionally records its own `git_sha` in `run_log.jsonl`.

## 4. Per-generation rescoring is cheap only for the elite set (brief SS4.3)

Brief SS4.3 describes the per-generation divergence as cheap because "the
molecules are already logged and predictions are cached". The ADMET-AI cache
does nothing for the independent scorers, whose featurization runs at roughly
80 molecules/second; rescoring all ~455,000 logged molecules of a full run would
cost about as much as the entire experiment.

**Resolution:** the analysis covers the per-generation **elite set** only, with
featurization shared across scorer variants and a persistent score cache. The
elite-set definition is pre-declared in `docs/PREREGISTRATION.md`.

## 5. The GA did not record elite membership

Elites are carried into the next generation without being re-evaluated, so a
carried elite's trajectory row keeps its original generation label and nothing
recorded which molecules were elite at generation `g` -- which brief SS4.3
requires.

**Resolution:** `src/rhmoo/ga.py` now logs the elite set explicitly to a
separate `*_elites.parquet`, leaving the trajectory schema untouched. Replaying
the elite set from the trajectory was rejected: it would couple a load-bearing
number to log row order and to argsort tie-breaking.

## 6. Arm C's SA constraint is a soft penalty, not a hard filter (brief SS3.3)

Carried over from v1. Brief SS3.3 specifies "SA score <= threshold" for arm C
and mandates a soft penalty only for arm B's AD constraint. Arm C uses the same
soft-penalty shape, because a hard filter would reintroduce the score cliff that
arm B's soft penalty exists to avoid, and the two constraints should be
comparable.

## 7. Committed artifacts live outside `results/` (brief SS5, SS3.4)

Brief SS5 places `battery_calibration.csv` under `results/tables/` while SS4.5
requires it to be committed, and SS3.4 requires the approved-drug set to be
"pinned and hashed like any other data artifact". `.gitignore` excludes `data/`
and `results/` wholesale, so neither is possible as written.

**Resolution:** a committed top-level `calibration/` directory holds the arm R
artifact, its manifest, and the calibration outputs. `make reproduce` copies
`battery_calibration.csv` into `results/<name>/tables/` so the documented output
layout still holds.

## 8. Figure 1 layout (brief SS5)

Brief SS5 requires figure 1 to be "publication-legible at blog width". Rendering
20 structures in a single row produces an image roughly 4,400px wide.

**Resolution:** 5 structures per row across 4 rows per arm, with the composite
score, the headline independent-scorer score and the arm R percentile annotated
on each.

## 9. The implausibility battery was recalibrated (supersedes PLAN Phase 8)

Phase 8 froze thresholds by declaration. Measured against arm R they flag 15.43%
of approved drugs, against brief SS4.5's 5% ceiling. The thresholds were
recalibrated; see `docs/PREREGISTRATION.md`. The old values are retained in
`docs/PLAN.md` because the size of the delta is the justification.

## 10. The reported top-k spans the run, not the final population

Not specified by the brief, which asks only for "the top-k molecules per arm per
seed" (SS4). The top-k is drawn from every distinct molecule the run evaluated,
keyed on canonical SMILES.

Drawing from the final population is fragile: an elitist GA converges, so the
final population can hold very few distinct structures, and SELFIES decoding
emits several spellings of the same molecule. Together those inflate every `n`
and make rank correlations degenerate. Population collapse is still reported, as
`final_population_unique_fraction`.

## 11. Mutation rate raised from 0.1 to 1.0

Not specified by the brief, which fixes the operator set but not the rate
(SS3.2). Crossover is off by default, so mutation is the only variation operator
and a rate of 0.1 made 90% of offspring exact clones of their parent: 86 of 580
evaluations were distinct molecules, and the reported top-20 held 1-2 of them.

Measured at 1.0 the achieved objective score is unchanged within seed noise
(0.8118 vs 0.8276), so this does not flatter or penalise the optimizer -- it
stops ~90% of the compute budget being spent re-evaluating duplicates. The full
A/B is in `docs/PREREGISTRATION.md`.

## 12. The implausibility battery has a disqualifying tier (brief SS4.5)

Brief SS4.5 defines implausibility as "fails >= 2 of a pre-declared battery".
That treats every criterion as an exchangeable soft signal, which is right for
the continuous criteria and wrong for a structural impossibility: a carbanion is
not half-implausible, and such molecules are small enough to pass every size and
lipophilicity criterion, so a pure counting rule never flags them.

`implausible_charge_centres` therefore flags a molecule on its own. The
constraint SS4.5 exists to protect -- no more than 5% of approved drugs flagged
-- is still enforced over the combined rule (measured 0.0492).

The criterion was added after a reduced-scale rehearsal showed the counting-only
battery reporting an implausibility rate of 0.000 for every optimized arm while
a large share of those molecules carried implausible charges. That provenance,
and the four things that bound the resulting risk, are disclosed in
`docs/PREREGISTRATION.md` rather than left implicit.

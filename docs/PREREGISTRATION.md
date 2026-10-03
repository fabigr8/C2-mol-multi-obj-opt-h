# Pre-registration

Thresholds and detector definitions fixed before the first full run, per brief
SS9 ("Pre-declare thresholds") and SS4.5 ("Calibrate, don't just freeze").

This document and the artifacts it references were committed in a single commit
on top of `d9364430138e4e542cd1eabd17408ccc93a10d5f`, before any arm A molecule was generated. `results/` did
not exist at that point. A commit cannot contain its own SHA, so the commit is
additionally recorded out-of-band by the annotated tag `prereg/battery-2026-10-03`,
and every run records its own `git_sha` in `run_log.jsonl`.

## 1. Implausibility battery (brief SS4.5)

Calibrated against **arm R** -- the 2,845 approved drugs of the DrugBank set
shipped with ADMET-AI, scored by the pinned pipeline into
`calibration/arm_r_scored.parquet`. Arm R is the only data input to the
calibration.

### Supersedes the Phase 8 freeze

`docs/PLAN.md` Phase 8 froze these thresholds by *declaration* on 2026-09-19:
MW <= 700, cLogP <= 7, TPSA <= 200, rotatable bonds <= 15, SA <= 6.0, 0 PAINS,
0 Brenk, fails >= 2.

Measured against arm R, that battery flags **15.43%** of approved drugs -- more
than 3x brief SS4.5's 5% ceiling. The dominant cause is the zero-Brenk rule:
**48.30%** of approved drugs carry at least one Brenk alert. Per-criterion flag
rates under the old thresholds were Brenk 0.4830, SA 0.0787, TPSA 0.0773,
MW 0.0699, rotatable bonds 0.0503, PAINS 0.0485, cLogP 0.0281.

A battery that flags one approved drug in six cannot distinguish an exploited
molecule from an ordinary drug, so every downstream implausibility number would
have been uninterpretable. The old values are kept in `docs/PLAN.md` rather than
deleted, because the size of the delta is itself the justification for
recalibrating.

### The rule

The criteria set is **fixed** at the seven pre-declared checks; calibration
moves only thresholds. Adding criteria at a fixed `fail_threshold_count` makes
the battery *looser*, because each threshold must then sit further into the
approved-drug tail to hold the joint rate -- measured directly: a 9-criterion
variant drove alpha* to 0.03 and MW to 1040.

Thresholds are a function of a single tail mass `alpha`:

- **Continuous criteria** take arm R's `(1 - alpha)` empirical quantile, rounded
  **up** to a declared grid (MW and TPSA to 10, cLogP and SA to 0.5, rotatable
  bonds to 1), so the frozen value is a readable number rather than a
  six-decimal quantile.
- **Count criteria** (PAINS, Brenk) take the smallest integer `c` with
  `P_armR(count > c) <= alpha`. This is what converts a zero-tolerance alert
  count into a calibrated one.

`alpha` is scanned over the declared grid `[0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]` and the
**largest** alpha whose joint `fails >= 2` rate on arm R is `<= 0.05` is
chosen -- the most sensitive battery consistent with the constraint. The joint
rate is monotone non-decreasing in alpha, so the solution is unique.

### The outcome

**alpha\* = 0.04**, joint flag rate on arm R = **0.0425**.

| criterion | column | threshold | arm R flag rate | rule |
|---|---|---|---|---|
| `mw_max` | `molecular_weight` | 880.0 | 0.0401 | ceil(quantile(arm_R[molecular_weight], 1-alpha) / 10.0) * 10.0 |
| `clogp_max` | `clogp` | 6.5 | 0.0397 | ceil(quantile(arm_R[clogp], 1-alpha) / 0.5) * 0.5 |
| `tpsa_max` | `tpsa` | 290.0 | 0.0376 | ceil(quantile(arm_R[tpsa], 1-alpha) / 10.0) * 10.0 |
| `rotatable_bonds_max` | `rotatable_bonds` | 18.0 | 0.0376 | ceil(quantile(arm_R[rotatable_bonds], 1-alpha) / 1.0) * 1.0 |
| `sa_score_max` | `sa_score` | 7.5 | 0.0348 | ceil(quantile(arm_R[sa_score], 1-alpha) / 0.5) * 0.5 |
| `pains_alerts_max` | `pains_alerts` | 1.0 | 0.0039 | min integer c with P(arm_R[pains_alerts] > c) <= alpha |
| `brenk_alerts_max` | `brenk_alerts` | 3.0 | 0.0130 | min integer c with P(arm_R[brenk_alerts] > c) <= alpha |
| `JOINT_fails_ge_2` | `implausible` | 2.0 | 0.0425 | fail_count >= 2 |

Full scan (`calibration/battery_frozen.json` carries this too):

| alpha | joint flag rate | within target |
|---|---|---|
| 0.10 | 0.1216 | no |
| 0.09 | 0.1107 | no |
| 0.08 | 0.0998 | no |
| 0.07 | 0.0861 | no |
| 0.06 | 0.0752 | no |
| 0.05 | 0.0573 | no |
| 0.04 | 0.0425 | yes |
| 0.03 | 0.0334 | yes |
| 0.02 | 0.0250 | yes |
| 0.01 | 0.0105 | yes |

### Why this is pre-registration, not post-hoc fitting

Four properties, each mechanically checkable:

1. The only data input is arm R. No arm A molecule existed when it ran.
2. The rule has a **single scalar degree of freedom**, solved by a declared
   monotone scan. There is no per-criterion freedom -- which is precisely the
   freedom post-hoc threshold picking exploits.
3. The criteria set, rounding grids, alpha grid, target rate and selection rule
   were written down before the script ran.
4. `rhmoo.preflight` makes the runner **refuse to start** unless the active
   config's battery byte-matches `calibration/battery_frozen.json`.

### Declared sensitivity, including a blind spot

Sensitivity is demonstrated on pre-declared positive controls -- hand-written
degenerate structures -- so it can be measured without looking at arm A.
Results are committed to `calibration/battery_positive_controls.csv`.

The calibrated battery flags all three **size/lipophilicity/alert-driven**
degenerates (a 60-carbon alkane chain, a polyhalogenated chain, a high-MW
repeat unit).

It does **not** flag either **ring-topology** degenerate: a 42-membered
all-carbon macrocycle and a heavily fused polyaromatic each fail only one
criterion. The seven frozen criteria contain no ring-count, ring-size or
aromaticity check, so this class of implausibility is invisible to the battery.

This is declared, not patched: the criteria set is fixed (see above), and adding
ring criteria would loosen every other threshold. The consequence for
interpretation is recorded in `docs/LIMITATIONS.md` -- **the implausibility rate
is a lower bound on implausibility, not a complete account of it.**

## 2. H3b decoupling detector (brief SS4.3)

Declared in `configs/*.yaml` under `decoupling:`.

- `elite_set(g)` = the top `n_elite = max(1, int(population_size * elitism_fraction))`
  molecules of the population at the **end** of generation `g`, ranked by
  **fitness** (penalised -- exactly what the GA carries forward), ties broken by
  stable order. `elite_set(0)` is the initial population's top `n_elite`. Logged
  directly by the GA to `<label>_seed<seed>_elites.parquet`, because carried
  elites are not re-evaluated and so leave no per-generation row in the
  trajectory.
- Both the objective's ADMET-AI values and each independent scorer's predictions
  are pushed through the **same** DrugBank-percentile normalization and the
  **same** aggregation, restricted to the two covered endpoints
  (hERG, Solubility_AqSolDB) with weights renormalised. This yields two
  directly comparable `[0, 1]` series.
- `gap(g) = obj_restricted(g) - indep_composite(g)`. The decoupling generation
  is the first `g` at which `gap(g) - gap(0) > margin` for `persistence`
  consecutive generations. **The delta is used, not the raw gap**, so the
  constant oracle offset cancels (brief SS9: "some measured divergence is oracle
  error").
- Primary: `margin = 0.1`, `persistence = 3`, scorer variant
  `data_control_half` (brief SS4.2: "the data-control is the load-bearing
  scorer"). Declared sensitivity sweep over margins 0.05 / 0.10 / 0.15.
- "Did not decouple within budget" is a legitimate recorded outcome.

## 3. Reported top-k and search configuration

Both fixed before the first full run, after a reduced-scale rehearsal showed the
original configuration could not support the brief SS4 metrics.

### The reported top-k spans the whole run

`top_k(arm, seed)` = the `k_top` highest-**fitness** distinct molecules the run
evaluated, keyed on canonical SMILES, ties broken by SMILES so the selection
does not depend on evaluation order.

Brief SS4 asks for "the top-k molecules per arm per seed" and does not specify
the final population. Drawing from the final population is fragile: an elitist
GA converges, so a 200-slot final population routinely holds only a handful of
distinct structures, and the highest-scoring molecules the optimizer found --
which is exactly where the exploits are -- may have been discovered and then
crowded out generations earlier.

Keying on the canonical SMILES matters independently: SELFIES decoding emits
whichever spelling the token sequence produces, so one molecule otherwise
occupies several top-k slots and inflates every `n` and rank correlation.

Population collapse is still measured and reported, as
`final_population_unique_fraction`. It is a real property of the search, so it
is recorded rather than allowed to silently thin every downstream statistic.

### Mutation rate

`ga.mutation_rate` is **1.0**: every offspring receives exactly one token
mutation.

Crossover is off by default (brief SS3.2), so mutation is the only variation
operator and any rate below 1.0 makes that fraction of offspring exact clones of
their parent. At the original 0.1, measured on arm A at rehearsal scale
(population 40, 15 generations):

| `mutation_rate` | offspring that were clones | distinct molecules evaluated (of 580) | distinct molecules in the reported top-20 | objective score (mean) |
|---|---|---|---|---|
| 0.1 | 84% | 86 | 1.5 | 0.8276 |
| 0.5 | 46% | 247 | 4.0 | 0.8510 |
| **1.0** | **0%** | **457** | **16.5** | **0.8118** |

The achieved objective score is unchanged within seed noise, so this is not a
change that makes the optimizer look better or worse at its task -- it stops
roughly 90% of the compute budget being spent re-evaluating duplicates, and it
makes the brief SS4 metrics computable on more than one or two molecules.

A genetic algorithm that clones 90% of its offspring is not a naive optimizer,
it is a non-functioning one. The experiment's subject is naive *objective
design*, so the optimizer itself has to actually search.

## 4. Independent-scorer competence gate (brief SS4.2)

Declared in `scripts/train_independent_scorer.py` before the variants were
trained: hERG ROC-AUC >= 0.65, Solubility_AqSolDB MAE <= 1.2. Not a filter --
every variant is trained and reported either way (brief SS9: report the null).
The gate exists so that "excluded from the headline for lack of competence"
would be a pre-registered decision rather than a post-hoc one.

Measured, all six models pass:

| variant | hERG ROC-AUC | solubility MAE | n_train (hERG / sol.) | disjoint from ADMET-AI's training data |
|---|---|---|---|---|
| `model_control` | 0.8437 | 0.7445 | 523 / 7985 | no |
| `data_control_half` | 0.8084 | 0.7794 | 266 / 3993 | no |
| `data_control_test` | 0.8260 | 0.8811 | 132 / 1995 | **yes** |

## 5. Unchanged from Phase 8

AD theta sweep (0.3 / 0.4 / 0.5), SA penalty threshold (6.0), and the budget
(population 200, 100 generations, 5 seeds). The budget is unchanged: the
mutation-rate fix increases the number of *distinct* molecules evaluated within
the same number of evaluations rather than requiring more of them. Any later change to these, or to
anything above, is a new experiment and must be logged as one (brief SS9).

# Limitations

What the measurements in this repository cannot show, and why. Each item
affects how a number in `RESULTS.md` should be read.

## 1. The data-control is not fully disjoint from the objective's training data

Renz et al. (2019)'s data-control requires the objective model and the control
model to be trained on disjoint splits. That cannot be fully realised here.

ADMET-AI is a fixed pretrained artifact fit on the **whole** TDC `ADMET_Group`
`train_val` split, and the brief forbids retraining the objective's predictors.
So no half of `train_val` is disjoint from the objective predictor's training
data. The repository therefore ships three scorer variants rather than two:

| variant | trained on | disjoint from ADMET-AI's training data |
|---|---|---|
| `model_control` | full `train_val` | no -- same data, different architecture |
| `data_control_half` | a deterministic half of `train_val` | **no** |
| `data_control_test` | the TDC `test` split | yes |

`data_control_half` is Renz-shaped, but it is a data control only relative to a
hypothetical half-trained objective. **It must not be described as disjoint from
ADMET-AI's training set**; shared dataset bias remains invisible to it, exactly
as it is to the model-control. A test asserts this limitation explicitly
(`tests/test_independent_scorer.py`), so that the claim cannot later be
"corrected" in the wrong direction.

`data_control_test` is the only genuine data control, and it is trained on far
less data (132 hERG molecules, 1,995 solubility molecules). Its held-out
performance is reported alongside every number it produces.

Scores from the three variants are reported separately and never averaged. If
they disagree with one another, that disagreement is the finding.

## 2. Only two of four learned endpoints are covered

Independent scorers exist for hERG and Solubility_AqSolDB. The objective also
contains **Caco2_Wang** and **CYP3A4_Veith**, for which no independent scorer
exists, plus QED, which is an exact RDKit computation rather than a learned
predictor and so cannot be exploited in the same sense.

Exploitation routed through Caco-2 or CYP3A4 is therefore **invisible to H3**.
To make the size of that blind spot measurable, `RESULTS.md` reports each
component's contribution to the achieved composite (`component_contributions.csv`,
and the `*_log_share` columns of `topk_detail.csv`). A high share on the
uncovered endpoints means the H3 divergence number is measuring a correspondingly
small part of what the optimizer actually did.

## 3. Oracle noise

The independent scorers are themselves imperfect models. Held-out performance:

| variant | hERG ROC-AUC | Solubility MAE (log units) |
|---|---|---|
| `model_control` | 0.8437 | 0.7445 |
| `data_control_half` | 0.8084 | 0.7794 |
| `data_control_test` | 0.8260 | 0.8811 |

Some measured divergence is therefore oracle error rather than exploitation.
Two consequences:

- The H3b decoupling detector triggers on the **change** in the
  objective-minus-independent gap relative to generation 0, not on the raw gap,
  so a constant oracle offset cancels.
- Absolute disagreement values are not interpretable on their own. The
  comparison between the starting population, arm R and the optimized top-k is.

## 4. The applicability domain is measured in the wrong representation

The applicability-domain constraint is computed over ECFP4 Tanimoto similarity,
while the objective's predictors are Chemprop message-passing neural networks.
**A molecule can sit comfortably inside the ECFP4 applicability domain and still
be far outside the region where the GNN is competent.** Arm B and arm C's
constraint is therefore a proxy, and H4's trade-off curve describes the cost of
*that* proxy, not the cost of a representation-matched constraint.

As a partial, representation-matched counterweight, every prediction also
records the spread across the five members of each endpoint's Chemprop ensemble
(`*_ensemble_std`). This is computed from the same forward passes ADMET-AI
already runs; it is a measure of the ensemble's internal disagreement, which is
a weaker signal than a true competence estimate but is at least computed in the
model's own representation.

## 5. The implausibility battery has a declared blind spot

The battery is calibrated so that no more than 5% of approved drugs are flagged
(`docs/PREREGISTRATION.md`). Pre-declared positive controls show it catches
degeneracy driven by size, lipophilicity, flexibility and structural alerts.

It also catches implausible **ionization states** -- carbanions, phosphonium
species, non-aromatic azanides -- through a disqualifying criterion that flags a
molecule on its own rather than contributing one vote toward "fails >= 2". That
criterion was added after a rehearsal showed the counting-only battery reporting
0.000 for every optimized arm; its provenance is disclosed in
`docs/PREREGISTRATION.md`.

It does **not** catch ring-topology degeneracy. A 42-membered all-carbon
macrocycle and a heavily fused polyaromatic each fail only one criterion and are
not flagged, because the frozen criteria contain no ring-count, ring-size or
aromaticity check. Adding more *continuous* criteria was rejected rather than
adopted: at a fixed "fails >= 2" rule each one is defined by an approved-drug
quantile and so flags about alpha of approved drugs by construction, forcing
every other threshold further into the tail to hold the 5% ceiling and making
the battery looser overall. (Cheap *count* criteria over near-always-zero
quantities, like the alert counts and the charge criterion, do not have this
cost -- which is why the charge criterion could be added for free.)

**The reported implausibility rate is therefore a lower bound on implausibility,
not a complete account of it.**

## 6. Population collapse and what the top-k represents

The reported top-k is the `k_top` highest-fitness **distinct** molecules the run
evaluated, keyed on canonical SMILES, not the final population (see
`docs/PREREGISTRATION.md`). Two consequences for reading the numbers:

- The top-k is a "best found" set, not a snapshot of what the optimizer
  converged on. Molecules in it may have been discovered early and then lost
  from the population.
- Convergence is reported separately as `final_population_unique_fraction`. A
  low value means the population collapsed onto a few structures. That is a real
  property of an elitist search and is informative, but it no longer silently
  thins the reported statistics.

Keying on canonical SMILES is load-bearing: SELFIES decoding emits whichever
spelling the token sequence produces, so the same molecule can appear as both an
aromatic and a Kekule form. Without canonicalization it occupies several top-k
slots and inflates every `n` and rank correlation.

Scaffold diversity is a separate matter from molecule count. At reduced
rehearsal scale the top-k shared a single Murcko scaffold even with a healthy
number of distinct molecules, so a small `n_scaffold_clusters` should be read as
a finding about what the search converged on rather than as a defect -- but with
the caveat that at small cluster counts the brief SS5.1 "N recurring motifs"
claim rests on very few groups.

## 7. Scope

Single optimizer family (a SELFIES genetic algorithm), single objective
composition, no docking or structure-based scoring, and no wet-lab or synthesis
validation of any kind. Molecules reported here are artifacts of an adversarial
search procedure and are not drug candidates, hits or leads.

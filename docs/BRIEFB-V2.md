# Project Brief: Reward Hacking in Molecular Multi-Objective Optimization

**Version:** 2 (supersedes v1; see §12 for what changed and why)
**Audience:** an AI coding agent that will decompose this into tasks and implement it.
**Deliverable:** a reproducible research repository. **Code only** — do not write analysis prose, do not write a blog post, do not interpret results beyond emitting the specified machine-readable outputs.
**Scale:** core experiment (§3–§5) is a weekend project. Stretch experiments (§6) are explicitly scoped as extensions and must not block delivery of the core.

---

## 1. Context and hypothesis

Practitioners in computational drug discovery increasingly compose optimization objectives out of several learned property predictors (ADMET endpoints, drug-likeness, affinity proxies) and hand the composite to a generative or search-based optimizer. This is structurally identical to reinforcement learning against a learned reward model, and it is expected to fail the same way: the optimizer finds inputs that maximize the *predictor* rather than the *property the predictor was trained to estimate*.

### 1.1 Prior work — read this before claiming novelty

This failure mode is established in the cheminformatics literature. The repo must not be written as though it were discovered here.

- **Renz et al. (2019), "On failure modes in molecule generation and optimization"** (Drug Discovery Today: Technologies). Optimization scores climb while control scores, computed with models trained on held-out data, stagnate at far lower values. Their experimental design — an optimization model, a *model-control*, and a *data-control* trained on a disjoint split — is the methodological ancestor of this project's H3 and is adopted directly in §4.
- **Langevin et al. (2022), "Explaining and avoiding failure modes in goal-directed generation of small molecules"** (J. Cheminformatics 16:s13321-022-00601-y). Reproduces Renz across Graph GA, SMILES-LSTM and MSO, and analyses causes.
- **Gao & Coley (2020), "The Synthesizability of Molecules Proposed by Generative Models"** (JCIM, arXiv:2002.07007). Quantifies how often generated molecules cannot be readily synthesized, and shows that synthetic-complexity heuristics bias generation toward tractable space but necessarily detract from the primary objective. **H4 below is a replication of this result on a different objective, and must be described as such.**

**What is new here, and the only things that may be framed as contributions:**

1. A per-constraint ablation on a *shared* composite objective, yielding a quantified cost curve (objective score retained vs. implausibility removed) per constraint, rather than a single synthesizability result.
2. Pre-registered implausibility thresholds, calibrated against approved drugs (§4.5), fixed before the first full run.
3. A multi-objective **ADMET triage** objective rather than the single-target activity QSAR attacked in the prior work.
4. Per-generation divergence tracking (§4.3), which prior work reports only as endpoint trajectories of the control scores, not as an operational monitoring signal.

A `docs/RELATED_WORK.md` listing these references must exist in the repo. No result may be described anywhere as a novel finding about optimizer failure.

### 1.2 Hypotheses

**H1 (primary).** A naive gradient-free optimizer maximizing a composite objective built from learned ADMET predictors will converge on molecules that score near-ceiling on the objective while being chemically implausible.

**H2 (mechanism).** The high-scoring molecules will lie measurably further from the predictors' training distribution than the starting population, and the objective score will correlate with that distance.

**H3 (discrimination — this is the critical one).** Independent predictors of the same properties, *not* included in the objective, will disagree with the objective's scores on the optimized molecules far more than they disagree on the starting population. This is what distinguishes genuine reward hacking from a mere claim that "the model went off-distribution."

**H3b (decoupling point).** The objective score and the independent-scorer estimate agree for an initial phase of the search and then decouple at an identifiable generation. The location and sharpness of that decoupling is reportable.

**H4 (mitigation).** Adding an applicability-domain constraint reduces implausibility at a quantifiable cost in achieved objective score, producing a trade-off curve. (Replication of Gao & Coley on a new objective; frame accordingly.)

**A negative result is a valid and publishable outcome.** If H1 fails — if the optimizer produces reasonable molecules — that is a real finding. Do not tune the experiment until it produces degenerate molecules. See §9.

---

## 2. Scope

**In scope:** the optimization experiment, its metrics, its figures, and a reproducible repo.

**Out of scope:** training new property predictors *other than* the independent scorers of §4.2 (permitted exception, cheap, non-negotiable for H3); docking or any structure-based scoring; wet-lab or synthesis validation; benchmarking against other generative models; the blog post itself.

---

## 3. Experimental design

### 3.1 Objective function

Compose a single scalar objective from public pretrained predictors. Use **ADMET-AI** (`pip install admet-ai`; Chemprop models pretrained on all 22 TDC ADMET endpoints) as the objective's predictor source, plus RDKit's QED.

| Component | Source | Direction |
|---|---|---|
| Aqueous solubility (AqSolDB) | ADMET-AI | maximize |
| hERG blocker probability | ADMET-AI | minimize |
| CYP3A4 inhibition probability | ADMET-AI | minimize |
| Caco-2 permeability | ADMET-AI | maximize |
| **QED** | RDKit | maximize |

**QED is mandatory in the default objective.** It is the component most likely to resist degenerate structures, and an objective without a drug-likeness term is both easier to hack and less representative of real triage pipelines. If the agent removes it, the removal must be recorded in the config with a justification and the resulting objective must be run as an **additional** arm, never as a replacement for the default. (v1 of this brief listed QED as adjustable; the implementation dropped it. That is now a deviation requiring explicit sign-off.)

Aggregation: normalize each component to `[0, 1]` using a fixed, config-declared transform (percentile ranks against the DrugBank reference set shipped with ADMET-AI, or explicit min–max bounds — pick one, document it, do not change it between arms). Combine via **weighted geometric mean** so no single component can be fully sacrificed. Weights live in config; default to uniform.

The aggregation function must be identical across all arms *within* a comparison. The aggregation ablation of §6.2 is a separate, clearly-labelled experiment.

### 3.2 Optimizer

A **SELFIES-based genetic algorithm**, implemented in-repo. Rationale: SELFIES guarantees syntactic validity of every mutation, so 100% of candidates are parseable molecules and the experiment cannot be dismissed as an artifact of invalid-SMILES handling. Implementing it directly (rather than pulling a framework) keeps the dependency surface small and makes the "the attack is embarrassingly simple" point stronger.

Specification:

- Representation: SELFIES tokens (`pip install selfies`).
- Operators: token substitution, insertion, deletion. Optional single-point crossover — implement, default off, expose in config.
- Population 200, generations 100, elitism keep-top-10%, tournament selection (k=3). All in config.
- Fitness = the §3.1 composite, plus arm-specific constraints (§3.3).
- Every evaluated molecule is logged, not just survivors. The trajectory is data, and §4.3 depends on it.

### 3.3 Arms

Run all arms with identical objective, budget, and seeds.

| Arm | Description | Constraint added |
|---|---|---|
| **D (control)** | Random sample from starting library, no optimization | — |
| **R (reference)** | Approved drugs: the DrugBank set shipped with ADMET-AI, scored but never optimized | — |
| **A (naive)** | Unconstrained optimization | none |
| **B (AD-constrained)** | + applicability domain | reject/penalize if max ECFP4 Tanimoto to predictor training set < θ |
| **C (AD + synth)** | B's constraint plus synthesizability | SA score ≤ threshold |

**Arm R is new in v2 and is near-free.** It costs one batch of predictions and gives the composite score an interpretable scale. The headline comparison becomes "arm A's molecules score higher on this objective than any approved drug does," which is far more legible than an absolute composite value. Arm R also supplies the calibration set for §4.5.

For arm B, implement the constraint as a **soft penalty** (multiplicative decay below θ) rather than a hard filter, and sweep θ over ≥3 values (e.g. 0.3 / 0.4 / 0.5) to produce H4's trade-off curve. Record θ in the results.

**Applicability-domain caveat (must be recorded in the config comments and in `docs/LIMITATIONS.md`):** the AD is computed over ECFP4, while the objective predictors are Chemprop GNNs. A molecule can sit inside the ECFP4 AD and still be far outside the GNN's region of competence. If ADMET-AI exposes per-endpoint ensemble members, additionally log **ensemble prediction variance** per molecule as a second, representation-matched reliability signal. Check for this before implementing anything custom; if unavailable, record that it was checked.

### 3.4 Starting population and reference sets

- **Starting population:** a drug-like subset sampled from a public library (ChEMBL or ZINC drug-like). Pin the exact source, version, filter criteria, and sampling seed. Cache locally; the repo must not silently re-download different data on re-run.
- **Predictor training reference set:** for the applicability-domain metric, use the TDC training molecules for the endpoints in the objective (retrievable via the `PyTDC` package). Document precisely which molecules constitute the reference set — this metric is meaningless if the reference is vague.
- **Approved-drug reference set:** the DrugBank set shipped with ADMET-AI. Used for arm R and for threshold calibration (§4.5). Pin and hash it like any other data artifact.

### 3.5 Seeds

**5 seeds per arm, minimum.** Every reported number carries a mean and a spread across seeds. Single-seed results are not acceptable output.

---

## 4. Metrics

Computed for the top-k molecules per arm per seed (k=100, configurable) and for the control and reference arms.

### 4.1 Objective-side

- Achieved composite score: trajectory per generation, and final top-k distribution.
- Per-component scores of the top-k, so it is visible which component is being exploited hardest.
- **Percentile of each arm's top-k composite score within the arm R (approved drug) distribution.**

### 4.2 Independent-scorer divergence (H3) — highest priority metric

v1 permitted a single in-repo baseline. That is insufficient: a LightGBM model trained on the *same* TDC `train_val` split as ADMET-AI is a **model-control** only. It differs in architecture but shares the dataset, so any bias in the data is invisible to it. Following Renz, implement both:

- **Model-control:** distinct architecture (LightGBM on RDKit descriptors + MACCS keys), same training data as the objective predictor.
- **Data-control:** the *same* LightGBM recipe, trained on a disjoint half of the TDC `train_val` split, with the objective-side half excluded. Split deterministically by a recorded seed; assert zero SMILES overlap between halves in a test.

The data-control is the load-bearing scorer. Report both; if they diverge from each other, say so rather than averaging them.

**Endpoint coverage is partial and must be stated.** Independent scorers exist for hERG and Solubility_AqSolDB only. Caco-2 and CYP3A4 are uncovered, so exploitation routed through those components is invisible to H3. Two requirements:

1. Report per-component contribution to the final composite for arm A, so the reader can see how much of the achieved score came through uncovered endpoints.
2. `docs/LIMITATIONS.md` must state this explicitly.

Report: rank correlation and mean absolute disagreement between objective scorer and each independent scorer, computed separately on (a) the starting population, (b) arm R, and (c) the optimized top-k. The *difference* between (a) and (c) is the headline number of the whole project.

### 4.3 Decoupling trajectory (H3b) — new in v2

Compute the H3 divergence **per generation**, not only on the final top-k. Score each generation's elite set with the independent scorers and plot objective score and independent-scorer estimate on the same axis over generations.

This is cheap (the molecules are already logged and predictions are cached) and it is the only output of this project with direct operational value: it indicates whether divergence onset is detectable mid-run, which would make it usable as an early-stopping or monitoring signal. Report the generation at which the two series separate by more than a pre-declared margin, per seed, with spread.

### 4.4 Distribution shift (H2)

- Max ECFP4 (r=2, 2048-bit) Tanimoto similarity to nearest molecule in the predictor reference set.
- Same, against the starting population.
- Ensemble prediction variance per molecule, if available per §3.3.

### 4.5 Chemical plausibility and the implausibility battery

Panel: MW, cLogP, TPSA, HBD/HBA, rotatable bonds, ring count, largest ring size, fraction sp3, formal charge, longest aliphatic chain, SA score (RDKit `sascorer`), PAINS and Brenk structural alerts (RDKit FilterCatalog).

**Implausibility rate:** *fails ≥2 of a pre-declared battery.*

**Calibration requirement (new in v2, blocking).** Thresholds must be calibrated against arm R before being frozen. Choose thresholds such that **no more than 5% of the approved-drug reference set is flagged implausible**. A battery that most real drugs also fail makes every downstream number uninterpretable, which is the mirror-image failure of post-hoc threshold picking.

Calibration uses arm R only. It must occur before any arm A molecule is generated, and the calibration script, its output, and the resulting frozen config must all be committed in one commit, with the git SHA recorded. Emit `results/tables/battery_calibration.csv` showing the flag rate of each individual criterion on arm R.

---

## 5. Outputs

```
results/
  raw/            # every evaluated molecule, all arms/seeds/generations (parquet)
  tables/
    battery_calibration.csv           # §4.5, flag rate per criterion on arm R
    *.csv                             # per-arm summaries, all metrics, mean ± sd
  figures/
    fig1_structure_grid.(png|svg)        # top-20 arm A vs arm B, scores annotated
    fig2_score_vs_nn_distance.(png|svg)  # scatter, colored by arm
    fig3_objective_vs_independent.(png|svg)  # H3, start pop vs arm R vs optimized
    fig4_ad_tradeoff.(png|svg)           # achieved score vs θ, with implausibility rate
    fig5_trajectories.(png|svg)          # score per generation, seed spread shaded
    fig6_decoupling.(png|svg)            # H3b: objective vs independent score per generation
    fig7_exploit_clusters.(png|svg)      # §5.1, scaffold clusters of arm A top-k
  RESULTS.md      # generated tables + figure references ONLY. No prose interpretation.
docs/
  RELATED_WORK.md   # §1.1 references
  LIMITATIONS.md    # AD representation mismatch, endpoint coverage gap, oracle noise
```

`RESULTS.md` is a machine-generated artifact: numbers, tables, figure embeds. The agent must not write conclusions, narrative, or claims into it.

Figure 1 is the headline. Render clean 2D structures via RDKit with the composite score, the independent-scorer score, **and the arm R percentile** annotated on each. Make it publication-legible at blog width.

### 5.1 Exploit characterization (new in v2)

Cluster arm A's top-k by Murcko scaffold, and separately group by which component dominates the achieved score. Emit counts per cluster and a representative structure per cluster.

This is a `groupby` over data already in `results/raw/`. It converts the finding from a scalar ("implausibility rate 0.78 ± 0.06") into a structural claim ("the search converged on N recurring motifs"), which is both more informative and far more legible.

---

## 6. Stretch experiments — explicitly out of the weekend core

Implement only after §10's definition of done is fully met. Both are pre-declared here so that running them cannot be mistaken for the result-tuning forbidden in §9.

### 6.1 Iterated patching ("whack-a-mole")

Run arm A, identify the dominant exploit motif from §5.1, add one targeted patch to the objective (a structural alert term, a component clamp, an extra penalty), re-run, repeat for ≥3 rounds. Report per round: the motif patched, the objective cost of the patch on arm R, generations until a new dominant exploit emerges, and whether implausibility rate recovers.

This is the closest chemical analogue to patching a reward model. It is inherently **sequential** — each round depends on inspecting the previous one — and is the long pole of the project. It is a legitimate standalone follow-up if time runs out.

Each round is a separate pre-declared arm with its own config and run log. The §9 prohibition on tuning applies to the *core* arms and is not waived here; what is permitted is this declared protocol, not ad-hoc adjustment of arm A.

### 6.2 Aggregation ablation

Re-run arm A with arithmetic mean, weighted geometric mean, and soft-min aggregation, holding everything else fixed. Report achieved score, implausibility rate, and H3 divergence per aggregation function.

The geometric mean was chosen in §3.1 specifically so that no component can be fully sacrificed. This tests whether that choice actually buys anything, and a positive result yields a concrete, transportable design recommendation. One config knob plus two extra runs; parallel with the core batch if compute allows.

### 6.3 Explicitly deprioritized

A second optimizer (hill-climbing, MolScore GuacaMol wrapper) to show the finding is not GA-specific. Prior work already covers three optimizer families. Implement only if everything above is complete and time remains.

---

## 7. Engineering requirements

- Python pinned in `pyproject.toml` (currently `>=3.13,<3.14`). Dependencies pinned via `uv` lockfile. `rdkit`, `admet-ai`, `selfies`, `PyTDC`, `numpy`, `pandas`, `pyarrow`, `matplotlib`, `lightgbm`.
  - Note: `PyTDC` pins `rdkit<2024.3.1`, incompatible with `admet-ai`'s `rdkit` requirement. TDC-touching scripts run in an isolated environment; keep this separation.
- **Config-driven.** One YAML config per experiment run. No hardcoded constants in logic. Config is written into the results directory alongside outputs.
- **One-command reproduction:** `make reproduce` runs all core arms, all seeds, regenerates every table and figure from scratch.
- Deterministic given a seed. Seed numpy, python `random`, and any model inference path.
- Cache ADMET-AI predictions keyed by canonical SMILES — the GA will re-evaluate duplicates constantly and this is the main runtime cost. §4.3 relies on this cache making per-generation rescoring cheap.
- Structured run logs: config hash, git SHA, timestamp, wall time, molecules evaluated.
- Tests: unit tests for the SELFIES mutation operators (validity preserved), the objective aggregation (known inputs → known outputs), the Tanimoto/AD calculation, and **a test asserting zero SMILES overlap between the objective-side and data-control training splits**. A fast smoke-test config (pop 20, gens 5) that runs in under a minute in CI.
- README: what this is, how to run it, what each config knob does. Factual, no argumentation.

---

## 8. Task decomposition

Phases 1–6 of v1 are complete (scaffold, data layer, objective module, SELFIES GA, constraint modules, independent scorer). The remaining and revised work:

**Blocking pre-run work — must all land before the first core run, or the run is wasted:**

1. **Restore QED** to the default objective, or record the deviation per §3.1.
2. **Data-control scorer:** split TDC `train_val` deterministically, retrain the LightGBM recipe on the disjoint half, commit frozen artifacts alongside the existing model-control. Add the overlap assertion test.
3. **Arm R:** score the DrugBank reference set with the objective and both independent scorers.
4. **Battery calibration (§4.5):** calibrate against arm R, freeze thresholds, commit with calibration output. This is the Phase 8 pre-registration sign-off.
5. **Ensemble-variance check (§3.3):** determine whether ADMET-AI exposes per-member predictions; implement or record as unavailable.
6. **Aggregation knob** exposed in config (even if §6.2 is deferred, the knob is cheap now and expensive later).

**Core batch run (parallel once the above is frozen):**

7. Arms D/R/A/B/C × 5 seeds × θ sweep.

**Post-run analysis (all from existing logs):**

8. Metrics module: §4 in full, including per-generation divergence (§4.3) and exploit clustering (§5.1).
9. Figures 1–7.
10. `RESULTS.md` generation, `RELATED_WORK.md`, `LIMITATIONS.md`.
11. Determinism verification: re-run one arm, same seed, identical top-k.

**Stretch (sequential, after the above):**

12. §6.1 iterated patching, §6.2 aggregation ablation.

---

## 9. Scientific-integrity guardrails

These are requirements, not suggestions. The value of this project is entirely in its credibility.

- **Pre-declare thresholds.** The implausibility battery, AD thresholds, and SA cutoff go into config before the first full run, and the battery is calibrated per §4.5. Record any post-hoc change explicitly in the run log with a reason.
- **Calibrate, don't just freeze.** A pre-declared battery that flags most approved drugs is as worthless as a post-hoc one. §4.5 is blocking.
- **No cherry-picking.** Figure 1 shows the top-20 by objective score, unmodified. Not a hand-picked selection of the funniest molecules.
- **Report the null.** If arm A does not degenerate, output that faithfully. Do not increase the generation budget, reweight components, or swap endpoints in order to produce a more dramatic result. Any such change is a new experiment and must be logged as one. The §6 stretch experiments are pre-declared protocols, not exceptions to this rule.
- **Do not conflate off-distribution with bad.** H2 alone proves nothing — every optimizer leaves the training distribution. The H3 independent-scorer divergence, and specifically the *data*-control, is the load-bearing evidence. If H2 holds but H3 does not, the correct output is "distribution shift without demonstrable scorer exploitation."
- **State the oracle's limits.** The independent scorers are themselves imperfect models (hERG ROC-AUC ≈ 0.84, solubility MAE ≈ 0.75) covering two of four endpoints. Some measured divergence is oracle error. `docs/LIMITATIONS.md` must say so.
- **Log failed and invalid candidates,** don't silently drop them.

---

## 10. Definition of done (core)

- [ ] QED restored or deviation signed off and run as an additional arm.
- [ ] Both model-control and data-control independent scorers trained, frozen, committed; split-overlap test passes.
- [ ] Battery calibrated against arm R at ≤5% flag rate, frozen, committed with calibration table and git SHA.
- [ ] `make reproduce` runs end-to-end from a clean environment.
- [ ] All core arms (D, R, A, B, C) × 5 seeds complete; θ swept over ≥3 values for arm B.
- [ ] All seven figures generated and legible at blog width.
- [ ] `RESULTS.md` populated with mean ± sd for every §4 metric.
- [ ] H3 divergence reported for starting population, arm R, and optimized top-k, for both independent scorers.
- [ ] H3b decoupling generation reported per seed with spread.
- [ ] Exploit clusters emitted with counts and representative structures.
- [ ] `RELATED_WORK.md` and `LIMITATIONS.md` present and complete.
- [ ] Unit tests pass; smoke config runs under 60s.
- [ ] Determinism verified: one arm re-run with the same seed reproduces identical top-k.
- [ ] No interpretive prose anywhere in the outputs.

---

## 11. Framing constraints (for any text the agent does emit — README, docstrings, log messages, `RELATED_WORK.md`, `LIMITATIONS.md`)

- This project critiques **naive composite objective design**, not ADMET-AI or TDC. Both are well-built resources being deliberately used outside their intended operating regime. Any text that reads as an attack on a specific tool or its authors is incorrect output.
- **Do not present H1, H2 or H4 as novel.** §1.1 establishes the prior art. Any text implying this project discovered that optimizers exploit learned predictors is incorrect output.
- Generated molecules are **artifacts of an adversarial search procedure**. Never describe, label, or annotate them as drug candidates, hits, leads, or anything implying therapeutic relevance.
- No claims about clinical, toxicological, or regulatory implications anywhere in the repo.

---

## 12. Changelog: v1 → v2

| Change | Why |
|---|---|
| §1.1 prior-work section added; novelty claims restricted | H1/H2/H4 are established results (Renz 2019, Langevin 2022, Gao & Coley 2020). Unattributed framing would sink the project's credibility with chemists. |
| H3 split into model-control **and data-control** scorers | A LightGBM trained on the same TDC split differs only in architecture, so shared dataset bias is invisible. The data-control is what makes the argument, per Renz's original design. |
| Endpoint-coverage gap made explicit | Independent scorers cover 2 of 4 endpoints; exploitation via Caco-2 or CYP3A4 is currently undetectable. |
| QED restored as mandatory | The implementation dropped it. It is the component most resistant to degenerate structures; without it the hack is easier and the setup less realistic. |
| Arm R (approved drugs) added | Near-free. Gives the composite score an interpretable scale and supplies the calibration set. |
| Battery calibration (≤5% flag rate on arm R) made blocking | Pre-declaring thresholds blind risks a battery most real drugs also fail, which makes every implausibility number meaningless. |
| H3b per-generation decoupling added (§4.3, fig6) | Cheap given existing logs; the only output with direct operational value as a monitoring signal. |
| Exploit clustering added (§5.1, fig7) | Converts a scalar rate into a structural claim; `groupby` over existing data. |
| AD representation mismatch documented; ensemble variance added if available | ECFP4 AD does not describe where a Chemprop GNN is reliable. |
| §6 stretch section: iterated patching, aggregation ablation; second optimizer deprioritized | Separates the weekend core from the genuinely novel but sequential extensions, and pre-declares them so they don't collide with §9. |
| Python version corrected to match `pyproject.toml` | v1 said 3.11; the repo pins `>=3.13,<3.14`. |
| `docs/RELATED_WORK.md`, `docs/LIMITATIONS.md` added to required outputs | Credibility requirements, not optional extras. |
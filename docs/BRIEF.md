# Project Brief: Reward Hacking in Molecular Multi-Objective Optimization

**Audience:** an AI coding agent that will decompose this into tasks and implement it.
**Deliverable:** a reproducible research repository. **Code only** — do not write analysis prose, do not write a blog post, do not interpret results beyond emitting the specified machine-readable outputs.
**Scale:** weekend project. Single machine, CPU-sufficient, no training of new models.

---

## 1. Context and hypothesis

Practitioners in computational drug discovery increasingly compose optimization objectives out of several learned property predictors (ADMET endpoints, drug-likeness, affinity proxies) and hand the composite to a generative or search-based optimizer. This is structurally identical to reinforcement learning against a learned reward model, and it is expected to fail the same way: the optimizer finds inputs that maximize the *predictor* rather than the *property the predictor was trained to estimate*.

**H1 (primary).** A naive gradient-free optimizer maximizing a composite objective built from learned ADMET predictors will converge on molecules that score near-ceiling on the objective while being chemically implausible.

**H2 (mechanism).** The high-scoring molecules will lie measurably further from the predictors' training distribution than the starting population, and the objective score will correlate with that distance.

**H3 (discrimination — this is the critical one).** Independent predictors of the same properties, *not* included in the objective, will disagree with the objective's scores on the optimized molecules far more than they disagree on the starting population. This is what distinguishes genuine reward hacking from a mere claim that "the model went off-distribution."

**H4 (mitigation).** Adding an applicability-domain constraint reduces implausibility at a quantifiable cost in achieved objective score, producing a trade-off curve.

**A negative result is a valid and publishable outcome.** If H1 fails — if the optimizer produces reasonable molecules — that is a real finding. Do not tune the experiment until it produces degenerate molecules. See §8.

---

## 2. Scope

**In scope:** the optimization experiment, its metrics, its figures, and a reproducible repo.

**Out of scope:** training new property predictors; docking or any structure-based scoring; wet-lab or synthesis validation; benchmarking against other generative models; the blog post itself.

---

## 3. Experimental design

### 3.1 Objective function

Compose a single scalar objective from public pretrained predictors. Use **ADMET-AI** (`pip install admet-ai`; Chemprop models pretrained on all 22 TDC ADMET endpoints) as the objective's predictor source, plus RDKit's QED.

Select a small, defensible endpoint set that mirrors a realistic early-stage triage objective. Suggested (agent may adjust with justification recorded in the config):

| Component | Source | Direction |
|---|---|---|
| Aqueous solubility (AqSolDB) | ADMET-AI | maximize |
| hERG blocker probability | ADMET-AI | minimize |
| CYP3A4 inhibition probability | ADMET-AI | minimize |
| Caco-2 permeability | ADMET-AI | maximize |
| QED | RDKit | maximize |

Aggregation: normalize each component to `[0, 1]` using a fixed, config-declared transform (percentile ranks against the DrugBank reference set shipped with ADMET-AI, or explicit min–max bounds — pick one, document it, do not change it between arms). Combine via **weighted geometric mean** so no single component can be fully sacrificed. Weights live in config; default to uniform.

The aggregation function must be identical across all arms. Any change invalidates cross-arm comparison.

### 3.2 Optimizer

A **SELFIES-based genetic algorithm**, implemented in-repo. Rationale: SELFIES guarantees syntactic validity of every mutation, so 100% of candidates are parseable molecules and the experiment cannot be dismissed as an artifact of invalid-SMILES handling. Implementing it directly (rather than pulling a framework) keeps the dependency surface small and makes the "the attack is embarrassingly simple" point stronger.

Specification:

- Representation: SELFIES tokens (`pip install selfies`).
- Operators: token substitution, insertion, deletion. Optional single-point crossover — implement, default off, expose in config.
- Population 200, generations 100, elitism keep-top-10%, tournament selection (k=3). All in config.
- Fitness = the §3.1 composite, plus arm-specific constraints (§3.3).
- Every evaluated molecule is logged, not just survivors. The trajectory is data.

**Optional secondary optimizer, only if time permits:** repeat arm A with a second search strategy (e.g. hill-climbing, or MolScore's GuacaMol goal-directed wrapper) to show the finding is not GA-specific. Mark clearly as optional; do not block delivery on it.

### 3.3 Arms

Run all arms with identical objective, budget, and seeds.

| Arm | Description | Constraint added |
|---|---|---|
| **D (control)** | Random sample from starting library, no optimization | — |
| **A (naive)** | Unconstrained optimization | none |
| **B (AD-constrained)** | + applicability domain | reject/penalize if max ECFP4 Tanimoto to predictor training set < θ |
| **C (AD + synth)** | + synthesizability | B's constraint plus SA score ≤ threshold |

For arm B, implement the constraint as a **soft penalty** (multiplicative decay below θ) rather than a hard filter, and sweep θ over ≥3 values (e.g. 0.3 / 0.4 / 0.5) to produce H4's trade-off curve. Record θ in the results.

### 3.4 Starting population and reference sets

- **Starting population:** a drug-like subset sampled from a public library (ChEMBL or ZINC drug-like). Pin the exact source, version, filter criteria, and sampling seed. Cache locally; the repo must not silently re-download different data on re-run.
- **Predictor training reference set:** for the applicability-domain metric, use the TDC training molecules for the endpoints in the objective (retrievable via the `PyTDC` package). Document precisely which molecules constitute the reference set — this metric is meaningless if the reference is vague.

### 3.5 Seeds

**5 seeds per arm, minimum.** Every reported number carries a mean and a spread across seeds. Single-seed results are not acceptable output.

---

## 4. Metrics

Computed for the top-k molecules per arm per seed (k=100, configurable) and for the control.

**Objective-side**

- Achieved composite score: trajectory per generation, and final top-k distribution.
- Per-component scores of the top-k, so it is visible which component is being exploited hardest.

**Distribution shift (H2)**

- Max ECFP4 (r=2, 2048-bit) Tanimoto similarity to nearest molecule in the predictor reference set.
- Same, against the starting population.

**Independent-scorer divergence (H3) — highest priority metric**

- Score the top-k with predictors of the *same* properties that are **not** part of the objective. Options, in order of preference:
  1. A held-out ADMET-AI-independent model for at least one endpoint (e.g. a separately trained/obtained hERG or solubility model).
  2. Failing that, retrain a simple, clearly-distinct baseline in-repo (RDKit descriptors + LightGBM on the same TDC endpoint) and use *that* as the independent scorer. This is a permitted exception to "no training" — it is cheap and the experiment's central claim depends on it.
- Report: rank correlation and mean absolute disagreement between objective scorer and independent scorer, computed separately on (a) the starting population and (b) the optimized top-k. The *difference* between (a) and (b) is the headline number of the whole project.

**Chemical plausibility**

- Physchem panel: MW, cLogP, TPSA, HBD/HBA, rotatable bonds, ring count, largest ring size, fraction sp3, formal charge, longest aliphatic chain.
- SA score (RDKit `sascorer`).
- PAINS and Brenk structural alerts (RDKit FilterCatalog).
- **Implausibility rate:** operationalize as *fails ≥2 of a pre-declared battery*. Declare the battery in config **before running** — thresholds chosen post hoc are not evidence.

---

## 5. Outputs

```
results/
  raw/            # every evaluated molecule, all arms/seeds/generations (parquet)
  tables/         # per-arm summary CSVs, all metrics, mean ± sd across seeds
  figures/
    fig1_structure_grid.(png|svg)     # top-20 arm A vs arm B, scores annotated
    fig2_score_vs_nn_distance.(png|svg)  # scatter, colored by arm
    fig3_objective_vs_independent.(png|svg)  # H3, start pop vs optimized
    fig4_ad_tradeoff.(png|svg)        # achieved score vs θ, with implausibility rate
    fig5_trajectories.(png|svg)       # score per generation, seed spread shaded
  RESULTS.md      # generated tables + figure references ONLY. No prose interpretation.
```

`RESULTS.md` is a machine-generated artifact: numbers, tables, figure embeds. The agent must not write conclusions, narrative, or claims into it.

Figure 1 is the headline. Render clean 2D structures via RDKit with the composite score and the independent-scorer score annotated on each. Make it publication-legible at blog width.

---

## 6. Engineering requirements

- Python 3.11. Dependencies pinned via `uv` or `pip-tools` lockfile. `rdkit`, `admet-ai`, `selfies`, `PyTDC`, `numpy`, `pandas`, `pyarrow`, `matplotlib`, `lightgbm`.
- **Config-driven.** One YAML config per experiment run. No hardcoded constants in logic. Config is written into the results directory alongside outputs.
- **One-command reproduction:** `make reproduce` (or equivalent) runs all arms, all seeds, regenerates every table and figure from scratch.
- Deterministic given a seed. Seed numpy, python `random`, and any model inference path.
- Cache ADMET-AI predictions keyed by canonical SMILES — the GA will re-evaluate duplicates constantly and this is the main runtime cost.
- Structured run logs: config hash, git SHA, timestamp, wall time, molecules evaluated.
- Tests: unit tests for the SELFIES mutation operators (validity preserved), the objective aggregation (known inputs → known outputs), and the Tanimoto/AD calculation. A fast smoke-test config (pop 20, gens 5) that runs in under a minute in CI.
- README: what this is, how to run it, what each config knob does. Factual, no argumentation.

---

## 7. Suggested task decomposition

1. Repo scaffold, lockfile, config schema, smoke-test harness.
2. Data layer: starting-population sampling, TDC reference-set retrieval, caching.
3. Objective module: ADMET-AI wrapper, normalization, aggregation, prediction cache.
4. SELFIES GA: representation, operators, selection loop, full-trajectory logging. Unit tests.
5. Constraint modules: applicability domain (soft penalty), SA score.
6. Independent scorer: held-out model or in-repo LightGBM baseline.
7. Metrics module: all of §4.
8. Runner: arms × seeds × θ sweep, orchestration, artifact writing.
9. Figures.
10. `RESULTS.md` generation.
11. Full reproduction run, verify determinism by re-running one arm.

---

## 8. Scientific-integrity guardrails

These are requirements, not suggestions. The value of this project is entirely in its credibility.

- **Pre-declare thresholds.** The implausibility battery, AD thresholds, and SA cutoff go into config before the first full run. Record any post-hoc change explicitly in the run log with a reason.
- **No cherry-picking.** Figure 1 shows the top-20 by objective score, unmodified. Not a hand-picked selection of the funniest molecules.
- **Report the null.** If arm A does not degenerate, output that faithfully. Do not increase the generation budget, reweight components, or swap endpoints in order to produce a more dramatic result. Any such change is a new experiment and must be logged as one.
- **Do not conflate off-distribution with bad.** H2 alone proves nothing — every optimizer leaves the training distribution. The H3 independent-scorer divergence is the load-bearing evidence. If H2 holds but H3 does not, the correct output is "distribution shift without demonstrable scorer exploitation."
- **Log failed and invalid candidates,** don't silently drop them.

---

## 9. Framing constraints (for any text the agent does emit — README, docstrings, log messages)

- This project critiques **naive composite objective design**, not ADMET-AI or TDC. Both are well-built resources being deliberately used outside their intended operating regime. Any text that reads as an attack on a specific tool or its authors is incorrect output.
- Generated molecules are **artifacts of an adversarial search procedure**. Never describe, label, or annotate them as drug candidates, hits, leads, or anything implying therapeutic relevance.
- No claims about clinical, toxicological, or regulatory implications anywhere in the repo.

---

## 10. Definition of done

- [ ] `make reproduce` runs end-to-end from a clean environment.
- [ ] All 4 arms × 5 seeds complete; θ swept over ≥3 values for arm B.
- [ ] All five figures generated and legible at blog width.
- [ ] `RESULTS.md` populated with mean ± sd for every §4 metric.
- [ ] H3 divergence number reported for both starting population and optimized top-k.
- [ ] Unit tests pass; smoke config runs under 60s.
- [ ] Determinism verified: one arm re-run with the same seed reproduces identical top-k.
- [ ] No interpretive prose anywhere in the outputs.

---

## Reference links

- ADMET-AI: [paper](https://academic.oup.com/bioinformatics/article/40/7/btae416/7698030) · [GitHub](https://github.com/swansonk14/admet_ai) · [PyPI](https://pypi.org/project/admet-ai/)
- Therapeutics Data Commons: [site](https://tdcommons.ai/) · [docs](https://tdc.readthedocs.io/)
- MolScore (optional secondary optimizer): [paper](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00861-w)
- SELFIES: [PyPI](https://pypi.org/project/selfies/)

# Implementation Plan

Working plan for the project described in [Project-Brief.md](Project-Brief.md). Engineering document; not an experiment output.

## Environment facts (probed)

- macOS, arm64 (Apple Silicon), 14 cores, 36 GiB RAM. No CUDA GPU; MPS available.
- `python3.11` present at `/opt/homebrew/bin/python3.11`; system default is 3.13.
- `admet-ai` current release is 2.0.1 (Chemprop v2, requires Python >= 3.11). The brief was written against v1 semantics; v2 predictions differ from the paper's. Version is pinned in the lockfile and recorded in config.

## Phase 0 results (completed)

Environment (deviations from the brief, recorded deliberately):

- **Python 3.13.2**, not 3.11 (user's existing `.venv-c2`). Full stack installs and imports.
- Deps managed with `uv` (`uv` installed into the venv itself); `uv.lock` generated for macOS and Linux. The project targets Python 3.13; uv selects a managed 3.13 interpreter when the host default is newer.
- Installed: admet-ai 2.0.1, chemprop 2.2.4, torch 2.14.0, rdkit 2026.3.6, selfies 2.2.0, lightgbm 4.7.0, numpy 2.5.3, pandas 3.0.5, pyarrow 25.0.1, matplotlib 3.11.1, scikit-learn 1.9.0.
- LightGBM needs `brew install libomp` on macOS (done); record in README.
- **PyTDC cannot share the experiment environment**: `admet-ai>=2.0.1` requires `rdkit>=2025.9.5`, `PyTDC==1.1.15` pins `rdkit<2024.3.1`. TDC is therefore run once in an isolated pinned env (`uv run --isolated --no-project --python 3.11 --with PyTDC==1.1.15 --with 'pandas<3' --with 'setuptools<81'`; the `setuptools` pin is needed because TDC imports `pkg_resources`). Its output is cached as CSV and the experiment env never imports TDC.
- **Linux/Codespaces check:** uv resolves the experiment environment successfully on Linux x86_64 without Homebrew or MPS. A Codespace with Python 3.14 must still use uv's Python 3.13 interpreter because the project requirement is `>=3.13,<3.14`.

Measured ADMET-AI throughput (batch of 512, single process, 14-core M-series):

| Configuration | mol/s |
|---|---|
| CPU, physchem + DrugBank percentiles (default) | 165 |
| CPU, physchem, no DrugBank percentiles | 199 |
| **CPU, no physchem, no DrugBank percentiles** | **390** |
| MPS, physchem, no DrugBank percentiles | 125 |

Derived configuration decisions:

- **CPU, not MPS** — MPS is ~40% slower here.
- `ADMETModel(include_physchem=False, drugbank_path=None)`; QED computed directly with RDKit (1819 mol/s) and DrugBank percentiles computed in-repo for only the 5 objective components from the shipped `drugbank_approved.csv` (2845 approved drugs, contains all 5 columns). Same semantics as ADMET-AI's own percentile path, ~2x faster.
- **`ADMETModel.predict()` silently drops invalid SMILES** (verified: 4 in, 2 out, no exception) and returns a SMILES-indexed frame. The wrapper must realign on the input list and log dropped candidates, per brief §8.
- Lightning's progress bar is hardcoded `enable_progress_bar=True` and writes to **stdout**; must be suppressed (redirect + logger level) or it corrupts piped output.
- SELFIES round-trip on DrugBank: 998/1000 valid, 2 encoder errors, 4025 mol/s.

**Budget: GO.** 5 optimization configs (A, B@3 θ, C) × 5 seeds × 200 pop × 100 gens ≈ 500k evaluations; at ~390 mol/s before cache dedup that is well under an hour of predictor time. The brief's 200×100 budget stands unchanged.

Reference set exported to `data/reference/` (TDC `ADMET_Group` `train_val` splits, with per-file sha256 in `manifest.json`): Caco2_Wang 728, Solubility_AqSolDB 7985, hERG 523, CYP3A4_Veith 9861 → **18,856 unique molecules**.

## Phase 0 — De-risking spike — DONE

See "Phase 0 results" above. Artifacts: `pyproject.toml`, `uv.lock`, `.gitignore`, `scripts/spike_admet.py`, `scripts/spike_admet_variants.py`, `scripts/export_tdc_reference.py`, `data/reference/`.

## Phase 1 — Scaffold — DONE

- `git init`, `.gitignore` (venv, caches, `data/raw`, large artifacts).
- `pyproject.toml` + `uv.lock`; `src/` layout package (e.g. `rhmoo/`).
- Config schema (dataclass or pydantic) + `configs/smoke.yaml` (pop 20, gens 5) and `configs/main.yaml`. No constants in logic.
- `Makefile`: `setup`, `test`, `smoke`, `reproduce`, `figures`, `results`.
- Structured run logging: config hash, git SHA, timestamp, wall time, molecules evaluated → JSONL.
- `pytest` harness + CI-style smoke target that must finish under 60s.

Artifacts: `src/rhmoo/{config,run_log,runner}.py`, `configs/{smoke,main}.yaml`, `Makefile`, `tests/test_{config,run_log,runner}.py`.

## Phase 2 — Data layer — DONE

- **Starting population.** Decision: GuacaMol v1 training set (ChEMBL-derived, pre-filtered for drug-likeness by GuacaMol's own curation). Pinned by URL + MD5 (verified against the value published in the GuacaMol README). `scripts/build_starting_population.py` downloads it (cached under `data/raw/guacamol/`, skipped on re-run if the checksum already matches), draws 10,000 molecules without replacement with a fixed seed (0), canonicalizes and deduplicates with RDKit, and writes `data/processed/starting_population.csv` + `starting_population_manifest.json` (source URL/MD5, pool size, sampled count, dropped-invalid count, seed, filter criteria, output hash). Verified deterministic across re-runs (identical output SHA-256).
- **Predictor reference set.** DONE in Phase 0 — `data/reference/` holds the TDC `ADMET_Group` `train_val` molecules for exactly the four objective endpoints, plus a sha256 manifest. Folded into the Makefile (`make data-reference`, isolated PyTDC env) and `make data-starting-population`; both run via `make data`. Provenance documented in `README.md`.

## Phase 3 — Objective module — DONE

- **`src/rhmoo/predictors.py`**: `ADMETPredictor` wraps `ADMETModel(include_physchem=False, drugbank_path=None)` (Phase 0 config) plus RDKit QED. Realigns every prediction onto the requested SMILES list (canonicalizes, dedupes queries, logs invalid/dropped counts via `logging` rather than silently dropping — brief §8). `PredictionCache` is a canonical-SMILES-keyed, parquet-persisted cache (`objective.cache_path` in config) shared across calls, so GA duplicate re-evaluation is cheap.
- **`src/rhmoo/normalization.py`**: `DrugBankPercentileNormalizer` — direction-aware percentile rank (mean-rank tie handling) against a reference distribution; `from_admet_ai_drugbank()` sources it from `admet_ai`'s shipped `drugbank_approved.csv`, which already carries all 5 objective columns (`QED, CYP3A4_Veith, hERG, Caco2_Wang, Solubility_AqSolDB`) precomputed for 2,845 approved drugs — same reference set decided in Phase 0, no extra fitting needed.
- **`src/rhmoo/objective.py`**: `weighted_geometric_mean` (zero-component correctly zeros the result; validates weight/score domains), `normalize_components`, `aggregate` (dispatches on `config.aggregation`, currently only `weighted_geometric_mean`).
- Unit tests (`tests/test_normalization.py`, `tests/test_objective.py`, both `smoke`-marked, pure numpy/pandas, no model calls): golden-value percentile ranks, direction handling, geometric-mean golden values and zero-component behaviour, mismatched-key/negative-score/unsupported-aggregation error paths, and an end-to-end normalize+aggregate check. `tests/test_predictors.py` (not smoke-marked — loads the real Chemprop checkpoint) covers canonicalization/QED, invalid-SMILES realignment, dedup, and cross-instance persistent-cache reuse. Full suite: 23 tests, ~8.5s.

## Phase 4 — SELFIES GA — DONE

- **`src/rhmoo/selfies_ops.py`**: tokenize/detokenize via `selfies.split_selfies`/`decoder`; mutation operators (substitute/insert/delete) drawing replacement symbols from `selfies.get_semantic_robust_alphabet()`; single-point crossover (implemented, wired through `GAConfig.crossover_enabled`/`crossover_rate`, default off in both configs). Verified property: any mutation/crossover of a realistic (multi-token) molecule decodes to a valid RDKit mol — the one documented exception is substituting the sole token of a one-heavy-atom molecule with a ring/branch-only symbol, which can decode to the empty molecule; this is treated as an ordinary invalid candidate, not a bug.
- **`src/rhmoo/ga.py`**: `SelfiesGA` — tournament selection (k configurable), elitism (top `elitism_fraction`, carried over without re-evaluation), crossover then mutation (each independently gated by its own config rate), invalid offspring logged (generation, parents, operator, `fitness=NaN`, `valid=False`) and retried rather than silently dropped, so the final population always has exactly the requested size. A single `random.Random(seed)` drives every stochastic choice for full (arm, seed) reproducibility. Fitness is injected via a `fitness_fn: list[str] -> DataFrame` callback (must include a `fitness` column; any other columns are propagated into the trajectory log verbatim) so this module has no ADMET-AI dependency and stays fast to test; Phase 9's runner wires in the real objective + constraints (Phase 5) as that callback. Every evaluated molecule (not just survivors) is appended to a full trajectory log (`pd.DataFrame`), per brief SS3.2/SS8.
- Unit tests (`tests/test_selfies_ops.py`, `tests/test_ga.py`, both smoke-marked): mutation/crossover operators always decode to a valid RDKit mol (parametrized over operators and molecules), reproducibility under a fixed seed, tokenize/decode round-trips to the same canonical molecule, the documented single-atom edge case, GA population-size/validity invariants, full-trajectory logging counts, determinism of a full GA run under a fixed seed, elitism-driven monotonic best-fitness-so-far, and that invalid offspring are logged (not dropped) using a monkeypatched flaky validity check. Full suite: 38 tests, ~5.6s (smoke subset: 35 tests, ~1.1s).

## Phase 5 — Constraint modules — DONE

- **`src/rhmoo/applicability_domain.py`**: ECFP4 (r=2, 2048-bit, via `rdFingerprintGenerator`) fingerprints packed as 0/1 matrices; `bulk_max_tanimoto` computes every query-vs-reference Tanimoto in one vectorized float32 matmul (dot product of 0/1 vectors = AND-popcount; union from popcount sums), avoiding per-pair Python loops. `ReferenceFingerprintIndex.from_csv` builds this once from `data/reference/reference_union.csv` and caches it to a `.npz` (`applicability_domain.fingerprint_cache_path` in config, now `data/processed/reference_fingerprints.npz`) — 3.4s cold build for the real 18,856-molecule reference set, 0.04s cached reload, ~0.4s to score a 200-candidate batch (measured, not estimated). `soft_penalty_above_threshold_is_bad(value, theta)` is the shared soft-penalty shape (`min(1, value/theta)`): 1.0 at/above theta, continuous linear decay below it — used for arm B.
- **`src/rhmoo/synthesizability.py`**: wraps RDKit's bundled `sascorer` contrib script (not a public rdkit API — path added at import time). `sa_penalty(score, threshold) = min(1, threshold/score)`: 1.0 at/below threshold, continuous decay above. The brief only mandates a soft penalty for arm B's AD constraint and leaves arm C's SA cutoff otherwise unspecified; for consistency (and because a hard filter would reintroduce the score cliff the AD penalty is explicitly designed to avoid) arm C's SA constraint is implemented as the same soft-penalty shape. Recorded here as a deliberate, justified deviation from a literal "SA score ≤ threshold" hard filter, per brief §3.1's "agent may adjust with justification recorded in the config."
- Unit tests (`tests/test_applicability_domain.py`, `tests/test_synthesizability.py`, both smoke-marked): `bulk_max_tanimoto` against hand-computed 4-bit toy fingerprints (including the all-zero/all-zero edge case, defined as similarity 0, not 1 or NaN); both penalty functions equal 1.0 at/beyond their threshold and decay continuously past it; fingerprint equivalence for equivalent SMILES and all-zero rows for invalid SMILES; reference-index caching (a monkeypatched fingerprinting function that raises proves a cache hit never recomputes). Full suite: 49 tests, ~8.5s (smoke subset: 46 tests, ~1.4s).

## Phase 6 — Independent scorer (H3 — highest priority) — DONE

- Went straight to the fallback path rather than a time-boxed search for a public independent model: identifying, vetting, and safely integrating an unfamiliar third-party pretrained model within an automated-agent workflow carries more risk (unverifiable provenance, possible license/version issues) than the brief's explicitly pre-approved fallback, which is cheap and sufficient.
- **`scripts/train_independent_scorer.py`**: trains on the official TDC `ADMET_Group` splits already cached in Phase 0 (`data/raw/tdc/admet_group/{herg,solubility_aqsoldb}/{train_val,test}.csv`) — hERG as LightGBM binary classification (523 train / 132 test), Solubility_AqSolDB as LightGBM regression (7,985 train / 1,995 test after dropping 2 RDKit-unparseable test rows, logged not silently dropped). Features: RDKit's full `Descriptors.CalcMolDescriptors` (217 physchem descriptors) + MACCS keys (167 bits) — deliberately distinct from both ADMET-AI's Chemprop GNN and the AD constraint's ECFP4. Fixed hyperparameters (no tuning sweep — not warranted at this scale), fixed seed. Held-out test performance (documented, not assumed): **hERG ROC-AUC 0.844**, **Solubility_AqSolDB MAE 0.745** (log-solubility units).
- **`src/rhmoo/independent_scorer.py`**: `featurize()` (same descriptor/MACCS pipeline, invalid SMILES → logged + all-NaN row, never silently dropped) and `IndependentScorer` which loads the frozen boosters from `models/independent_scorer/` (committed to the repo, `make train-independent-scorer` to regenerate deliberately) and scores SMILES batches, returning `hERG_independent` / `Solubility_AqSolDB_independent` columns for the Phase 7 metrics module to compare against the objective's own ADMET-AI scores.
- `models/independent_scorer/metadata.json` records feature columns, library versions, hyperparameters, seed, held-out metrics, and training-data provenance.
- Unit tests (`tests/test_independent_scorer.py`, smoke-marked — LightGBM inference on ~1KB text models is fast): featurization shape/determinism/invalid-SMILES handling, metadata content, and end-to-end scoring (valid SMILES in range, invalid SMILES → NaN, never dropped from the output). Full suite: 53 tests, ~8.2s (smoke subset: 50 tests, ~2.5s).

## Phase 7 — Metrics module — DONE

- **`src/rhmoo/metrics.py`**: implements all of brief SS4. `physchem_panel` (RDKit descriptors: MW, cLogP, TPSA, HBD/HBA, rotatable bonds, ring count, largest ring size, fraction sp3, formal charge); `longest_aliphatic_chain` is not an RDKit builtin — implemented as the standard double-BFS longest-path algorithm over the induced subgraph of non-ring carbons, which is provably a forest (any cycle is by definition a ring, so its atoms are `IsInRing`-flagged and excluded). `structural_alerts` wraps RDKit's `FilterCatalog` PAINS/BRENK catalogs. `sa_scores` batches `synthesizability.sa_score`. `implausibility_battery` applies the frozen config thresholds, treating an unparseable molecule as failing every check it can't be evaluated on (not silently passing). `select_top_k`, `objective_trajectory` (per-generation mean/max/min/std, ignoring logged-invalid rows), `component_breakdown`, `nearest_neighbor_similarity` (thin wrapper over Phase 5's `ReferenceFingerprintIndex`), `independent_divergence` (spearman rank correlation + mean absolute disagreement between an objective raw column and its independent-scorer counterpart, NaN-pair-dropping), and `divergence_headline` (starting-population vs optimized-top-k side by side — the project's headline number). No ADMET-AI dependency, so the whole module is smoke-fast.
- Unit tests (`tests/test_metrics.py`, smoke-marked): hand-computed longest-aliphatic-chain cases (linear chain, all-ring, aromatic ring, ring+substituent, toluene), physchem/alerts/SA valid-vs-invalid-SMILES behaviour, implausibility-battery counting and the NaN-fails-every-check rule, top-k ordering, per-generation trajectory aggregation, component summary stats, nearest-neighbour similarity via a tiny reference index, and independent-divergence golden values (perfect agreement, NaN-dropping). Full suite: 87 tests.

## Phase 8 — Pre-registration freeze — DONE

- Implausibility battery and SA threshold placeholders signed off as-is (user confirmed 2026-09-19): MW ≤ 700, cLogP ≤ 7, TPSA ≤ 200, rotatable bonds ≤ 15, SA ≤ 6.0, 0 PAINS alerts, 0 Brenk alerts, fails if ≥ 2 violated. `configs/main.yaml` and `configs/smoke.yaml` updated to drop the "pending sign-off" TODO markers. Any later change to these values, to the AD theta sweep, or to the budget is a new experiment and must be logged as one (brief SS8).

## Phase 9 — Runner — DONE

- **`src/rhmoo/runner.py`**: orchestrates arms × seeds (× theta for arm B) end to end. `ObjectivePipeline` is the GA `fitness_fn`: ADMET-AI/QED raw values → DrugBank-percentile normalization → weighted geometric mean, multiplied by the AD soft penalty (arm B/C) and/or SA soft penalty (arm C) when configured; failed predictions score 0 rather than corrupting GA selection with NaN. Design decisions made where the brief/config schema left the exact procedure open (recorded in the module docstring):
  - **Arm D (control)**: draws a `k_top`-sized random sample per seed (no GA — nothing to rank), metrics computed on the whole sample.
  - **Arms A/B/C**: run the full GA per seed; the top `k_top` of the *final* population by fitness is the metrics set.
  - **Arm C's AD theta**: fixed at the middle of `applicability_domain.theta_values` (arm B alone sweeps the full range for H4's trade-off curve; arm C only needs one setting alongside SA).
  - **H3 "starting population" baseline**: computed once per experiment (not per arm/seed) since it doesn't depend on either; its sample size is a new config field, `starting_population_baseline_size` (`null` = use the whole population, as `configs/main.yaml` does; `configs/smoke.yaml` caps it at 15 to stay inside the smoke budget).
  - Every run writes its full trajectory (or D's scored sample) to `results/<name>/raw/<arm_label>_seed<seed>.parquet` (brief SS3.2/SS8: every evaluated molecule is data — nothing is filtered before persisting). `results/<name>/tables/` gets `summary_per_seed.csv`, `summary_by_arm.csv` (mean ± sd across seeds, brief SS3.5), `topk_detail.csv` (one row per top-k molecule across every arm/seed, with every raw/normalized component, AD similarity to both reference sets, SA score, physchem panel, alerts, implausibility flags, and independent-scorer scores — the shared input for Phase 10's figures/report), and `starting_population_divergence.csv`.
- **Compute-budget finding (this container, not the Phase 0 Mac)**: this devcontainer has 2 vCPUs / 7.8 GiB RAM. Measured ADMET-AI throughput here (best config, no DrugBank/physchem) is **~56 mol/s**, not the ~390 mol/s measured on the Phase 0 14-core Mac (~7× slower, tracking the core-count ratio). The full pre-registered budget (D + A + B×3θ + C, 5 seeds, pop 200 × gen 100 ≈ 455k raw evaluations before cache dedup) would take on the order of 2–3 hours of sequential wall time here; with only 2 cores, process-pool parallelism across runs would not meaningfully help. **Decision (user confirmed 2026-09-19): finish Phases 7–10 in code and validate via the smoke config; do not execute the full `configs/main.yaml` run in this container.** Running the full experiment is deferred to whenever stronger/more-parallel hardware is available; the code path is otherwise complete and `make reproduce` will run it unchanged.
- Unit test: `tests/test_runner.py` runs the real `configs/smoke.yaml` end to end (real ADMET-AI/LightGBM/RDKit calls, no mocks) and asserts the expected artifacts exist. Full local run: ~16–27s (cache-dependent), inside the 60s smoke budget.

## Phase 10 — Figures and RESULTS.md — DONE

- **`src/rhmoo/figures.py`**: renders all five brief SS5 figures from a results directory's `tables/`/`raw/` artifacts — `fig1_structure_grid` (top-n by *unmodified* objective score, arm A vs the same representative arm-B theta Phase 9 fixed for arm C, RDKit `MolsToGridImage` annotated with objective + independent-hERG score), `fig2_score_vs_nn_distance` (objective score vs `1 - AD similarity to the predictor reference set`, colored by arm), `fig3_objective_vs_independent` (H3 headline: starting-population vs optimized-top-k mean absolute disagreement, grouped bar per property), `fig4_ad_tradeoff` (arm B's achieved objective score ± sd vs θ, twin-axis implausibility rate — H4), `fig5_trajectories` (per-generation mean fitness per non-control arm, min–max shaded across seeds). `generate_all_figures(results_dir)` renders all five from the standard artifact layout; `scripts/generate_figures.py` is the CLI entry point wired to `make figures`.
- **`src/rhmoo/report.py`**: `generate_results_md` writes `RESULTS.md` as tables (hand-rolled Markdown — no `tabulate` dependency) and figure embeds only, per brief SS8 ("the agent must not write conclusions, narrative, or claims into it"): per-arm summary (mean ± sd across seeds), H3 divergence for the starting population and for the optimized top-k, then each figure section (only if the file exists). `scripts/generate_results.py` is the CLI entry point wired to `make results`. A unit test asserts every non-blank line is a heading, a table row, or an image embed — i.e. no prose can slip in undetected.
- Unit tests (`tests/test_figures.py`, `tests/test_report.py`, smoke-marked, synthetic DataFrames — no ADMET-AI/GA dependency): representative-arm-B selection, each figure function creates a non-empty file (including the "only one seed, no shaded band" and "arm without a `generation` column is skipped" edge cases for fig5), `generate_all_figures` produces exactly five files, and RESULTS.md's figure-embed and prose-free-output behaviour. Also manually verified end-to-end against a real (smoke-scale) `configs/smoke.yaml` run: all five figures render sensibly and `RESULTS.md` is well-formed. Full suite: **98 tests, ~20s** (all smoke-marked).
- `make reproduce` now chains the full run with `make figures` and `make results`, matching brief SS6's "one-command reproduction... regenerates every table and figure from scratch." Not executed for the full `configs/main.yaml` budget in this container — see Phase 9's compute-budget note.

## Phase 11 — Verification

`make reproduce` from a clean venv; re-run one arm with the same seed and assert identical top-k; unit tests green; smoke config under 60s; check outputs for stray interpretive prose and for any language implying therapeutic relevance. **Not yet done** — blocked on running the full `configs/main.yaml` experiment (see Phase 9's compute-budget note), which needs stronger/more-parallel hardware than this devcontainer.

## Risks

| Risk | Mitigation |
|---|---|
| ~~ADMET-AI CPU throughput too low~~ | Resolved: 390 mol/s, budget confirmed |
| ~~ADMET-AI v2 lacks the DrugBank percentile path~~ | Resolved: percentiles reproducible from the shipped reference CSV |
| Silent dropping of invalid SMILES by `predict()` | Wrapper realigns on input and logs drops |
| GA converges to a degenerate token repeat that trivially breaks the objective | That is a legitimate result; report it, do not tune it away |
| H2 holds but H3 does not | Report "distribution shift without demonstrable scorer exploitation" as specified |
| Independent scorer is too weak to be evidence | Publish its held-out metrics alongside the divergence number |

## Decisions

Resolved in Phase 0:

2. Normalization: **DrugBank approved percentile**, computed in-repo from `admet_ai/resources/data/drugbank_approved.csv` for the 5 objective components.
4. Budget: **unchanged** (200 pop × 100 gens × 5 seeds × 5 configs).

Resolved in Phase 2:

1. Starting library choice: **GuacaMol v1 training set**, 10,000-molecule sample (seed 0). See "Phase 2 — Data layer" above.

Resolved in Phase 6:

3. Independent scorer path: **in-repo LightGBM** (RDKit descriptors + MACCS keys) on hERG + Solubility_AqSolDB, per the brief's permitted-exception fallback. See "Phase 6 — Independent scorer" above.

Resolved in Phase 8 (user confirmed 2026-09-19):

5. Implausibility battery + SA threshold sign-off: **frozen as the placeholder values** (MW ≤ 700, cLogP ≤ 7, TPSA ≤ 200, rotatable bonds ≤ 15, SA ≤ 6.0, 0 PAINS, 0 Brenk, fails if ≥ 2 violated). See "Phase 8 — Pre-registration freeze" above.

Resolved in Phase 9 (user confirmed 2026-09-19):

6. Compute budget vs. this container's hardware (2 vCPU/7.8GiB, ~56 mol/s measured vs. the Phase 0 Mac's ~390 mol/s): **finish Phases 7–10 in code, validate via the smoke config, do not execute the full `configs/main.yaml` run here.** The full experiment run and Phase 11 verification are deferred to stronger/more-parallel hardware. See "Phase 9 — Runner" above.

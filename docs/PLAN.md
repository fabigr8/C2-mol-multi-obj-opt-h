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

## Phase 5 — Constraint modules

- Applicability domain: max ECFP4 (r=2, 2048-bit) Tanimoto to the reference set, as a **multiplicative soft penalty** below θ. Needs a fast NN search (bulk Tanimoto over a packed fingerprint matrix) — this runs on every candidate, so it must be vectorized.
- SA score via RDKit `sascorer`, threshold from config (arm C).
- Unit tests: Tanimoto against hand-computed pairs; penalty is continuous and equals 1 above θ.

## Phase 6 — Independent scorer (H3 — highest priority)

- Preferred: an ADMET-AI-independent public model for at least one endpoint. Time-boxed search; if nothing clean is available, fall back immediately.
- Fallback (expected path): in-repo LightGBM on RDKit physchem descriptors + a *different* fingerprint, trained on the TDC train split for hERG and aqueous solubility. Report its own held-out test performance so the scorer's competence is documented rather than assumed.
- Frozen model artifact committed/cached; never retrained during a run.

## Phase 7 — Metrics module

All of §4 of the brief, computed per arm × seed for top-k (k=100) and for control arm D:
objective trajectory and final distribution; per-component breakdown; AD nearest-neighbour distances (vs reference set and vs starting population); **objective-vs-independent rank correlation and mean absolute disagreement, on starting population and on optimized top-k separately** (the difference is the headline); physchem panel; SA; PAINS/Brenk; implausibility rate as "fails ≥2 of the pre-declared battery".

## Phase 8 — Pre-registration freeze

Before the first full run, commit the frozen `configs/main.yaml` containing the implausibility battery thresholds, AD θ values, SA cutoff, and budget. Any later change is a new experiment with an explicit log entry and reason.

## Phase 9 — Runner

Orchestrate arms × seeds × θ; shared prediction cache across runs; parallel over runs (process pool) rather than inside ADMET-AI, whichever the Phase 0 measurement favours; write `results/raw/*.parquet` + config copy + run log.

## Phase 10 — Figures and RESULTS.md

Figures 1–5 per the brief, sized for blog width; fig1 = top-20 by objective score, unmodified, annotated with composite and independent scores. `RESULTS.md` generated purely from tables/figures — numbers only, no interpretation.

## Phase 11 — Verification

`make reproduce` from a clean venv; re-run one arm with the same seed and assert identical top-k; unit tests green; smoke config under 60s; check outputs for stray interpretive prose and for any language implying therapeutic relevance.

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

Still open, needed before Phase 6 / Phase 8 respectively:

3. Independent scorer path (recommend in-repo LightGBM on hERG + solubility).
5. Sign-off on the pre-declared implausibility battery thresholds.

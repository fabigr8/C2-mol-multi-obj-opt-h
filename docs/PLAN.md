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

## Phase 1 — Scaffold

- `git init`, `.gitignore` (venv, caches, `data/raw`, large artifacts).
- `pyproject.toml` + `uv.lock`; `src/` layout package (e.g. `rhmoo/`).
- Config schema (dataclass or pydantic) + `configs/smoke.yaml` (pop 20, gens 5) and `configs/main.yaml`. No constants in logic.
- `Makefile`: `setup`, `test`, `smoke`, `reproduce`, `figures`, `results`.
- Structured run logging: config hash, git SHA, timestamp, wall time, molecules evaluated → JSONL.
- `pytest` harness + CI-style smoke target that must finish under 60s.

## Phase 2 — Data layer

- **Starting population.** Pin one public drug-like set, cache locally with a checksum, never silently re-download. Candidates in order of preference: the GuacaMol/MOSES ChEMBL-derived training set (small, standard, pinned by URL+hash) → ZINC drug-like tranche → full ChEMBL dump (largest download, least attractive).
- Record filter criteria and sampling seed in config.
- **Predictor reference set.** DONE in Phase 0 — `data/reference/` holds the TDC `ADMET_Group` `train_val` molecules for exactly the four objective endpoints, plus a sha256 manifest. Remaining: fold the export into the Makefile and document provenance in the README.

## Phase 3 — Objective module

- ADMET-AI wrapper with batched prediction and a persistent cache keyed by canonical SMILES (SQLite or parquet + in-memory dict). This is the main runtime cost.
- Per-component normalization to [0,1] with declared direction; one transform, chosen in Phase 0, identical across all arms.
- Weighted geometric mean aggregation; uniform default weights.
- Unit tests: known inputs → known outputs (golden values), direction handling, zero-component behaviour.

## Phase 4 — SELFIES GA

- Representation: SELFIES token lists; mutation ops = substitution, insertion, deletion; single-point crossover implemented but default off.
- Tournament selection (k=3), elitism top-10%, all knobs in config.
- Per-`(arm, seed)` seeded RNG; numpy + `random` + any inference path seeded for determinism.
- Log **every** evaluated molecule (generation, parent, operator, all component scores, penalties, final fitness) and log failed/invalid candidates rather than dropping them.
- Unit tests: mutations always decode to a valid RDKit mol; operators are reproducible under a fixed seed.

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

Still open, needed before Phase 2:

1. Starting library choice (recommend GuacaMol/MOSES ChEMBL subset for size and pinnability).
3. Independent scorer path (recommend in-repo LightGBM on hERG + solubility).
5. Sign-off on the pre-declared implausibility battery thresholds.

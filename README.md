# rhmoo — Reward Hacking in Molecular Multi-Objective Optimization

Reproducible experiment testing whether a naive gradient-free optimizer, maximizing a
composite objective built from learned ADMET predictors, converges on molecules that
score near-ceiling on the objective while being chemically implausible and/or exploiting
the predictors rather than the properties they estimate. See [docs/BRIEF.md](docs/BRIEF.md)
for the full experimental design and [docs/PLAN.md](docs/PLAN.md) for the implementation
plan and phase status.

Generated molecules are artifacts of an adversarial search procedure, not drug
candidates. This project critiques naive composite objective design, not the
ADMET-AI or TDC tools it uses.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python is pinned to `>=3.13,<3.14` in
`pyproject.toml`; `uv` provisions a matching interpreter regardless of the system default.

```sh
make setup   # uv sync — installs the pinned dependency stack from uv.lock
make data    # builds data/reference/ (TDC) and data/processed/ (starting population)
make test    # full pytest suite
make smoke   # fast pytest subset (must finish in well under a minute)
```

## Data provenance

**Starting population** (`data/processed/starting_population.csv`,
`scripts/build_starting_population.py`): 10,000 molecules sampled without replacement
(seed 0) from the GuacaMol v1 training set (ChEMBL-derived; MD5-pinned download,
`data/raw/guacamol/`). GuacaMol's own curation pipeline applies the drug-likeness and
reactive-group filters; this script additionally canonicalizes, deduplicates, and drops
any RDKit-unparseable entries. Full provenance (source URL, checksum, pool size, sampling
seed, filter criteria) is written to `data/processed/starting_population_manifest.json`
on every run.

**Predictor reference set** (`data/reference/`, `scripts/export_tdc_reference.py`): the
TDC `ADMET_Group` `train_val` split for the four objective endpoints backed by
ADMET-AI (`Caco2_Wang`, `Solubility_AqSolDB`, `hERG`, `CYP3A4_Veith`) — i.e. the molecules
those predictors were fit on. Run in an isolated environment (`make data-reference`)
because `PyTDC` pins `rdkit<2024.3.1`, incompatible with `admet-ai`'s `rdkit>=2025.9.5`
in the main project environment. Per-file SHA-256 hashes are recorded in
`data/reference/manifest.json`.

Both artifacts are cached locally and regenerated deterministically from their pinned
sources; neither script silently re-downloads or re-samples different data on re-run.

## Config

Each experiment run is driven by a single YAML file (`configs/smoke.yaml`,
`configs/main.yaml`); see `src/rhmoo/config.py` for the schema. Key sections:

- `objective`: the composite's components (source, direction, weight), normalization,
  and aggregation method.
- `ga`: population size, generation count, elitism fraction, tournament size, crossover.
- `applicability_domain` / `synthesizability`: constraint knobs for arms B and C.
- `arms`: which constraints are active per arm (D/A/B/C, brief SS3.3).
- `implausibility_battery`: pre-declared thresholds for the plausibility metrics
  (brief SS4, SS8). Values in the committed configs are placeholders pending the
  Phase 8 pre-registration sign-off — see `docs/PLAN.md`.

## Status

Phase 0 (environment spike), Phase 1 (scaffold: config schema, run logging, Makefile,
tests), Phase 2 (data layer: starting population + predictor reference set), and Phase 3
(objective module: ADMET-AI + QED wrapper with a persistent cache, DrugBank-percentile
normalization, weighted geometric mean aggregation) are complete. See `docs/PLAN.md` for
phase-by-phase status and remaining work.

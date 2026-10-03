# rhmoo — Reward Hacking in Molecular Multi-Objective Optimization

Reproducible experiment testing whether a naive gradient-free optimizer, maximizing a
composite objective built from learned ADMET predictors, converges on molecules that
score near-ceiling on the objective while being chemically implausible and/or exploiting
the predictors rather than the properties they estimate. See [docs/BRIEFB-V2.md](docs/BRIEFB-V2.md)
for the full experimental design and [docs/PLAN.md](docs/PLAN.md) for the implementation
plan and phase status.

Generated molecules are artifacts of an adversarial search procedure, not drug
candidates. This project critiques naive composite objective design, not the
ADMET-AI or TDC tools it uses.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python is pinned to `>=3.13,<3.14` in
`pyproject.toml`; `uv` provisions a matching interpreter regardless of the system default.

```sh
make setup       # uv sync — installs the pinned dependency stack from uv.lock
make data        # builds data/reference/ (TDC) and data/processed/ (starting population)
make test        # full pytest suite
make smoke       # fast pytest subset, enforced under 60s
make preflight   # check the pre-run invariants (frozen battery, artifact hashes)
make dry-run     # count planned evaluations and measure this machine's throughput
make reproduce   # the full experiment, then every table and figure
```

Every target takes `CONFIG=` to select an experiment, e.g.
`make reproduce CONFIG=configs/dress_rehearsal.yaml`. The default is
`configs/main.yaml`.

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

**Independent scorers** (`models/independent_scorer/<variant>/`,
`scripts/train_independent_scorer.py`): LightGBM models (RDKit descriptors + MACCS keys —
deliberately distinct from ADMET-AI's Chemprop GNN and from the AD constraint's ECFP4) for
hERG and Solubility_AqSolDB. Three variants of one recipe, differing only in training data,
reported separately and never averaged:

| variant | trained on | evaluated on | hERG ROC-AUC | solubility MAE | disjoint from ADMET-AI's training data |
|---|---|---|---|---|---|
| `model_control` | full TDC `train_val` | `test` | 0.8437 | 0.7445 | no |
| `data_control_half` | a deterministic half of `train_val` | `test` | 0.8084 | 0.7794 | no |
| `data_control_test` | TDC `test` | `train_val` | 0.8260 | 0.8811 | yes |

ADMET-AI is pretrained on all of `train_val`, so only `data_control_test` is genuinely
disjoint from the objective predictor's training data; see `docs/LIMITATIONS.md`. A
competence gate declared before training is recorded per variant. Frozen artifacts are
committed (`make train-independent-scorer` to deliberately regenerate); never retrained
during an experiment run.

**Approved-drug reference set** (`calibration/arm_r_scored.parquet`,
`scripts/score_arm_r.py`): arm R — the 2,845-molecule DrugBank set shipped with ADMET-AI,
scored by the full pipeline but never optimized. Source file, its SHA-256, and the output
hash are recorded in `calibration/arm_r_manifest.json`. The runner loads and hash-verifies
this artifact rather than rescoring, so the arm R reported in results is the same set the
implausibility battery was calibrated against.

## Config

Each experiment run is driven by a single YAML file (`configs/smoke.yaml`,
`configs/main.yaml`); see `src/rhmoo/config.py` for the schema. Key sections:

- `objective`: the composite's components (source, direction, weight), normalization,
  and aggregation method.
- `ga`: population size, generation count, elitism fraction, tournament size, crossover.
- `applicability_domain` / `synthesizability`: constraint knobs for arms B and C.
- `arms`: which constraints are active per arm (D/A/B/C, brief SS3.3).
- `independent_scorers`: which scorer variants to run, and which is the pre-declared
  headline H3 oracle.
- `decoupling`: the H3b detector's margin, persistence and declared margin sweep.
- `reference_arm`: path to the frozen arm R artifact.
- `implausibility_battery`: thresholds **calibrated** against arm R so that no more than
  5% of approved drugs are flagged, then frozen. `rhmoo.preflight` refuses to start a run
  whose config does not match `calibration/battery_frozen.json`, so these cannot be edited
  without recalibrating. See `docs/PREREGISTRATION.md`.

## Pre-registration

Thresholds and detector definitions were fixed before the first full run, and the
implausibility battery was calibrated against approved drugs rather than declared.
`docs/PREREGISTRATION.md` records the rule, the search, the outcome, and the battery's
declared blind spot. `make calibrate` regenerates it; the runner enforces it.

## Running the full experiment on another machine

ADMET-AI throughput varies by roughly 7x across machines, so the full
`configs/main.yaml` budget is minutes or hours depending on where it runs.

```sh
make setup                 # pinned dependency stack
make data-starting-population   # pinned GuacaMol sample
make preflight             # verifies frozen battery + committed artifact hashes
make dry-run               # measured throughput and an ETA for this machine
make reproduce             # the full run, then all tables and figures
make verify-determinism    # re-runs one arm and asserts an identical top-k
```

`make data-reference` is only needed to rebuild the applicability-domain reference set;
it requires the isolated PyTDC environment described above. The scorer variants and the
arm R artifact are committed, so they do not need regenerating — and `make reproduce`
deliberately verifies their hashes rather than rebuilding them, since retraining inside a
reproduction would void the frozen-artifact guarantee the pre-registration rests on.

`configs/dress_rehearsal.yaml` runs the identical code path at ~1/30 the compute.

## Documentation

- `docs/BRIEFB-V2.md` — the experimental design this repository implements.
- `docs/PREREGISTRATION.md` — thresholds and detector definitions fixed before the run.
- `docs/LIMITATIONS.md` — what the measurements cannot show, and why.
- `docs/RELATED_WORK.md` — prior work; H1, H2 and H4 are replications.
- `docs/DEVIATIONS.md` — where the implementation departs from the brief, and where the
  brief is incorrect or internally inconsistent.
- `docs/PLAN.md` — phase-by-phase implementation status.

## Status

The code path is complete and validated end to end at reduced scale
(`configs/dress_rehearsal.yaml`). The full `configs/main.yaml` run is deferred to
hardware with more cores than the 2-vCPU container it was developed in; see
`docs/PLAN.md`.

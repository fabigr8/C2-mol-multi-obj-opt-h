# Which experiment to run. Override to run a different config without editing
# this file:  make reproduce CONFIG=configs/dress_rehearsal.yaml
CONFIG ?= configs/main.yaml
RESULTS ?= results/$(notdir $(basename $(CONFIG)))

.PHONY: setup test smoke data data-reference data-starting-population \
	train-independent-scorer arm-r calibrate preflight dry-run \
	reproduce dress-rehearsal verify-determinism figures results

setup:
	uv sync

test:
	uv run pytest -q

# brief SS7 requires the smoke subset to finish in under a minute; the timeout
# makes that an enforced budget rather than an aspiration.
smoke:
	timeout 60 uv run pytest -q -m smoke

# TDC ADMET_Group train_val export (brief SS3.4). Isolated env: PyTDC pins
# rdkit<2024.3.1, incompatible with admet-ai 2.x in the main project env.
data-reference:
	uv run --isolated --no-project --python 3.11 \
		--with PyTDC==1.1.15 --with 'pandas<3' --with 'setuptools<81' \
		python scripts/export_tdc_reference.py

# Starting population (brief SS3.4): pinned GuacaMol v1 sample, cached under
# data/processed/. Re-run is a no-op if the cached source already matches the
# pinned checksum.
data-starting-population:
	uv run python scripts/build_starting_population.py

data: data-reference data-starting-population

# Independent scorers (brief SS4.2, H3): frozen LightGBM models under
# models/independent_scorer/<variant>/, committed to the repo and never
# retrained during a run. Re-run only to deliberately refresh the frozen
# artifacts -- `reproduce` verifies their hashes instead of rebuilding them,
# because retraining inside a reproduction would void the frozen-artifact
# guarantee the pre-registration rests on.
train-independent-scorer:
	uv run python scripts/train_independent_scorer.py

# Arm R (brief SS3.3): score the approved-drug reference set once and freeze it.
# Must precede calibration.
arm-r:
	uv run python scripts/score_arm_r.py --config $(CONFIG)

# Implausibility-battery calibration (brief SS4.5 -- blocking). Must precede any
# optimized run; the runner refuses to start otherwise.
calibrate:
	uv run python scripts/calibrate_battery.py --config $(CONFIG)

# Pre-run invariants: frozen battery, arm R artifact hash, scorer artifacts.
preflight:
	uv run python -c "from rhmoo.config import load_config; from rhmoo.preflight import check_experiment; \
		check_experiment(load_config('$(CONFIG)')); print('preflight OK: $(CONFIG)')"

# Count planned evaluations and measure this machine's throughput before
# committing hours to a run. Throughput varies by ~7x across machines.
dry-run:
	uv run python -m rhmoo.runner --config $(CONFIG) --dry-run

# One-command reproduction (brief SS7): all arms x seeds x theta, then every
# table and figure regenerated from scratch.
reproduce: preflight
	uv run python -m rhmoo.runner --config $(CONFIG)
	$(MAKE) figures CONFIG=$(CONFIG)
	$(MAKE) results CONFIG=$(CONFIG)

# Structurally identical to the main run at ~1/30 the compute: same arms, seeds
# and theta sweep, smaller population and generation count. Exercises every code
# path the main run takes.
dress-rehearsal:
	$(MAKE) reproduce CONFIG=configs/dress_rehearsal.yaml

# brief SS10: re-run one arm with the same seed and assert an identical top-k.
verify-determinism:
	uv run python scripts/verify_determinism.py --config $(CONFIG)

figures:
	uv run python scripts/generate_figures.py --results-dir $(RESULTS)

results:
	uv run python scripts/generate_results.py --results-dir $(RESULTS)

.PHONY: setup test smoke data data-reference data-starting-population train-independent-scorer reproduce figures results

setup:
	uv sync

test:
	uv run pytest -q

smoke:
	uv run pytest -q -m smoke

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

# Independent scorer (brief SS4/SS6, Phase 6, H3): frozen LightGBM models under
# models/independent_scorer/, committed to the repo and never retrained during
# a run. Re-run only to deliberately refresh the frozen artifacts.
train-independent-scorer:
	uv run python scripts/train_independent_scorer.py

# One-command reproduction (brief SS6): all arms x seeds x theta, then every
# table and figure regenerated from scratch.
reproduce:
	uv run python -m rhmoo.runner --config configs/main.yaml
	$(MAKE) figures
	$(MAKE) results

figures:
	uv run python scripts/generate_figures.py --results-dir results/main

results:
	uv run python scripts/generate_results.py --results-dir results/main

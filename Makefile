.PHONY: setup test smoke data data-reference data-starting-population reproduce figures results

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

# One-command reproduction (brief SS6). The arms x seeds optimization loop lands
# in Phase 9; today this exercises config loading, output layout, and run
# logging end to end.
reproduce:
	uv run python -m rhmoo.runner --config configs/main.yaml

figures:
	@echo "figures: not implemented yet (PLAN Phase 10)"

results:
	@echo "RESULTS.md generation: not implemented yet (PLAN Phase 10)"

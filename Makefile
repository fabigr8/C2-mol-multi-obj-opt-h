.PHONY: setup test smoke reproduce figures results

setup:
	uv sync

test:
	uv run pytest -q

smoke:
	uv run pytest -q -m smoke

# One-command reproduction (brief SS6). The arms x seeds optimization loop lands
# in Phase 9; today this exercises config loading, output layout, and run
# logging end to end.
reproduce:
	uv run python -m rhmoo.runner --config configs/main.yaml

figures:
	@echo "figures: not implemented yet (PLAN Phase 10)"

results:
	@echo "RESULTS.md generation: not implemented yet (PLAN Phase 10)"

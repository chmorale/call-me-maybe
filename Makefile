# ==============================================================================
#                                  MAKEFILE
# ==============================================================================

install:
	uv sync

run: install
	uv run python -m src

debug: install
	uv run python -m pdb -m src

lint: install
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: install
	uv run flake8 .
	uv run mypy . --strict

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache

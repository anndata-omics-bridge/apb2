VENV_BIN := .venv/bin
DOCS_PORT ?= 8000

# Corpus-backed tests run when a downloaded vendor corpus is present; they skip otherwise.
APB2_TEST_DATA ?= $(abspath ../legacy/test_data_download)
export APB2_TEST_DATA

.DEFAULT_GOAL := help
.PHONY: help sync format format-check lint typecheck deps test build docs docs-serve \
	docs-serve-public check clean

help:  ## Show developer commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

sync:  ## Synchronize the locked development environment
	uv sync --frozen --group dev

format:  ## Format and autofix source and tests
	$(VENV_BIN)/ruff format src tests
	$(VENV_BIN)/ruff check --fix src tests

format-check:  ## Check formatting without changing files
	$(VENV_BIN)/ruff format --check src tests

lint:  ## Run code and import-architecture lint checks
	$(VENV_BIN)/ruff check src tests
	$(VENV_BIN)/lint-imports

typecheck:  ## Run standard Pyright in strict mode
	$(VENV_BIN)/pyright

deps:  ## Validate dependency declarations
	$(VENV_BIN)/deptry .

test:  ## Run tests with branch coverage
	$(VENV_BIN)/pytest --cov --cov-branch

build:  ## Build and validate source and wheel distributions
	uv build
	$(VENV_BIN)/twine check dist/*

docs:  ## Build user documentation with strict warnings
	uv run --frozen --group docs zensical build --clean --strict

docs-serve:  ## Serve user documentation locally
	uv run --frozen --group docs zensical serve

docs-serve-public:  ## Serve the prebuilt public directory without rebuilding
	@test -f public/index.html || (echo "public/index.html is missing; run 'make docs' first" >&2; exit 1)
	$(VENV_BIN)/python -m http.server $(DOCS_PORT) --directory public

check:  ## Run every merge-blocking quality gate
	uv lock --check
	$(MAKE) format-check lint typecheck deps test build docs

clean:  ## Remove generated build and quality artifacts
	$(VENV_BIN)/python -c "import shutil; [shutil.rmtree(path, ignore_errors=True) for path in ('build', 'dist', 'public', '.pytest_cache', '.ruff_cache')]"

# Contributing

TreeShaker is open to contributions. Here's how to get started.

## Development setup

```bash
git clone https://github.com/SeanPeppers/TreeShaker.git
cd TreeShaker
pip install -e ".[dev]"
```

This installs TreeShaker in editable mode with all development tools (ruff, pytest, mypy, pytest-cov).

## Code quality

TreeShaker uses three tools to maintain code quality:

### Linting (ruff)

```bash
ruff check src/ tests/ demo/
```

### Formatting (ruff)

```bash
ruff format src/ tests/ demo/
```

### Type checking (mypy)

```bash
mypy
```

Mypy is configured in `pyproject.toml` and checks only `src/treeshaker/`.

### Running all checks

```bash
# Lint
ruff check src/ tests/ demo/

# Format
ruff format src/ tests/ demo/

# Tests with coverage
pytest --cov=treeshaker --cov-report=html

# Type check
mypy
```

## Testing

Run the test suite:

```bash
pytest
```

Run tests with coverage report:

```bash
pytest --cov=treeshaker --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`. TreeShaker targets ≥85% line coverage (enforced by pytest configuration in `pyproject.toml`).

## Dependency management

TreeShaker uses [`uv`](https://docs.astral.sh/uv/) to pin exact dependency versions in `uv.lock`. When you update `pyproject.toml` dependencies, regenerate the lock file:

```bash
uv lock
```

Commit the updated `uv.lock` alongside your `pyproject.toml` changes. This ensures local installs and CI use the same resolved versions.

## Git workflow

1. **Create a feature branch** off `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit. Follow these rules:
   - **One PR per fix/feature** — don't bundle unrelated changes.
   - **Never copy-paste code** — extract shared functions/classes/fixtures instead.
   - **Sync docs with API changes** (hard rule):
     - If you touch `Config`/nested dataclasses (config.py), `FeatureSelectionPipeline` public methods (pipeline.py), or CLI flags (cli.py) → update README.md and `src/treeshaker/mcp_server.py` in the same PR.
     - Update the config/flag table in documentation.
     - Update `list_config_schema()` return type and `mcp_server.py` tool signatures if the API changes.

3. **Run checks before pushing**:
   ```bash
   ruff check src/ tests/ demo/
   ruff format src/ tests/ demo/
   mypy
   pytest
   ```

4. **Push and create a PR**:
   ```bash
   git push -u origin feature/your-feature-name
   ```

5. **CI runs automatically** — GitHub Actions runs ruff lint/format checks on every push and PR. Merges are blocked if checks fail.

## Documentation

The main documentation is auto-generated from this Sphinx site. When you add or change features:

- Update the relevant `.md` file in `docs/source/`
- If adding a new major feature, consider a new page
- Build locally to verify:
  ```bash
  pip install ".[docs]"
  sphinx-build docs/source docs/_build
  open docs/_build/index.html
  ```

The README.md is now a short landing page; detailed docs live in Sphinx.

## Project structure

```
TreeShaker/
├── src/treeshaker/
│   ├── __init__.py              # Public API exports
│   ├── cli.py                   # Command-line interface
│   ├── config.py                # Configuration dataclasses
│   ├── pipeline.py              # Main FeatureSelectionPipeline class
│   ├── importance.py            # SHAP, permutation, mutual information
│   ├── leakage_detection.py     # Leakage and logical-error checks
│   ├── harm_detection.py        # Low importance flagging
│   ├── dtype_handling.py        # Data type validation and preprocessing
│   ├── residual_diagnostics.py  # Residual correlation checks
│   ├── synthetic_features.py    # Feature suggestion from interactions
│   ├── mcp_server.py            # MCP server implementation
│   ├── utils.py                 # Shared utilities
│   └── exceptions.py            # Custom exception classes
├── tests/                       # Test suite
├── docs/                        # Sphinx documentation
├── demo/                        # Gradio demo app
├── README.md                    # Landing page + quick start
├── pyproject.toml               # Package metadata and dependencies
├── uv.lock                      # Pinned dependency versions
└── CLAUDE.md                    # Project guidelines (this file)
```

## Key design principles

From `CLAUDE.md`:

1. **Sync docs+MCP with API changes** — documentation and MCP schemas must stay in sync with the code.
2. **One PR per fix/feature** — no bundling unrelated changes.
3. **Never copy-paste** — extract shared code instead.
4. **Validate against real data** — synthetic fixtures alone aren't enough for dtype/leakage/harm-detection changes.

## Validate the reduction

When making changes to harm or leakage detection, test against real datasets (e.g., `sklearn.datasets.fetch_openml`) to ensure the changes don't inadvertently mark safe features as harmful or miss true leakage.

## Common tasks

### Add a new CLI flag

1. Add the argument to `parse_arguments()` in `cli.py`
2. Map it to the `Config` object in `main()`
3. Update `Config` or a nested config dataclass in `config.py` if needed
4. Document the flag in `docs/source/config_reference.md` and `docs/source/cli_reference.md`
5. Update `README.md` CLI table if it's a major flag
6. Update `mcp_server.py` if the MCP API should expose it
7. Add tests in `tests/test_cli.py` and `tests/test_config.py`

### Add a new configuration option

1. Add a field to the appropriate dataclass in `config.py` (or create a new nested config if it's a logical group)
2. Add validation in `__post_init__()` if needed
3. Document in `docs/source/config_reference.md`
4. If it's CLI-exposed, follow the steps above
5. Update `mcp_server.py` if the tool signature should change
6. Add tests in `tests/test_config.py`

### Modify the pipeline API

If you change public methods in `pipeline.py`:

1. Update the method signature and docstring
2. Add tests in `tests/test_pipeline.py`
3. Update `docs/source/quickstart.md` or `docs/source/how_it_works.md` if the usage example changes
4. If the return type changes, update `mcp_server.py` tool return types

## Getting help

- Check existing tests for examples of how to use the API
- Read the docstrings in `src/treeshaker/*.py`
- Open an issue on GitHub for questions or discussion

## Contributor License Agreement

By submitting a pull request, you agree that your contributions will be licensed under the MIT License, the same as TreeShaker.

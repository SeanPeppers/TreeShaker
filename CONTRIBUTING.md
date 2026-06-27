# Contributing to TreeShaker

Thank you for your interest in contributing! We welcome bug reports, feature requests, and pull requests.

## Development Setup

### Prerequisites

- Python 3.13+
- Git

### Clone and Install

```bash
git clone https://github.com/SeanPeppers/TreeShaker.git
cd TreeShaker
pip install -e ".[dev]"
```

Alternatively, with `uv`:

```bash
uv sync --extra dev
```

## Running Checks Locally

Before pushing, run the full test suite and lint checks to ensure your changes pass CI:

### Lint and Format

```bash
# Check for lint violations
ruff check src/ tests/

# Auto-format code
ruff format src/ tests/
```

### Type Checking

```bash
mypy src/
```

### Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=treeshaker --cov-report=html
```

### Security Audit

```bash
pip-audit
```

## Workflow

1. Create a new branch off `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit with descriptive messages:
   ```bash
   git add .
   git commit -m "feat: add new feature" # or "fix: resolve bug", "docs: update", etc.
   ```

3. Push your branch and open a pull request against `main`:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Ensure all CI checks pass (ruff lint/format, pytest, mypy, pip-audit).

## CI Requirements

Every push to `main` and pull request runs GitHub Actions with the following jobs:

- **ruff** — code linting and format checking
- **pytest** — test suite
- **mypy** — type checking
- **pip-audit** — security dependency audit

Your PR will not merge unless all checks pass. If a check fails, the error output will guide you to fix it locally.

## Code Style

- **Linter** — ruff (E, F, W, I rules)
- **Formatter** — ruff format
- **Line length** — 88 characters
- **Type hints** — required for all function signatures in `src/treeshaker/`

## Dependency Pinning

This project uses `uv` and `uv.lock` to pin exact dependency versions. When updating `pyproject.toml`:

```bash
uv lock
```

Commit the updated `uv.lock` alongside your `pyproject.toml` change.

## Questions or Issues?

Open a GitHub issue to report bugs or discuss features. Be as specific as possible, including:

- Python version and OS
- Full error message and stack trace (if applicable)
- Steps to reproduce the issue
- Expected vs. actual behavior

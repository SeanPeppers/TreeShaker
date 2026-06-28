# Installation

Requires Python 3.13+.

## Standard install

```bash
pip install treeshaker
```

This installs the core library and CLI. No external dependencies beyond XGBoost, SHAP, Optuna, pandas, scikit-learn, and numpy.

## Development install

To contribute or work with the source:

```bash
git clone https://github.com/SeanPeppers/TreeShaker.git
cd TreeShaker
pip install -e ".[dev]"
```

This installs TreeShaker in editable mode with development tools (ruff, pytest, mypy, pytest-cov).

## Optional extras

### Demo UI

Try TreeShaker interactively without writing code:

```bash
pip install "treeshaker[demo]"
python -m treeshaker.demo
```

This installs Gradio and launches a web UI on `http://127.0.0.1:7860` where you can upload a CSV, configure it, and see recommendations in real time.

### MCP server

To use TreeShaker as an MCP server with Claude Code or other AI clients:

```bash
pip install "treeshaker[mcp]"
treeshaker-mcp
```

See [MCP server](mcp_server.md) for setup details.

### Documentation

To build the Sphinx documentation locally:

```bash
pip install "treeshaker[docs]"
sphinx-build docs/source docs/_build
```

Then open `docs/_build/index.html` in your browser.

## Verification

Verify your installation:

```bash
treeshaker --help
```

You should see the CLI help text and version information. If not, check that the Python environment contains treeshaker (e.g., `pip list | grep treeshaker`).

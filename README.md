# TreeShaker

[![Lint](https://github.com/SeanPeppers/TreeShaker/actions/workflows/lint.yml/badge.svg)](https://github.com/SeanPeppers/TreeShaker/actions/workflows/lint.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/SeanPeppers/TreeShaker/badge)](https://scorecard.dev/viewer/?uri=github.com/SeanPeppers/TreeShaker)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/15980/badge)](https://www.bestpractices.dev/projects/15980)
![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)

Automated feature selection using XGBoost, SHAP, and Optuna. Give it a CSV and a target column; it tunes a model, ranks every feature by importance, and flags the ones that are hurting you.

## Why TreeShaker?

Most feature-importance tools report one number per feature from one model fit, so you can't tell signal from noise. TreeShaker checks each number several ways:

- **Held-out evaluation:** importance is computed on a test split the model never sees.
- **Stability:** permutation importance is averaged over shuffles, with an opt-in bootstrap ensemble and multi-seed stability report (`--n-stability-runs`).
- **Plain-English recommendations:** every feature gets Drop, Caution, or Keep with its reasoning.
- **Independent cross-checks:** SHAP, permutation importance, mutual information, leakage statistics, and residual correlation, with an opt-in divergence score for features where they disagree.
- **Leakage and noise detection:** seven leakage checks (including look-ahead and time-series split problems) and shadow probes that set a noise floor.
- **Optional extra selectors:** Boruta, RFECV, LightGBM, CatBoost, Random Forest, and scikit-learn permutation importance.
- **Scale options:** GPU, Optuna pruning, parallel evaluation, stratified sampling, and a streaming path for datasets too big for RAM.
- **Reports and tracking:** CSV or HTML output, MLflow logging, and synthetic-feature suggestions (DFE).

Every option is opt-in and listed in the [CLI reference](docs/source/cli_reference.md) and [configuration reference](docs/source/config_reference.md).

## Quick start

**CLI:**
```bash
# Not published to PyPI: install from GitHub. Pin to a release tag
# rather than @main for a stable version.
pip install git+https://github.com/SeanPeppers/TreeShaker.git@v0.1.2
treeshaker --input data.csv --target price --task regression
```


**Python API:**
```python
from treeshaker import FeatureSelectionPipeline, Config

config = Config(
    input_file="data.csv",
    target_column="price",
    task="regression",
)
pipeline = FeatureSelectionPipeline(config)
results = pipeline.run()
print(results)
```

[`examples/california_housing_walkthrough.ipynb`](examples/california_housing_walkthrough.ipynb) is an executed, end-to-end walkthrough on a real dataset, with leakage and multicollinearity injected to show detection in action.

## Documentation

The docs aren't hosted. The source is in `docs/source/`:

- [Installation guide](docs/source/installation.md)
- [How it works](docs/source/how_it_works.md)
- [Feature importance methods](docs/source/importance_methods.md)
- [Leakage detection](docs/source/leakage_detection.md)
- [Configuration reference](docs/source/config_reference.md)
- [CLI reference](docs/source/cli_reference.md)
- [MCP server setup](docs/source/mcp_server.md)

Build a browsable HTML site locally:
```bash
pip install "treeshaker[docs]"
sphinx-build docs/source docs/_build
open docs/_build/index.html
```

## Development

```bash
git clone https://github.com/SeanPeppers/TreeShaker.git
cd TreeShaker
pip install -e ".[dev]"

# Lint, format, type-check, test

ruff check src/ tests/ demo/
ruff format src/ tests/ demo/
mypy
pytest
```

See [Contributing](docs/source/contributing.md) for details.

## License

MIT

# TreeShaker

[![Lint](https://github.com/SeanPeppers/TreeShaker/actions/workflows/lint.yml/badge.svg)](https://github.com/SeanPeppers/TreeShaker/actions/workflows/lint.yml)
![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)

Automated feature selection using XGBoost, SHAP, and Optuna. Give it a CSV and a target column — it tunes a model, ranks every feature by importance, and flags the ones that are hurting you.

## Why TreeShaker?

Most feature-importance tools give you a single number per feature from a single model fit, with no way to tell whether that number is signal or noise. TreeShaker's core differentiator is **explainability you can actually trust**, backed by real mechanism rather than a one-shot importance score. Key features include:

- **Genuinely held-out evaluation** — every importance number is computed on a test split the model never sees, not on data it has partially memorized.
- **Stability, not just a point estimate** — permutation importance is averaged over multiple shuffles, and an opt-in bootstrap ensemble shows whether a feature's rank holds up.
- **Plain-English explanations** — each feature gets a Drop/Caution/Keep recommendation with clear reasoning.
- **Cross-checked, not single-signal** — SHAP, permutation importance, mutual information, leakage statistics, and residual correlation are all computed independently. An opt-in importance divergence score (`compute_importance_divergence` / `--compute-importance-divergence`) flags features where these signals disagree about a feature's rank.
- **Boruta feature selection** — opt-in feature selection with `--use-boruta` (`use_boruta=True`), adding a `boruta_decision` column (`Confirmed`, `Tentative`, or `Rejected`).
- **RFECV feature selection** — opt-in feature elimination with `--use-rfecv` (`use_rfecv=True`), adding an `rfecv_ranking` column (1 = selected feature rank).
- **LightGBM importance cross-validation** — opt-in model swap with `--use-lightgbm` (`use_lightgbm=True`), adding a `lightgbm_importance` column.
- **CatBoost importance backend** — opt-in model evaluation with native categorical handling using `--use-catboost` (`use_catboost=True`), adding a `catboost_importance` column.

## Quick start

**CLI:**
```bash
pip install treeshaker
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

## Documentation

Complete documentation is at [https://treeshaker.readthedocs.io](https://treeshaker.readthedocs.io/), including:

- [Installation guide](https://treeshaker.readthedocs.io/en/latest/installation.html)
- [How it works](https://treeshaker.readthedocs.io/en/latest/how_it_works.html)
- [Feature importance methods](https://treeshaker.readthedocs.io/en/latest/importance_methods.html)
- [Leakage detection](https://treeshaker.readthedocs.io/en/latest/leakage_detection.html)
- [Configuration reference](https://treeshaker.readthedocs.io/en/latest/config_reference.html)
- [CLI reference](https://treeshaker.readthedocs.io/en/latest/cli_reference.html)
- [MCP server setup](https://treeshaker.readthedocs.io/en/latest/mcp_server.html)

Or build locally:
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

See [Contributing](https://treeshaker.readthedocs.io/en/latest/contributing.html) for details.

## License

MIT

# TreeShaker

Automated feature selection using XGBoost, SHAP, and Optuna. Give it a CSV and a target column — it tunes a model, ranks every feature by importance, and flags the ones that are hurting you.

## Why TreeShaker?

Most feature-importance tools give you a single number per feature from a single model fit, with no way to tell whether that number is signal or noise. TreeShaker's core differentiator is **explainability you can actually trust**, backed by real mechanism rather than a one-shot importance score:

- **Genuinely held-out evaluation** — every importance number is computed on a test split the model never sees during tuning or fitting, not on data it has partially memorized.
- **Stability, not just a point estimate** — permutation importance is averaged over multiple independent shuffles, and an opt-in bootstrap ensemble refits the model to show whether a feature's rank holds up or jumps around.
- **Plain-English explanations** — each feature gets a Drop/Caution/Keep recommendation with reasoning you can read and act on.
- **Cross-checked, not single-signal** — SHAP, permutation importance, mutual information, leakage statistics, and (opt-in) residual correlation are all computed independently and reconciled.
- **Leakage detection** — catches suspicious patterns: high target correlation, near-duplicates, multicollinearity, suspicious column names.
- **Three output formats** — CSV results, HTML reports, or Python API.

## Quick links

- [Installation](installation.md) — Get started in seconds
- [Quick start](quickstart.md) — First run in 2 minutes
- [How it works](how_it_works.md) — Understand the pipeline
- [Feature importance methods](importance_methods.md) — SHAP, permutation, mutual information
- [Leakage detection](leakage_detection.md) — Catch suspicious columns
- [Harm detection](harm_detection.md) — Identify truly useless features
- [Configuration reference](config_reference.md) — All flags and options
- [CLI reference](cli_reference.md) — Complete command-line guide
- [MCP server](mcp_server.md) — Use TreeShaker via Claude Code
- [Contributing](contributing.md) — Develop locally

```{toctree}
:hidden:

installation
quickstart
how_it_works
importance_methods
leakage_detection
harm_detection
config_reference
cli_reference
mcp_server
contributing
```

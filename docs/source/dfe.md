# Derived feature engineering (DFE)

Derived feature engineering (DFE) turns TreeShaker's SHAP **interaction** analysis into concrete, testable feature suggestions. When two features interact strongly, the model is often reconstructing a derived quantity — a product or a ratio — piece by piece. DFE proposes that derived feature explicitly, scores it, and (optionally) lets it compete against the raw features that spawned it.

DFE is built on top of the SHAP interaction matrix, so it requires `compute_shap_interactions=True` (`--shap-interactions`). If interactions aren't computed, DFE is skipped with a logged warning.

## What DFE does

1. Take the top `interaction_top_n` strongest-interacting feature pairs from the SHAP interaction matrix.
2. For each pair (A, B) and each configured operator, build a candidate derived feature:
   - **product** — `A * B`, named `A__x__B`
   - **ratio** — `A / B`, named `A__div__B` (skipped when the denominator is numerically unstable: any exact zero, or a value that is negligibly small relative to the denominator's own scale — either would blow the ratio up and corrupt the score)
   - Categorical constituents (pandas `category` or `object` dtype) are **target-mean encoded** first — each category level is replaced by the mean of the target within that level — so a categorical feature can take part in a numeric product/ratio.
3. Score each candidate by **mutual information** against the target.
4. Keep every candidate whose MI score exceeds `mi_threshold`, sorted best-first.

## Advisory mode (default)

With `dfe_loop_feedback=False` (the default), DFE is purely advisory. Every candidate above `mi_threshold` is written to `<output_stem>_suggested_features.csv` with columns `feature_a`, `feature_b`, `operator`, `suggested_name`, and `mi_score`. **Nothing is added to the model** — a human reviews the suggestions and decides whether to engineer any of them by hand. Recommendations in the main results table are untouched.

## Loop-feedback mode

With `dfe_loop_feedback=True`, DFE runs a **real competition** instead of just reporting. The top `dfe_max_inject` candidates by MI are injected into the feature set, and TreeShaker tests whether they actually beat the raw features:

1. The injected synthetics are added to both a **train+val** frame and the held-out **test** frame.
2. A fresh model is re-fit (reusing the already-tuned `best_params`) on the augmented training features.
3. SHAP importance is computed on the augmented test split for the raw features **and** the synthetics together.
4. A raw constituent is marked `recommendation="ReplacedBySynthetic"` **only when its synthetic derivative's SHAP importance strictly exceeds its own**.
5. Each **winning** synthetic (one that displaced at least one raw feature) is appended to the main results table as its own `Keep` row, carrying the SHAP importance it earned in the competition, so the recommended replacement is visible right next to the features it displaced.

Because a single synthetic is derived from two raw features, both constituents A and B are eligible to be replaced — each is compared independently against the synthetic's importance.

### Leakage safety

The competition is designed so a synthetic can't win by cheating. For categorical constituents, the target-mean encoders are **fit on the training split only** (via `fit_interaction_encoders`) and the *same* fixed map is applied to the test split. A test-only category level unseen during fitting falls back to the mean of the fitted category means rather than encoding the held-out target. This prevents a categorical synthetic from winning the SHAP comparison simply because its encoding memorised the test labels.

(The advisory MI scoring is protected the same way: categorical encodings for the MI score are fit on train+val and merely applied to the test rows the MI is measured over.)

## Configuration reference

Every knob is exposed three ways — as a `SyntheticConfig` field in Python, as a CLI flag, and as a `run_pipeline` MCP parameter.

| `SyntheticConfig` field | CLI flag | MCP `run_pipeline` param | Default | Meaning |
|---|---|---|---|---|
| `synthesize_from_interactions` | `--synthesize-features` | `synthesize_features` | `False` | Enable DFE. Requires SHAP interactions. |
| `interaction_top_n` | `--synthesis-top-n` | `synthesis_top_n` | `5` | Number of strongest interacting pairs to consider. |
| `mi_threshold` | `--synthesis-mi-threshold` | `synthesis_mi_threshold` | `0.01` | Minimum MI a candidate must exceed to be kept. |
| `operators` | `--synthesis-operators` | `synthesis_operators` | `["product", "ratio"]` | Operators to build candidates from. |
| `dfe_loop_feedback` | `--dfe-loop-feedback` | `dfe_loop_feedback` | `False` | Run the injection competition instead of advisory-only. |
| `dfe_max_inject` | `--dfe-max-inject` | `dfe_max_inject` | `5` | Max synthetics to inject (top-scoring by MI) in loop feedback. |

In Python, these fields live on a `SyntheticConfig` passed as `Config(synthetic_config=...)`. The CLI and MCP layers assemble that `SyntheticConfig` for you from the flags/params above.

## Worked example

**CLI** — advisory suggestions written to `results_suggested_features.csv`:

```bash
treeshaker --input data.csv --target price --task regression \
  --shap-interactions --synthesize-features \
  --synthesis-top-n 8 --synthesis-mi-threshold 0.02 \
  --synthesis-operators product ratio \
  --output results.csv
```

Add loop feedback to let winners demote their raw constituents:

```bash
treeshaker --input data.csv --target price --task regression \
  --shap-interactions --synthesize-features --dfe-loop-feedback \
  --dfe-max-inject 3 --output results.csv
```

**Python API:**

```python
from treeshaker import FeatureSelectionPipeline, Config
from treeshaker.synthetic_features import SyntheticConfig

config = Config(
    input_file="data.csv",
    target_column="price",
    task="regression",
    compute_shap_interactions=True,   # required for DFE
    synthetic_config=SyntheticConfig(
        synthesize_from_interactions=True,
        interaction_top_n=8,
        mi_threshold=0.02,
        operators=["product", "ratio"],
        dfe_loop_feedback=True,
        dfe_max_inject=3,
    ),
)
pipeline = FeatureSelectionPipeline(config)
results = pipeline.run()
```

After a run, the suggestions object is also available on the pipeline as `pipeline.suggested_features`.

## Caveats

- **Loop feedback can demote *all* the raw constituents.** Because both A and B of a winning pair can be marked `ReplacedBySynthetic`, an aggressive run (low `mi_threshold`, many pairs) can leave few — or no — raw `Keep` rows in the results table. The winning synthetics are added as their own `Keep` rows (see above), so the table still tells you what to use; treat `ReplacedBySynthetic` as "the derived feature on this row carried this raw one's signal better," and re-engineer the winning synthetics into your own pipeline before dropping their constituents.

- **Synthetic result rows carry only the competition's SHAP importance.** A winning synthetic's `Keep` row reports the SHAP importance it earned in the re-fit competition; columns that don't apply to a freshly synthesized feature (e.g. permutation importance, mutual-information score from the main run) are left as `NaN`. The full candidate detail (operator, constituents, MI score) is still in `<output_stem>_suggested_features.csv`.

- **Ratios are skipped on unstable denominators.** A candidate `A__div__B` is dropped when any value of B is exactly zero, or is negligibly small relative to B's own scale (which would produce an extreme, unstable ratio). Some interacting pairs will therefore only yield a product candidate.

- **DFE needs SHAP interactions.** Without `compute_shap_interactions=True`, the whole DFE step is skipped and a warning is logged.

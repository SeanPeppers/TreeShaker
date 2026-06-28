# Leakage detection

Seven independent checks flag features that may be causing data leakage or are otherwise logically suspect. A feature can trigger more than one. Leakage warnings produce a `Caution` recommendation on their own; combined with harm detection they escalate to `Drop`.

## The seven checks

| Check | Trigger | Why it matters |
|---|---|---|
| **Target correlation** | `\|r\|` with target ≥ `--target-corr-threshold` (default `0.95`) | Feature may have been derived from the target |
| **Categorical target association** | Bias-corrected Cramer's V with target ≥ `--categorical-assoc-threshold` (default `0.8`) | Non-numeric feature may be a near-perfect proxy for the target |
| **SHAP dominance** | One feature accounts for ≥ `--dominant-shap-fraction` (default `0.7`) of total SHAP | A single feature this dominant may leak future/target info |
| **Near-constant** | Same value in > 99% of rows | No signal, wastes model capacity |
| **Near-duplicate** | Correlation ≥ 0.98 with another feature | Redundant — adds no extra signal |
| **Multicollinearity (VIF)** | Variance Inflation Factor ≥ `--vif-threshold` (default `10.0`) | Feature is a near-linear combination of other features |
| **Suspicious names** | Matches patterns like `_id`, `future_`, `label`, `target` | Likely an identifier or future-leaking column |

## Target correlation check

The target correlation check only applies to numeric features (Pearson correlation is undefined for categoricals).

```bash
treeshaker --input data.csv --target price --task regression \
  --target-corr-threshold 0.95
```

## Categorical target association check

For classification tasks (`binary`/`multiclass`), the categorical target association check fills the gap for non-numeric features using bias-corrected Cramer's V — this catches leakage like the Titanic dataset's `boat` column (lifeboat assignment, recorded only for survivors and thus a near-perfect proxy for `survived`), which is non-numeric and so was previously invisible to the correlation check.

Missing values are treated as their own category rather than dropped — for proxy columns like `boat`, the presence/absence of a value is itself the leakage signal.

Regression targets are not currently covered by this check.

## SHAP dominance check

Flags a single feature if it accounts for more than `--dominant-shap-fraction` (default `0.7`, i.e. 70%) of the total SHAP importance across all features. A single feature this dominant may leak future information or be a near-perfect proxy for the target.

```bash
treeshaker --input data.csv --target churn --task binary \
  --dominant-shap-fraction 0.7
```

## Multicollinearity (VIF) check

Variance Inflation Factor measures how much a feature's variance is inflated by linear dependence with other features. High VIF indicates a feature is a near-linear combination of the others — redundant and likely to hurt generalization.

```bash
treeshaker --input data.csv --target price --task regression \
  --vif-threshold 10.0
```

Note: This is a pairwise correlation check (Pearson or Spearman depending on dtype). It can catch redundancy that simple pairwise correlation alone might miss.

## Suspicious names check

Flags features with column names matching patterns like:

- `_id` / `_identifier` / `_key` (likely row identifiers)
- `future_` / `next_` / `tomorrow_` (likely future information)
- `_label` / `_target` / `_outcome` (likely the target itself)

Case-insensitive substring match. Configurable via `LeakageConfig.suspicious_name_patterns`.

## Temporal feature exemption

Time-series engineered features — lags, rolling/expanding windows, exponentially-weighted moving averages, shifts, diffs — are *expected* to correlate highly with the target. That's normal signal, not leakage, so flagging it would just create noise.

Feature names matching a temporal pattern (`lag_`, `rolling_`, `ewm_`, `shift_`, `diff_`, `expanding_` by default, case-insensitive substring match) are exempted from the target-correlation check.

Exempted features aren't given a free pass entirely — instead they get a structural check via `LeakageDetector.detect_temporal_structure()`: the leading-NaN count is compared against the lag/window offset parsed from the feature name (e.g. `lag_7` should have 7 leading NaNs), and any unexpected NaN gaps mid-series are flagged. A mismatch can indicate an off-by-one or windowing bug that leaks future information into earlier rows. This produces a **temporal structure warning**, a category distinct from leakage warnings — it's surfaced informationally in the `reason` column without changing the `Drop`/`Caution`/`Keep` recommendation on its own.

Set `LeakageConfig(exempt_temporal_features=False)` to disable the exemption entirely:

```python
from treeshaker.leakage_detection import LeakageConfig

config = Config(
    input_file="data.csv",
    target_column="churn",
    task="binary",
    leakage_config=LeakageConfig(exempt_temporal_features=False),
)
```

## Disabling leakage checks

Disable all seven checks with `--skip-leakage-check`:

```bash
treeshaker --input data.csv --target churn --task binary --skip-leakage-check
```

Or via the Python API:

```python
config = Config(
    input_file="data.csv",
    target_column="churn",
    task="binary",
    skip_leakage_check=True,
)
```

## How leakage detection runs

Unlike importance evaluation (SHAP, permutation importance, mutual information — all computed on the held-out `test` set), leakage detection is a statistical property check across the *entire* dataset, not a held-out-generalization concern. So these checks intentionally run against `train` + `val` + `test` combined, to see the full dataset's correlation/VIF/duplicate statistics.

This is correct and intentional: a feature that's a near-duplicate of another or has suspiciously high VIF is equally problematic regardless of which split it appears in.

## Configuration

Common configurations to adjust leakage detection:

```python
from treeshaker.leakage_detection import LeakageConfig

config = Config(
    input_file="data.csv",
    target_column="price",
    task="regression",
    leakage_config=LeakageConfig(
        target_corr_threshold=0.95,
        categorical_assoc_threshold=0.8,
        dominant_shap_fraction=0.7,
        vif_threshold=10.0,
        exempt_temporal_features=True,
    ),
)
```

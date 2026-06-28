# How it works

TreeShaker **automatically** splits your data three ways: **train**, **validation**, and **test** — you don't need to pre-split anything yourself.

## The three-way split

- **`--test-size`** (default `0.2`) is held out from the *full* dataset as the final test set.
- **`--val-size`** (default `0.2`) is then taken from the *remaining* train+val pool for validation.
- The **test set is never used to fit any model** — not during tuning, not in the final fit — so importance numbers reflect genuine out-of-sample behavior rather than performance on data the model has already seen.

This automatic, rigorous split is what makes every importance number downstream trustworthy: nothing in the ranking is computed on data the model was trained on.

## Pipeline steps

### 1. Tune (optional)

Optuna searches for the best XGBoost hyperparameters, training on `train` and evaluating each trial on `val` (model selection / early-stopping signal).

This step is optional: pass `--skip-tuning` to skip it entirely and train directly with the default or manually-specified (`--xgboost-params`) XGBoost parameters, which is useful for quick runs.

### 2. Train

A final XGBoost model is fit on `train` + `val` combined (more data for the production-quality model), using the tuned (or directly configured) parameters. `test` is never included in this fit.

### 3. Rank

SHAP values, permutation importance, and mutual information are computed for every feature, evaluated against the held-out `test` set.

### 4. Flag

Features are marked harmful if their SHAP importance is near zero *or* if shuffling them doesn't degrade the model (i.e., they carry no signal). Separate checks flag leakage/suspicious patterns independently.

### 5. Output

Results saved as a ranked CSV or HTML report, with recommendations and plain-English explanations for each feature.

## Why this matters

**Prior behavior (versions < 0.1.0):**

TreeShaker used only a train/validation split, and the final model's importance/SHAP/permutation/mutual-information scores were computed on `val_data` — the *same* validation data already included in the final model's training fit. That meant importance was measured on data the model had partially memorized, not genuinely held out, which could inflate importance scores (especially permutation importance and mutual information).

**The fix:**

The 3-way split fixes this: importance is now computed on a `test` set that is never fit on by any model. If you were relying on `--test-size` to mean "validation fraction," note its meaning has changed to "held-out final test fraction" — use the new `--val-size` flag to control the validation fraction.

**Expected change:**

Numeric importance values will differ from earlier versions, even with the same `--test-size`. Generally they'll be less noisy and more stable, but they're a real behavior change worth knowing about if you're comparing results across versions.

## Task types and metrics

TreeShaker automatically selects the matching evaluation metric per task type (you must pass `--task` explicitly):

| Task | Target column | Evaluation metric | For Optuna |
|---|---|---|---|
| `binary` | 2 unique values | ROC-AUC | ROC-AUC (tuning) |
| `multiclass` | 3+ unique values | Accuracy | Accuracy (tuning) |
| `regression` | Numeric | Negative MSE | Neg MSE (tuning) |

You can override the metric used for permutation importance only (not Optuna tuning) with `--permutation-metric` (see [Configuration reference](config_reference.md)).

## Data preprocessing

TreeShaker validates and normalizes column dtypes automatically before anything is fit:

- **Numeric (int/float)** — works as-is, no special handling
- **Boolean** — works as-is, XGBoost and SHAP handle it natively
- **Low-cardinality categorical** — auto-cast to pandas `category` dtype, XGBoost 2.x fits natively
- **Datetime** — auto-derives year/month/day/dayofweek columns, removes raw column
- **Missing values** — works as-is in every code path; XGBoost learns default split directions
- **Free text** — raises clear error; pre-vectorize or embed before passing to TreeShaker

See [Data types](config_reference.md#supported-data-types) for full details.

## GPU acceleration

TreeShaker **automatically** uses a CUDA GPU for XGBoost when one is usable — no flag needed. On startup, it runs a cheap, cached real-fit check to confirm a CUDA device actually works; if none is found, it silently falls back to CPU.

Use `--device` to override:

- `--device auto` (default) — use GPU if available, else CPU
- `--device cuda` — force GPU; raises error if no usable GPU detected
- `--device cpu` — force CPU regardless of GPU availability

```bash
treeshaker --input data.csv --target churn --task binary --device cuda
```

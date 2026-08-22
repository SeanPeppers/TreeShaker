"""Benchmark: does TreeShaker's default Keep/Caution/Drop actually match ground truth?

Every correctness check in this repo's history so far has been internal
consistency ("does the code do what it claims"). Nobody had checked whether
the recommendations are actually GOOD feature selection against a dataset
where the right answer is objectively known. This script does that.

Method: sklearn.datasets.make_classification / make_regression accept
n_informative / n_redundant / n_repeated. Every feature beyond
n_informative + n_redundant + n_repeated is pure noise by construction
(shuffle=False so column order is preserved: informative, then redundant,
then repeated, then noise -- see sklearn docs). That's real ground truth:
    - informative/redundant/repeated features carry real signal -> Keep or
      Caution is a correct call; only Drop is a miss.
    - trailing noise features carry none -> Drop is correct; Keep/Caution
      is a miss.
"Drop" is scored as the positive prediction for "is noise" (precision/recall/
F1), and Keep+Caution as the positive prediction for "is signal" (the mirror
confusion matrix). Runs several dataset shapes (binary, imbalanced binary,
multiclass, low-noise regression, high-noise regression) x 3 seeds each,
at FeatureSelectionPipeline's DEFAULT config (skip_tuning=True only, to keep
runtime sane -- no importance/harm-detection threshold changed).

Run:
    nix-shell -p uv --run "uv run python scripts/benchmark_recommendation_quality.py"

=== RESULTS (2026-08-22, real run: 15 runs, 5 shapes x 3 seeds, 405 features) ===

Aggregate (micro-averaged over all features, all runs):
    Drop-for-noise:          precision 0.779  recall 0.602  F1 0.679
    Keep/Caution-for-signal: precision 0.387  recall 0.594  F1 0.469

Per-shape (mean F1 across 3 seeds):
    binary_balanced        Drop F1 0.766  Signal F1 0.147
    binary_imbalanced      Drop F1 0.000  Signal F1 0.500
    multiclass              Drop F1 0.793  Signal F1 0.494
    regression_low_noise   Drop F1 0.725  Signal F1 0.741
    regression_high_noise  Drop F1 0.742  Signal F1 0.357

Honest read: default TreeShaker is NOT reliably good at this task -- it has
two distinct, opposite failure modes depending on data shape, and neither is
a close call.

1. Balanced/multiclass classification with several correlated signal
   features (binary_balanced, multiclass): over-aggressive. drop_recall is a
   clean 1.000 (every noise feature IS dropped) but drop_precision is only
   ~0.6-0.7 and signal_recall collapses to 0.0-0.45 -- most of the true
   informative/redundant/repeated features get Dropped too, not just the
   noise. A single-run per-feature dump (30 features, 12 signal) confirmed
   the mechanism directly: 11 of 12 signal features were Dropped, all with
   the same reason -- "permutation importance below the 0.050 threshold".
   permutation_importance_threshold=0.05 is a fixed absolute bar that does
   not scale with feature count or shared/collinear signal: XGBoost splits
   credit for correlated features across all of them, so shuffling any ONE
   of several features carrying the same signal barely moves the model's
   score even though the feature is genuinely useful. Only 1 of 12 signal
   features survived (as Caution, via the near-duplicate/VIF leakage note,
   not via collinearity gating). Collinearity gating itself never activates
   at true defaults (collinearity_threshold=0.0 is opt-in), so it cannot and
   did not rescue anything here -- the leakage detector's own multicollinearity
   check is what appended the demoting note in that one case.

2. Severely imbalanced binary classification (binary_imbalanced, ~7-8%
   minority class): the opposite failure -- passive. drop_recall is exactly
   0.000 in all 3 seeds: not a single noise feature was ever Dropped.
   pipeline.py's severe_class_imbalance branch disables the permutation
   importance check entirely and falls back to SHAP-threshold-only harm
   detection; at this shape SHAP importance apparently clears
   shap_importance_threshold=0.01 for every feature, so nothing gets flagged
   harmful and the noise features just sit there as Keep/Caution.

3. Regression is the best-behaved case, though not perfect. drop_precision
   is a clean 1.000 in both regression shapes -- true signal was never
   incorrectly Dropped. But drop_recall is only 0.51-0.67 in the high-noise
   shape (45 of 50 features are noise): roughly half the noise features are
   left sitting as Keep/Caution rather than Dropped, so it under-cleans
   rather than over-cleans here.

Bottom line: the single fixed-threshold harm-detection design
(shap_importance_threshold=0.01, permutation_importance_threshold=0.05,
same numbers regardless of feature count or class balance) is the root
cause of both failure modes -- too strict when many features share signal
(over-drops), and effectively bypassed under severe imbalance (under-drops).
Regression avoids both because make_regression's coefficient-driven signal
doesn't get split across correlated columns the way classification's shared
decision boundary does, and regression never has a class-imbalance fallback
to trigger. This is real, reproducible signal that default TreeShaker's
thresholds need work, not just "good enough."
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from treeshaker.config import Config
from treeshaker.pipeline import FeatureSelectionPipeline

SEEDS = (0, 1, 2)


@dataclass
class DatasetSpec:
    name: str
    task: str  # "binary" | "multiclass" | "regression"
    n_samples: int
    n_features: int
    n_informative: int
    n_redundant: int = 0
    n_repeated: int = 0
    n_classes: int = 2
    weights: list[float] | None = None
    noise: float = 10.0  # regression only


SPECS = [
    DatasetSpec(
        "binary_balanced",
        "binary",
        n_samples=3000,
        n_features=30,
        n_informative=6,
        n_redundant=4,
        n_repeated=2,
    ),
    DatasetSpec(
        "binary_imbalanced",
        "binary",
        n_samples=3000,
        n_features=30,
        n_informative=6,
        n_redundant=4,
        weights=[0.93, 0.07],
    ),
    DatasetSpec(
        "multiclass",
        "multiclass",
        n_samples=3000,
        n_features=25,
        n_informative=8,
        n_redundant=3,
        n_classes=4,
    ),
    DatasetSpec(
        "regression_low_noise",
        "regression",
        n_samples=2000,
        n_features=20,
        n_informative=8,
        noise=2.0,
    ),
    DatasetSpec(
        "regression_high_noise",
        "regression",
        n_samples=2000,
        n_features=50,
        n_informative=5,
        noise=15.0,
    ),
]


def build_dataset(spec: DatasetSpec, seed: int) -> tuple[pd.DataFrame, int]:
    """Generate one dataset for `spec`/`seed`. Returns (df, n_signal_features).

    Columns are named f0..f{n-1} in the order sklearn emits them:
    informative, then redundant, then repeated, then pure noise
    (shuffle=False preserves that order -- required for ground truth to
    line up with column index).
    """
    n_signal = spec.n_informative + spec.n_redundant + spec.n_repeated
    if spec.task == "regression":
        X, y = make_regression(
            n_samples=spec.n_samples,
            n_features=spec.n_features,
            n_informative=spec.n_informative,
            noise=spec.noise,
            shuffle=False,
            random_state=seed,
        )
    else:
        X, y = make_classification(
            n_samples=spec.n_samples,
            n_features=spec.n_features,
            n_informative=spec.n_informative,
            n_redundant=spec.n_redundant,
            n_repeated=spec.n_repeated,
            n_classes=spec.n_classes,
            n_clusters_per_class=1,
            weights=spec.weights,
            class_sep=1.5,
            shuffle=False,
            random_state=seed,
        )
    cols = [f"f{i}" for i in range(spec.n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    return df, n_signal


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def run_one(spec: DatasetSpec, seed: int, tmpdir: Path) -> dict:
    df, n_signal = build_dataset(spec, seed)
    input_csv = tmpdir / f"{spec.name}_{seed}.csv"
    output_csv = tmpdir / f"{spec.name}_{seed}_out.csv"
    df.to_csv(input_csv, index=False)

    config = Config(
        input_file=str(input_csv),
        target_column="target",
        task=spec.task,
        output_file=str(output_csv),
        skip_tuning=True,
        random_seed=seed,
    )
    pipeline = FeatureSelectionPipeline(config)
    start = time.time()
    results = pipeline.run()
    elapsed = time.time() - start

    feature_idx = results["feature_name"].str.removeprefix("f").astype(int)
    actual_noise = (feature_idx >= n_signal).to_numpy()
    predicted_drop = (results["recommendation"] == "Drop").to_numpy()

    tp = int(np.sum(predicted_drop & actual_noise))
    fp = int(np.sum(predicted_drop & ~actual_noise))
    fn = int(np.sum(~predicted_drop & actual_noise))
    tn = int(np.sum(~predicted_drop & ~actual_noise))

    drop_p, drop_r, drop_f1 = precision_recall_f1(tp, fp, fn)
    # Mirror confusion: Keep+Caution predicting "is signal".
    signal_p, signal_r, signal_f1 = precision_recall_f1(tn, fn, fp)

    return {
        "name": spec.name,
        "seed": seed,
        "task": spec.task,
        "n_features": spec.n_features,
        "n_signal": n_signal,
        "n_noise": spec.n_features - n_signal,
        "severe_imbalance": pipeline.severe_class_imbalance,
        "elapsed_s": elapsed,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "drop_precision": drop_p,
        "drop_recall": drop_r,
        "drop_f1": drop_f1,
        "signal_precision": signal_p,
        "signal_recall": signal_r,
        "signal_f1": signal_f1,
    }


def print_report(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)

    print("\n=== Per-run results ===")
    cols = [
        "name",
        "seed",
        "task",
        "n_features",
        "n_signal",
        "n_noise",
        "severe_imbalance",
        "drop_precision",
        "drop_recall",
        "drop_f1",
        "signal_precision",
        "signal_recall",
        "signal_f1",
        "elapsed_s",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[cols].round(3).to_string(index=False))

    print("\n=== Per-shape mean F1 (across seeds) ===")
    per_shape = df.groupby("name")[["drop_f1", "signal_f1"]].mean().round(3)
    print(per_shape.to_string())

    print("\n=== Aggregate (micro-averaged over all features, all runs) ===")
    tp, fp, fn, tn = df["tp"].sum(), df["fp"].sum(), df["fn"].sum(), df["tn"].sum()
    drop_p, drop_r, drop_f1 = precision_recall_f1(tp, fp, fn)
    signal_p, signal_r, signal_f1 = precision_recall_f1(tn, fn, fp)
    print(
        f"  Drop-for-noise:          precision {drop_p:.3f}"
        f"  recall {drop_r:.3f}  F1 {drop_f1:.3f}"
    )
    print(
        f"  Keep/Caution-for-signal: precision {signal_p:.3f}"
        f"  recall {signal_r:.3f}  F1 {signal_f1:.3f}"
    )
    print(f"  Total elapsed: {df['elapsed_s'].sum():.1f}s across {len(df)} runs")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        rows = []
        for spec in SPECS:
            for seed in SEEDS:
                print(f"Running {spec.name} seed={seed}...")
                rows.append(run_one(spec, seed, tmpdir))
    print_report(rows)


if __name__ == "__main__":
    main()

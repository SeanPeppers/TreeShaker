#!/usr/bin/env python3
"""
Validation script for LSH-based approximate leakage screening.

Builds a wide synthetic-but-realistic dataset from fetch_california_housing
by duplicating and jittering columns up to a few hundred features, then
compares approximate vs exact near-duplicate detection results and timing.
"""

import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

from treeshaker.leakage_detection import LeakageDetector, LeakageConfig


def build_wide_dataset(n_features: int = 300, n_duplicates: int = 50) -> pd.DataFrame:
    """Build a wide synthetic-but-realistic dataset with near-duplicates.

    Args:
        n_features: Target number of numeric features.
        n_duplicates: Number of intentional near-duplicate pairs.

    Returns:
        DataFrame with ~n_features numeric columns + 'target'.
    """
    # Load real data as the base
    housing = fetch_california_housing()
    df_base = pd.DataFrame(housing.data, columns=housing.feature_names)

    # Repeat and jitter columns to get to ~n_features
    df = df_base.copy()
    rng = np.random.default_rng(42)

    # Add pure duplicates with slight jitter (simulate leakage)
    for i in range(n_duplicates):
        base_col = df_base.iloc[:, i % len(df_base.columns)]
        jitter = rng.normal(0, 0.001 * base_col.std(), len(df_base))
        df[f"dup_{i}_a"] = base_col + jitter
        df[f"dup_{i}_b"] = base_col + jitter * 0.5 + rng.normal(0, 0.0005 * base_col.std(), len(df_base))

    # Add independent random features to reach target count
    target_features = n_features
    current_features = len([c for c in df.columns])
    while current_features < target_features:
        col_name = f"random_{current_features}"
        df[col_name] = rng.normal(0, 1, len(df))
        current_features += 1

    # Add target (continuous for regression or binary for classification)
    df["target"] = rng.normal(10, 5, len(df))

    return df


def main() -> None:
    """Validate LSH screening against exact screening."""
    print("Building wide dataset with intentional near-duplicates...")
    df = build_wide_dataset(n_features=300, n_duplicates=50)
    feature_names = [c for c in df.columns if c != "target"]
    print(f"  Dataset shape: {df.shape}")
    print(f"  Feature count: {len(feature_names)}")

    # Dummy SHAP values
    shap_values = np.random.default_rng(42).exponential(0.1, len(feature_names))
    shap_values = shap_values / shap_values.sum()

    # Test exact path
    print("\nRunning exact path screening...")
    start_exact = time.time()
    detector_exact = LeakageDetector(
        LeakageConfig(near_duplicate_corr_threshold=0.95),
        screening_mode="exact",
    )
    warnings_exact = detector_exact.detect(
        df, "target", feature_names, shap_values, task="regression"
    )
    elapsed_exact = time.time() - start_exact
    flagged_exact = {k for k, v in warnings_exact.items() if v}
    print(f"  Time elapsed: {elapsed_exact:.2f}s")
    print(f"  Features flagged: {len(flagged_exact)}")

    # Test approximate path
    print("\nRunning approximate path screening...")
    start_approx = time.time()
    detector_approx = LeakageDetector(
        LeakageConfig(near_duplicate_corr_threshold=0.95),
        screening_mode="approximate",
    )
    warnings_approx = detector_approx.detect(
        df, "target", feature_names, shap_values, task="regression"
    )
    elapsed_approx = time.time() - start_approx
    flagged_approx = {k for k, v in warnings_approx.items() if v}
    print(f"  Time elapsed: {elapsed_approx:.2f}s")
    print(f"  Features flagged: {len(flagged_approx)}")

    # Compare results
    print("\n--- Comparison Results ---")
    print(f"Exact path flagged: {len(flagged_exact)}")
    print(f"Approximate path flagged: {len(flagged_approx)}")
    print(f"Speedup (exact/approx): {elapsed_exact / elapsed_approx:.1f}x")

    # Compute overlap
    overlap = flagged_exact & flagged_approx
    if flagged_exact:
        recall = len(overlap) / len(flagged_exact)
        print(f"Approximate recall (% of exact found): {recall * 100:.1f}%")

    print(f"\nNote: Approximate LSH may miss some weak-correlation pairs due to")
    print(f"hash collisions, but should find most high-correlation duplicates.")
    print(f"\nFull validation:")
    print(f"  Exact path detected {len(flagged_exact)} near-duplicate features")
    print(f"  Approximate path detected {len(flagged_approx)} near-duplicate features")
    print(f"  Speedup: {elapsed_exact / elapsed_approx:.1f}x (approx is faster)")


if __name__ == "__main__":
    main()

"""Collinearity-aware grouped feature selection."""
from __future__ import annotations
import numpy as np
import pandas as pd


def cluster_correlated_features(
    X: pd.DataFrame,
    feature_names: list[str],
    threshold: float,
    method: str = 'spearman',
) -> list[list[str]]:
    if threshold <= 0 or len(feature_names) < 2:
        return [[f] for f in feature_names]
    subset = X[feature_names]
    corr = subset.corr(method=method).abs()
    n = len(feature_names)
    parent = list(range(n))
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if corr.iloc[i, j] >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups: dict[int, list[str]] = {}
    for i, name in enumerate(feature_names):
        root = find(i)
        groups.setdefault(root, []).append(name)
    return list(groups.values())


def select_cluster_representative(
    cluster: list[str],
    shap_importance: dict[str, float],
) -> str:
    return max(cluster, key=lambda f: shap_importance.get(f, 0.0))


def compute_grouped_permutation_importance(
    model: object,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    clusters: list[list[str]],
    task: str,
    n_repeats: int = 5,
    random_seed: int = 42,
) -> dict[str, float]:
    from .importance import _score_model
    rng = np.random.default_rng(random_seed)
    baseline = _score_model(model, X_val, y_val, task)
    result: dict[str, float] = {}
    for cluster in clusters:
        if len(cluster) == 1:
            result[cluster[0]] = 0.0
            continue
        drops = []
        X_perm = X_val.copy()
        for _ in range(n_repeats):
            saved = X_perm[cluster].to_numpy().copy()
            idx = rng.permutation(len(X_perm))
            X_perm[cluster] = X_perm[cluster].iloc[idx].to_numpy()
            score = _score_model(model, X_perm, y_val, task)
            drops.append(baseline - score)
            X_perm[cluster] = saved
        importance = float(np.mean(drops))
        for feat in cluster:
            result[feat] = importance
    return result

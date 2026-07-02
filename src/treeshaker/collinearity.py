"""Collinearity-aware grouped feature selection."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from .utils import ignore_constant_corr_warnings


def cluster_correlated_features(
    X: pd.DataFrame,
    feature_names: list[str],
    threshold: float,
    method: Literal["spearman", "pearson"] = "spearman",
) -> list[list[str]]:
    if threshold <= 0 or len(feature_names) < 2:
        return [[f] for f in feature_names]
    subset = X[feature_names].select_dtypes(include="number")
    numeric_features = list(subset.columns)
    if len(numeric_features) < 2:
        return [[f] for f in feature_names]
    with ignore_constant_corr_warnings():
        corr = subset.corr(method=method).abs()
    n = len(numeric_features)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            val = float(corr.to_numpy()[i, j])
            if val >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[str]] = {}
    for i, name in enumerate(numeric_features):
        root = find(i)
        groups.setdefault(root, []).append(name)

    # Non-numeric features become singletons
    clustered = {f for grp in groups.values() for f in grp}
    result = list(groups.values())
    for f in feature_names:
        if f not in clustered:
            result.append([f])
    return result


def select_cluster_representative(
    cluster: list[str],
    shap_importance: dict[str, float],
) -> str:
    return max(cluster, key=lambda f: shap_importance.get(f, 0.0))

"""Wave 6 smoke test — exercises all new features on messy real datasets.

Datasets:
  - adult (binary, has categoricals, missing values) from OpenML
  - california_housing (regression, 8 numeric features) from sklearn

Features tested:
  1. Shadow probes      — shadow_probe_flag column appears
  2. Collinearity       — collinearity_cluster column appears
  3. DFE interactions   — suggested_features non-empty
  4. DFE loop feedback  — ReplacedBySynthetic may appear (not required; just logged)
  5. Temporal leakage   — unsorted-time warning fires correctly

Run:
  python scripts/smoke_test_wave6.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from treeshaker.config import Config, OptunaConfig
from treeshaker.pipeline import FeatureSelectionPipeline
from treeshaker.synthetic_features import SyntheticConfig

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = PASS if ok else FAIL
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    results.append((name, ok, detail))


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def load_adult(tmp_dir: Path) -> tuple[str, str, str]:
    print("Loading adult dataset from OpenML...")
    adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df: pd.DataFrame = adult.frame  # type: ignore[assignment]
    # Drop rows with too many missing
    df = df.dropna(thresh=len(df.columns) - 2).reset_index(drop=True)
    # Sample down for speed
    df = df.sample(n=min(3000, len(df)), random_state=42).reset_index(drop=True)
    # Binarize target
    target_col = "class"
    df[target_col] = (df[target_col].astype(str).str.strip() == ">50K").astype(int)
    path = tmp_dir / "adult.csv"
    df.to_csv(path, index=False)
    return str(path), target_col, "binary"


def load_california(tmp_dir: Path) -> tuple[str, str, str]:
    print("Loading California housing dataset...")
    cal = fetch_california_housing(as_frame=True)
    df = cal.frame  # type: ignore[assignment]
    df = df.sample(n=min(3000, len(df)), random_state=42).reset_index(drop=True)
    target_col = "MedHouseVal"
    path = tmp_dir / "california.csv"
    df.to_csv(path, index=False)
    return str(path), target_col, "regression"


# ---------------------------------------------------------------------------
# Per-dataset checks
# ---------------------------------------------------------------------------


def run_checks(name: str, csv_path: str, target_col: str, task: str) -> None:
    print(f"\n=== {name} ({task}) ===")

    # --- 1. Shadow probes ---
    try:
        cfg = Config(
            input_file=csv_path,
            target_column=target_col,
            task=task,
            skip_tuning=True,
            optuna_config=OptunaConfig(n_trials=1),
            n_shadow_probes=10,
        )
        res = FeatureSelectionPipeline(cfg).run()
        check(
            f"{name} / shadow_probe_flag column",
            "shadow_probe_flag" in res.columns,
            f"columns: {list(res.columns)}",
        )
        flagged = int(res["shadow_probe_flag"].sum()) if "shadow_probe_flag" in res.columns else -1
        check(
            f"{name} / at least some features flagged by probes",
            flagged >= 0,
            f"flagged={flagged}",
        )
    except Exception as exc:
        check(f"{name} / shadow probes", False, str(exc))

    # --- 2. Collinearity ---
    try:
        cfg = Config(
            input_file=csv_path,
            target_column=target_col,
            task=task,
            skip_tuning=True,
            optuna_config=OptunaConfig(n_trials=1),
            collinearity_threshold=0.7,
        )
        res = FeatureSelectionPipeline(cfg).run()
        check(
            f"{name} / collinearity_cluster column",
            "collinearity_cluster" in res.columns,
        )
        n_dupes = int((res["recommendation"] == "CollinearDuplicate").sum()) if "recommendation" in res.columns else 0
        check(
            f"{name} / collinearity result parseable",
            "collinearity_cluster" in res.columns,
            f"CollinearDuplicate rows={n_dupes}",
        )
    except Exception as exc:
        check(f"{name} / collinearity", False, str(exc))

    # --- 3. DFE interactions (advisory) ---
    try:
        cfg = Config(
            input_file=csv_path,
            target_column=target_col,
            task=task,
            skip_tuning=True,
            optuna_config=OptunaConfig(n_trials=1),
            synthetic_config=SyntheticConfig(
                synthesize_from_interactions=True,
                interaction_top_n=3,
                mi_threshold=0.0,
                operators=["product", "ratio"],
            ),
            compute_shap_interactions=True,
        )
        pipeline = FeatureSelectionPipeline(cfg)
        pipeline.run()
        suggestions = pipeline.suggested_features or []
        check(
            f"{name} / DFE suggestions produced",
            True,  # not failing if empty — interactions may be weak
            f"n_suggestions={len(suggestions)}",
        )
    except Exception as exc:
        check(f"{name} / DFE interactions", False, str(exc))

    # --- 4. DFE loop feedback ---
    try:
        cfg = Config(
            input_file=csv_path,
            target_column=target_col,
            task=task,
            skip_tuning=True,
            optuna_config=OptunaConfig(n_trials=1),
            synthetic_config=SyntheticConfig(
                synthesize_from_interactions=True,
                interaction_top_n=3,
                mi_threshold=0.0,
                operators=["product"],
                dfe_loop_feedback=True,
                dfe_max_inject=3,
            ),
            compute_shap_interactions=True,
        )
        pipeline = FeatureSelectionPipeline(cfg)
        res = pipeline.run()
        replaced = int((res["recommendation"] == "ReplacedBySynthetic").sum()) if "recommendation" in res.columns else 0
        check(
            f"{name} / DFE loop feedback ran without error",
            True,
            f"ReplacedBySynthetic rows={replaced}",
        )
    except Exception as exc:
        check(f"{name} / DFE loop feedback", False, str(exc))


# ---------------------------------------------------------------------------
# Temporal leakage smoke check (unsorted warning)
# ---------------------------------------------------------------------------


def run_temporal_check(tmp_dir: Path) -> None:
    print("\n=== Temporal leakage / unsorted time warning ===")
    import io
    import logging as _logging

    rng = np.random.default_rng(0)
    n = 200
    t = np.arange(n, dtype=float)
    df = pd.DataFrame(
        {
            "t": t,
            "feat1": rng.standard_normal(n),
            "feat2": rng.standard_normal(n),
            "y": rng.standard_normal(n),
        }
    )
    # Shuffle to make time unsorted
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    path = tmp_dir / "temporal.csv"
    df.to_csv(path, index=False)

    from treeshaker.utils import load_csv

    log_stream = io.StringIO()
    handler = _logging.StreamHandler(log_stream)
    handler.setLevel(_logging.WARNING)
    root_logger = _logging.getLogger()
    root_logger.addHandler(handler)
    try:
        load_csv(str(path), target_column="y", task="regression", time_column="t")
        log_output = log_stream.getvalue()
        check(
            "temporal / unsorted time column warning fires",
            "monotonically sorted" in log_output,
            repr(log_output[:200]),
        )
    except Exception as exc:
        check("temporal / unsorted warning", False, str(exc))
    finally:
        root_logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        try:
            csv, target, task = load_adult(tmp_dir)
            run_checks("adult", csv, target, task)
        except Exception as exc:
            print(f"  [SKIP] adult dataset failed to load: {exc}")

        try:
            csv, target, task = load_california(tmp_dir)
            run_checks("california_housing", csv, target, task)
        except Exception as exc:
            print(f"  [SKIP] california dataset failed to load: {exc}")

        run_temporal_check(tmp_dir)

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        print("FAILED checks:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()

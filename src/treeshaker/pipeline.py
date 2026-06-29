"""Core feature selection pipeline orchestration."""

import dataclasses
import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .dtype_handling import prepare_feature_dtypes_splits
from .harm_detection import HarmDetector
from .importance import (
    compute_importance_divergence,
    compute_mutual_information,
    compute_permutation_importance,
    compute_shap_importance,
    compute_shap_importance_delta,
    compute_shap_interactions,
)
from .leakage_detection import LeakageDetector
from .residual_diagnostics import (
    compute_residual_correlation,
    compute_residuals,
    flag_residual_correlated_features,
)
from .synthetic_features import SyntheticFeatureSuggestion, suggest_synthetic_features
from .utils import (
    check_class_imbalance,
    derive_output_stem,
    get_metric_fn,
    load_csv,
    resolve_xgboost_device,
    save_results,
)


def _build_xgb_fit_params(
    base_params: dict, config: Config, random_seed: int | None = None
) -> dict:
    """Build the final XGBoost constructor kwargs shared across all fit sites.

    Merges `base_params` (tuned/default hyperparameters) with the fields
    that are always set the same way regardless of which pipeline stage is
    fitting a model (Optuna's objective, the final train_model fit, and
    each bootstrap importance refit): a deterministic random_state,
    silent verbosity, native pandas-categorical support, and the resolved
    training device. config.xgboost_config.extra_params is applied last so
    user overrides win.

    Args:
        base_params: Hyperparameter dict to start from (e.g. Optuna's
            suggested values, or config.xgboost_config.to_dict()).
        config: The pipeline Config, used for random_seed (when
            random_seed is not given), xgboost_config.device, and
            xgboost_config.extra_params.
        random_seed: Override for "random_state". Defaults to
            config.random_seed; bootstrap refits pass a per-iteration seed
            instead.

    Returns:
        A new dict ready to pass as XGBClassifier/XGBRegressor kwargs.
    """
    dev = resolve_xgboost_device(config.xgboost_config.device, use_gpu=config.use_gpu)
    params = {
        **base_params,
        "random_state": (config.random_seed if random_seed is None else random_seed),
        "verbosity": 0,
        # Lets XGBoost natively consume pandas `category` dtype columns
        # (produced by dtype_handling.prepare_feature_dtypes_splits for
        # string features) without pre-encoding. A no-op when there are no
        # categorical columns.
        "enable_categorical": True,
        # device="cuda" alone is sufficient for GPU training on modern
        # XGBoost; tree_method="gpu_hist" was removed and is no longer a
        # valid value (XGBoost picks "hist" automatically for either device).
        "device": dev,
    }
    params.update(config.xgboost_config.extra_params)
    return params


def _make_xgb_model(task: str, params: dict) -> Any:
    """Instantiate the right XGBoost estimator class for a task.

    Args:
        task: One of "binary", "multiclass", or "regression".
        params: Constructor kwargs, typically from _build_xgb_fit_params().

    Returns:
        An (unfitted) XGBClassifier (binary/multiclass) or XGBRegressor
        (regression) instance.
    """
    import xgboost as xgb

    if task in ("binary", "multiclass"):
        objective_str = "binary:logistic" if task == "binary" else "multi:softprob"
        p = params.copy()
        p.setdefault("objective", objective_str)
        return xgb.XGBClassifier(**p)
    return xgb.XGBRegressor(**params)


def _ordinal_encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode pandas `category` columns for sklearn estimators.

    sklearn's RFECV and RandomForest reject the categorical dtype XGBoost
    consumes natively elsewhere in this pipeline; this is a minimal,
    model-agnostic stand-in just for those calls.
    """
    X = X.copy()
    for col in X.select_dtypes(include="category").columns:
        X[col] = X[col].cat.codes
    return X


def run_boruta_shap(
    model: Any,
    data: pd.DataFrame,
    target_column: str,
    feature_names: list[str],
    task: str,
    random_seed: int = 42,
    n_iterations: int = 20,
) -> list[str]:
    """Run Boruta feature selection and return decisions per feature."""
    from scipy.stats import binomtest

    if not feature_names:
        return []

    X = data[feature_names].copy()
    y = data[target_column]
    n_features = len(feature_names)
    hits = np.zeros(n_features, dtype=int)

    base_params = model.get_params()

    for i in range(n_iterations):
        iter_seed = random_seed + i
        rng = np.random.default_rng(iter_seed)

        X_shadow = pd.DataFrame(index=X.index)
        for col in feature_names:
            X_shadow[f"_shadow_{col}"] = pd.Series(
                rng.permutation(np.asarray(X[col].values)),
                index=X.index,
                dtype=X[col].dtype,
            )

        X_combined = pd.concat([X, X_shadow], axis=1)

        iter_params = {**base_params, "random_state": iter_seed}
        iter_model = _make_xgb_model(task, iter_params)
        iter_model.fit(X_combined, y)

        importances = iter_model.feature_importances_
        real_importances = importances[:n_features]
        shadow_importances = importances[n_features:]

        max_shadow_importance = (
            float(np.max(shadow_importances)) if len(shadow_importances) > 0 else 0.0
        )

        hits += (real_importances > max_shadow_importance).astype(int)

    decisions = []
    for h in hits:
        res = binomtest(int(h), n_iterations, 0.5)
        hit_rate = h / n_iterations
        if res.pvalue < 0.05 and hit_rate > 0.5:
            decisions.append("Confirmed")
        elif res.pvalue < 0.05 and hit_rate < 0.5:
            decisions.append("Rejected")
        else:
            decisions.append("Tentative")

    return decisions


def run_rfecv(
    model: Any,
    data: pd.DataFrame,
    target_column: str,
    feature_names: list[str],
    task: str,
    random_seed: int = 42,
) -> list[int]:
    """Run sklearn RFECV and return feature ranking (1 = selected)."""
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import KFold, StratifiedKFold

    X = _ordinal_encode_categoricals(data[feature_names])
    y = data[target_column]

    if task in ("binary", "multiclass"):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        scoring = "roc_auc" if task == "binary" else "accuracy"
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)
        scoring = "neg_mean_squared_error"

    selector = RFECV(
        estimator=model,
        step=1,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )
    selector.fit(X, y)
    return [int(r) for r in selector.ranking_]


def run_lightgbm_importance(
    data: pd.DataFrame,
    target_column: str,
    feature_names: list[str],
    task: str,
    random_seed: int = 42,
) -> list[float]:
    """Fit LightGBM model and return normalized feature importances."""
    try:
        import lightgbm as lgb
    except ImportError as err:
        raise ImportError(
            "The 'lightgbm' package is required when use_lightgbm=True."
            " Install it with: pip install lightgbm"
        ) from err

    X = data[feature_names]
    y = data[target_column]

    if task in ("binary", "multiclass"):
        model = lgb.LGBMClassifier(random_state=random_seed, verbosity=-1)
    else:
        model = lgb.LGBMRegressor(random_state=random_seed, verbosity=-1)

    model.fit(X, y)
    importances = model.feature_importances_
    total = float(sum(importances))
    if total > 0:
        return [float(i / total) for i in importances]
    return [float(i) for i in importances]


def run_catboost_importance(
    data: pd.DataFrame,
    target_column: str,
    feature_names: list[str],
    task: str,
    random_seed: int = 42,
) -> list[float]:
    """Fit CatBoost model with native categorical handling and return importances."""
    try:
        import catboost as cb
    except ImportError as err:
        raise ImportError(
            "The 'catboost' package is required when use_catboost=True."
            " Install it with: pip install catboost"
        ) from err

    X = data[feature_names].copy()
    y = data[target_column]

    cat_features = [
        col
        for col in feature_names
        if isinstance(X[col].dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(X[col])
        or pd.api.types.is_string_dtype(X[col])
    ]
    for col in cat_features:
        X[col] = X[col].astype(str).fillna("nan")

    if task in ("binary", "multiclass"):
        model = cb.CatBoostClassifier(
            random_seed=random_seed,
            verbose=0,
            cat_features=(cat_features if cat_features else None),
        )
    else:
        model = cb.CatBoostRegressor(
            random_seed=random_seed,
            verbose=0,
            cat_features=(cat_features if cat_features else None),
        )

    model.fit(X, y)
    importances = model.get_feature_importance()
    total = float(sum(importances))
    if total > 0:
        return [float(i / total) for i in importances]
    return [float(i) for i in importances]


def run_rf_baseline_importance(
    data: pd.DataFrame,
    target_column: str,
    feature_names: list[str],
    task: str,
    random_seed: int = 42,
) -> list[float]:
    """Fit Random Forest baseline model and return normalized feature importances."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    X = _ordinal_encode_categoricals(data[feature_names])
    y = data[target_column]

    if task in ("binary", "multiclass"):
        model = RandomForestClassifier(random_state=random_seed, n_jobs=1)
    else:
        model = RandomForestRegressor(random_state=random_seed, n_jobs=1)

    model.fit(X, y)
    importances = model.feature_importances_
    total = float(sum(importances))
    if total > 0:
        return [float(i / total) for i in importances]
    return [float(i) for i in importances]


def run_sklearn_permutation_importance(
    model: Any,
    data: pd.DataFrame,
    target_column: str,
    feature_names: list[str],
    task: str,
    n_permutations: int = 5,
    random_seed: int = 42,
) -> list[float]:
    """Compute model-agnostic sklearn permutation importance."""
    from sklearn.inspection import permutation_importance

    X = data[feature_names]
    y = data[target_column]

    if task in ("binary", "multiclass"):
        scoring = "roc_auc" if task == "binary" else "roc_auc_ovr"
    else:
        scoring = "neg_mean_squared_error"

    res = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_permutations,
        random_state=random_seed,
        n_jobs=1,
    )
    return [float(i) for i in res.importances_mean]


def log_run_to_mlflow(
    config: Config,
    results: pd.DataFrame,
    experiment_name: str = "treeshaker",
) -> None:
    """Log a completed TreeShaker run to MLflow for experiment tracking.

    Logs three things under a single MLflow run:
      - Parameters: the flat (non-nested-dataclass) Config fields that were
        actually used for this run, e.g. task, test_size, shap_importance_threshold,
        the use_* feature toggles, etc.
      - Metrics: per-feature importance/score columns already computed by the
        pipeline (e.g. shap_importance, permutation_importance, any opt-in
        columns like lightgbm_importance), logged as "<column>__<feature_name>"
        so every feature's value for each metric column is tracked individually.
      - Artifact: the full results DataFrame, saved as a CSV and uploaded.

    Args:
        config: The Config used for this run.
        results: The pipeline's final results DataFrame (one row per feature).
        experiment_name: MLflow experiment name to log the run under.

    Raises:
        ImportError: If the optional 'mlflow' package is not installed.
    """
    try:
        import mlflow
    except ImportError as err:
        raise ImportError(
            "The 'mlflow' package is required when use_mlflow=True."
            " Install it with: pip install mlflow"
        ) from err

    import tempfile
    from pathlib import Path

    # Only log flat, scalar Config fields as params -- nested dataclass
    # fields (optuna_config, xgboost_config, leakage_config, etc.) aren't
    # MLflow-param-friendly (not strings/numbers/bools) and are skipped here
    # rather than flattened, keeping this additive and simple.
    params = {
        f.name: getattr(config, f.name)
        for f in dataclasses.fields(config)
        if not dataclasses.is_dataclass(getattr(config, f.name))
    }

    numeric_columns = [
        col
        for col in results.columns
        if pd.api.types.is_numeric_dtype(results[col]) and col != "feature_name"
    ]

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(params)

        if "feature_name" in results.columns:
            for col in numeric_columns:
                for feature_name, value in zip(
                    results["feature_name"], results[col], strict=True
                ):
                    mlflow.log_metric(f"{col}__{feature_name}", float(value))

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.csv"
            results.to_csv(results_path, index=False)
            mlflow.log_artifact(str(results_path))


def derive_prefix_groups(feature_names: list[str]) -> dict[str, list[str]]:
    """Auto-derive feature groups from a common prefix before the last underscore.

    E.g. "price_min" and "price_max" both map to group "price". Features with
    no underscore in their name are skipped (they don't belong to any group).
    Groups with only a single member are dropped, since a "group" of one
    feature provides no aggregation benefit over the per-feature row.

    Args:
        feature_names: List of feature column names.

    Returns:
        Dict mapping group name to list of feature names, excluding
        singleton groups.
    """
    candidates: dict[str, list[str]] = defaultdict(list)
    for name in feature_names:
        if "_" not in name:
            continue
        prefix = name.rsplit("_", 1)[0]
        candidates[prefix].append(name)
    return {prefix: names for prefix, names in candidates.items() if len(names) > 1}


def build_group_summary(
    feature_groups: dict[str, list[str]],
    feature_names: list[str],
    shap_importance: Any,
) -> pd.DataFrame:
    """Aggregate per-feature SHAP importance into a group-level summary.

    Importance is summed (not averaged) per group, since the sum reads as
    "total model attention this group accounts for" -- a more interpretable
    quantity for deciding whether to drop an entire group than a per-feature
    average would be.

    Features named in a group but not present in feature_names are silently
    ignored, so users can pass slightly stale or over-inclusive group specs.

    Args:
        feature_groups: Dict mapping group name to list of feature names.
        feature_names: All feature names evaluated by the pipeline, in the
            same order as shap_importance.
        shap_importance: Mean absolute SHAP value per feature, aligned with
            feature_names.

    Returns:
        DataFrame with columns: group_name, n_features, total_shap_importance,
        features. Sorted by total_shap_importance descending.
    """
    importance_by_feature = dict(zip(feature_names, shap_importance, strict=True))

    rows = []
    for group_name, members in feature_groups.items():
        present = [f for f in members if f in importance_by_feature]
        total = sum(importance_by_feature[f] for f in present)
        rows.append(
            {
                "group_name": group_name,
                "n_features": len(present),
                "total_shap_importance": total,
                "features": ", ".join(present),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("total_shap_importance", ascending=False)
        .reset_index(drop=True)
    )


class FeatureSelectionPipeline:
    """
    Orchestrates the full TreeShaker feature selection workflow.

    The pipeline runs in the following order:
    1. Load CSV data and split into train/validation/test sets.
    2. Tune XGBoost hyperparameters with Optuna (skipped when
       config.skip_tuning is True, in which case the configured
       XGBoostConfig values are used directly).
    3. Train the final XGBoost model on tuned hyperparameters.
    4. Compute SHAP importance for each feature.
    5. Compute permutation importance for each feature.
    6. Compute mutual information between each feature and the target.
    7. Flag harmful features using both importance signals.
    8. Detect data leakage warnings.
    9. (Optional) Refit on bootstrap resamples to assess importance
       ranking stability, when config.bootstrap_config.n_bootstrap > 0.
    10. Save results to CSV or HTML.
    11. (Optional, opt-in) Compute pairwise SHAP interaction values and save
        them to a separate `<output_stem>_interactions.csv` file.

    Args:
        config: A Config instance describing all pipeline settings.

    Example:
        >>> from treeshaker import FeatureSelectionPipeline, Config
        >>> config = Config(
        ...     input_file="data.csv", target_column="price", task="regression"
        ... )
        >>> pipeline = FeatureSelectionPipeline(config)
        >>> results = pipeline.run()
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.train_data: pd.DataFrame | None = None
        self.val_data: pd.DataFrame | None = None
        self.test_data: pd.DataFrame | None = None
        self.model: Any = None
        self.feature_names: list[str] = []
        self.results: pd.DataFrame | None = None
        self.group_summary: pd.DataFrame | None = None
        self.interactions: pd.DataFrame | None = None
        self.severe_class_imbalance: bool = False
        self.suggested_features: list[SyntheticFeatureSuggestion] | None = None

    def _train_val_data(self) -> pd.DataFrame:
        """Concatenate train and validation splits for final-model fitting.

        Used by every step that should see all non-test data (final model
        training, bootstrap stability checks, and the opt-in alternative
        importance backends) without ever touching self.test_data.
        """
        if self.train_data is None or self.val_data is None:
            raise RuntimeError("load_data() must be called first")
        return pd.concat([self.train_data, self.val_data], ignore_index=True)

    def load_data(self) -> None:
        """
        Load the CSV file and split into train, validation, and test sets.

        Populates self.train_data, self.val_data, self.test_data, and
        self.feature_names.
        """
        self.train_data, self.val_data, self.test_data = load_csv(
            filepath=self.config.input_file,
            target_column=self.config.target_column,
            task=self.config.task,
            test_size=self.config.test_size,
            val_size=self.config.val_size,
            random_seed=self.config.random_seed,
            time_column=self.config.time_column,
            large_dataset_sample_threshold=(self.config.large_dataset_sample_threshold),
            large_dataset_sample_frac=self.config.large_dataset_sample_frac,
        )
        feature_names = [
            col for col in self.train_data.columns if col != self.config.target_column
        ]

        # Validate and normalize feature dtypes (datetime expansion,
        # categorical casting, free-text rejection) once, consistently
        # across all three splits, before anything is fit. See
        # dtype_handling.py for the full rationale per dtype category --
        # in particular why this must be done jointly across train/val/test
        # rather than independently per split.
        (
            self.train_data,
            self.val_data,
            self.test_data,
            self.feature_names,
            _,
        ) = prepare_feature_dtypes_splits(
            self.train_data, self.val_data, self.test_data, feature_names
        )

        self.logger.info(f"Features to evaluate: {len(self.feature_names)}")

        combined = pd.concat(
            [self.train_data, self.val_data, self.test_data], ignore_index=True
        )
        imbalance_warning = check_class_imbalance(
            df=combined,
            target_column=self.config.target_column,
            task=self.config.task,
            imbalance_warning_threshold=self.config.imbalance_warning_threshold,
        )
        self.severe_class_imbalance = imbalance_warning is not None
        if imbalance_warning:
            self.logger.warning(imbalance_warning)

    def tune_hyperparams(self) -> dict:
        """
        Use Optuna to find the best XGBoost hyperparameters.

        Runs an Optuna study where each trial trains XGBoost on the training
        set and evaluates on the validation set. The metric depends on the task:
        - binary: log loss (minimized)
        - multiclass: log loss (minimized)
        - regression: MSE (minimized)

        Returns:
            Dictionary of best XGBoost hyperparameters found by Optuna.
        """
        import optuna
        import xgboost as xgb
        from sklearn.metrics import log_loss, mean_squared_error

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if self.train_data is None or self.val_data is None:
            raise RuntimeError("load_data() must be called before tune_hyperparams()")

        cfg = self.config.optuna_config
        task = self.config.task

        X_train = self.train_data[self.feature_names]
        y_train = self.train_data[self.config.target_column]
        X_val = self.val_data[self.feature_names]
        y_val = self.val_data[self.config.target_column]

        class _OptunaPruningCallback(xgb.callback.TrainingCallback):
            def __init__(self, trial: optuna.Trial) -> None:
                self.trial = trial

            def after_iteration(self, model: Any, epoch: int, evals_log: dict) -> bool:
                if not evals_log:
                    return False
                last_dataset = list(evals_log.values())[-1]
                if not last_dataset:
                    return False
                last_metric_logs = list(last_dataset.values())[-1]
                if last_metric_logs:
                    self.trial.report(float(last_metric_logs[-1]), step=epoch)
                    if self.trial.should_prune():
                        raise optuna.TrialPruned()
                return False

        def objective(trial: optuna.Trial) -> float:
            tuned_params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }
            params = _build_xgb_fit_params(tuned_params, self.config)
            if self.config.use_optuna_pruning:
                params["callbacks"] = [_OptunaPruningCallback(trial)]
                params["early_stopping_rounds"] = 10
            model = _make_xgb_model(task, params)

            fit_kwargs: dict[str, Any] = {
                "eval_set": [(X_val, y_val)],
                "verbose": False,
            }

            if task in ("binary", "multiclass"):
                model.fit(X_train, y_train, **fit_kwargs)
                # Binary and multiclass are both scored by log loss on the
                # predicted class probabilities.
                preds = model.predict_proba(X_val)
                return log_loss(y_val, preds)
            else:
                model.fit(X_train, y_train, **fit_kwargs)
                preds = model.predict(X_val)
                return mean_squared_error(y_val, preds)

        sampler = optuna.samplers.TPESampler(seed=cfg.seed)
        pruner: optuna.pruners.BasePruner = (
            optuna.pruners.MedianPruner()
            if self.config.use_optuna_pruning
            else optuna.pruners.NopPruner()
        )
        study = optuna.create_study(
            direction="minimize", sampler=sampler, pruner=pruner
        )
        self.logger.info(
            f"Starting Optuna tuning: {cfg.n_trials} trials, timeout={cfg.timeout}s"
        )
        study.optimize(
            objective,
            n_trials=cfg.n_trials,
            timeout=cfg.timeout,
            show_progress_bar=cfg.show_progress_bar,
        )
        self.logger.info(f"Best trial value: {study.best_value:.4f}")
        return study.best_params

    def train_model(self, best_params: dict) -> None:
        """
        Train the final XGBoost model using the best hyperparameters.

        Combines train and validation data for the final fit, giving the
        model access to as much data as possible before importance analysis.

        Args:
            best_params: Hyperparameter dict returned by tune_hyperparams().
        """
        combined = self._train_val_data()
        X = combined[self.feature_names]
        y = combined[self.config.target_column]

        params = _build_xgb_fit_params(best_params, self.config)
        self.model = _make_xgb_model(self.config.task, params)

        self.model.fit(X, y)
        self.logger.info("Final model trained on combined train+val data")

    def _run_bootstrap_importance(self, best_params: dict) -> np.ndarray:
        """
        Refit models on bootstrap resamples and collect SHAP importance.

        Uses the same best_params already found by Optuna for every
        iteration (re-running Optuna per bootstrap sample would be far too
        slow and isn't the point of this feature). Only SHAP importance is
        recomputed per iteration; permutation importance is skipped here
        because it requires an extra full prediction pass per feature per
        iteration, which would multiply the cost of this already n_bootstrap-
        times-more-expensive step. SHAP alone is sufficient to assess
        ranking stability.

        Args:
            best_params: Hyperparameter dict returned by tune_hyperparams().

        Returns:
            Array of shape (n_bootstrap, n_features) with SHAP importance
            from each bootstrap iteration.
        """
        from .importance import compute_shap_importance

        if self.train_data is None or self.val_data is None or self.test_data is None:
            raise RuntimeError(
                "load_data() must be called before _run_bootstrap_importance()"
            )

        bootstrap_cfg = self.config.bootstrap_config
        combined = self._train_val_data()
        n_rows = len(combined)
        sample_size = max(1, int(round(n_rows * bootstrap_cfg.sample_frac)))

        results = np.zeros((bootstrap_cfg.n_bootstrap, len(self.feature_names)))
        for i in range(bootstrap_cfg.n_bootstrap):
            seed = self.config.random_seed + i
            rng = np.random.default_rng(seed=seed)
            sample_idx = rng.integers(0, n_rows, size=sample_size)
            resampled = combined.iloc[sample_idx].reset_index(drop=True)

            X_resampled = resampled[self.feature_names]
            y_resampled = resampled[self.config.target_column]

            iter_params = _build_xgb_fit_params(
                best_params, self.config, random_seed=seed
            )
            bootstrap_model = _make_xgb_model(self.config.task, iter_params)

            bootstrap_model.fit(X_resampled, y_resampled)

            results[i] = compute_shap_importance(
                model=bootstrap_model,
                data=self.test_data,
                feature_names=self.feature_names,
                sample_size=self.config.shap_sample_size,
                random_seed=self.config.random_seed,
            )
            self.logger.info(
                f"Bootstrap importance iteration {i + 1}/{bootstrap_cfg.n_bootstrap}"
                " complete"
            )

        return results

    def _apply_stability_gating(
        self,
        bootstrap_importances: np.ndarray,
        recommendations: list[str],
        reasons: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Apply stability gating: demote recommendations if SHAP importance is unstable.

        When bootstrap is enabled and shap_importance_stability_threshold is set,
        compute per-feature coefficient of variation (CV = std / mean) of SHAP
        importance across bootstrap refits. If CV exceeds threshold, demote:
        Drop→Caution, Keep→Caution. Caution stays Caution. Adds a note to reason.

        Args:
            bootstrap_importances: Array of shape (n_bootstrap, n_features) with
                bootstrap SHAP importance values from _run_bootstrap_importance().
            recommendations: Current list of recommendations (Drop/Caution/Keep).
            reasons: Current list of reason strings.

        Returns:
            Tuple of (updated_recommendations, updated_reasons) with stability
            gating applied.
        """
        stability_threshold = (
            self.config.bootstrap_config.shap_importance_stability_threshold
        )
        if stability_threshold is None:
            # Stability gating disabled
            return recommendations, reasons

        means = bootstrap_importances.mean(axis=0)
        stds = bootstrap_importances.std(axis=0)

        updated_recommendations = list(recommendations)
        updated_reasons = list(reasons)

        for i, _feature in enumerate(self.feature_names):
            # Avoid division by zero: if mean SHAP is zero, CV is undefined.
            # In that case, the feature is already demoted by harm detection,
            # so we skip stability gating (no need to demote further).
            if means[i] == 0:
                continue

            cv = stds[i] / means[i]

            if cv > stability_threshold:
                # Feature ranking is unstable: demote if not already Caution.
                if updated_recommendations[i] == "Drop":
                    updated_recommendations[i] = "Caution"
                    note = (
                        f"SHAP importance ranking is unstable across bootstrap "
                        f"refits (coefficient of variation {cv:.3f} exceeds "
                        f"threshold {stability_threshold:.3f}); demoted from Drop "
                        f"to Caution pending further investigation."
                    )
                    updated_reasons[i] = (
                        f"{updated_reasons[i]} {note}".strip()
                        if updated_reasons[i]
                        else note
                    )
                elif updated_recommendations[i] == "Keep":
                    updated_recommendations[i] = "Caution"
                    note = (
                        f"SHAP importance ranking is unstable across bootstrap "
                        f"refits (coefficient of variation {cv:.3f} exceeds "
                        f"threshold {stability_threshold:.3f}); demoted from Keep "
                        f"to Caution pending further investigation."
                    )
                    updated_reasons[i] = note

        return updated_recommendations, updated_reasons

    def run(self) -> pd.DataFrame:
        """
        Execute the full feature selection pipeline.

        Returns:
            DataFrame with columns: feature_name, shap_importance,
            permutation_importance, permutation_importance_std,
            mutual_info_score, recommendation, reason.
            permutation_importance is the mean score drop across
            config.n_permutations shuffles per feature; permutation_importance_std
            is the standard deviation across those shuffles (a stability signal).
            recommendation is "Drop", "Caution", or "Keep".
            reason is a plain English explanation (empty string if "Keep").
            Sorted by shap_importance descending.

            This per-feature schema is unchanged regardless of grouping
            settings. If feature_groups is set, or group_by_prefix is True
            and at least one prefix group is found, a group-level summary
            is additionally written to "<output_stem>_groups.csv" next to
            the main output file (see build_group_summary()).

            If config.bootstrap_config.n_bootstrap > 0, two additional
            columns are included: shap_importance_mean and
            shap_importance_std, summarizing SHAP importance across
            bootstrap-resampled refits for ranking stability assessment.

            If config.compute_shap_interactions is True, pairwise SHAP
            interaction values are also computed and saved to
            `<output_stem>_interactions.csv`, and made available on
            self.interactions. This does not change the shape or schema
            of the returned DataFrame.

            If config.check_residuals is True, a residual_correlation
            column is added (Pearson correlation between each feature's raw
            values and the model's prediction residuals on the held-out
            test set; see residual_diagnostics.py for the per-task
            definition of "residual"). When a feature's residual
            correlation exceeds config.residual_config.residual_corr_threshold:
            if it was flagged "Drop", the recommendation is downgraded to
            "Caution" and a plain-English note is appended to its reason;
            otherwise the note is appended informationally without changing
            the recommendation tier.

            If config.compute_shap_interactions is True and
            config.synthetic_config.synthesize_from_interactions is True,
            candidate pairwise-product features are derived from the
            strongest SHAP interactions and scored by mutual information
            against the target (see synthetic_features.py). Candidates
            exceeding config.synthetic_config.mi_threshold are saved to
            `<output_stem>_suggested_features.csv` and made available on
            self.suggested_features. This is advisory only -- suggestions
            are never added to the model or training data. If
            synthesize_from_interactions is True but
            compute_shap_interactions is False, a warning is logged and
            this step is skipped (self.suggested_features stays None).
        """
        self.load_data()
        if self.train_data is None or self.val_data is None or self.test_data is None:
            raise RuntimeError("load_data() did not populate train/val/test data")

        if self.config.skip_tuning:
            self.logger.info(
                "Skipping Optuna tuning (skip_tuning=True); using configured"
                " XGBoost parameters directly."
            )
            best_params = self.config.xgboost_config.to_dict()
        else:
            best_params = self.tune_hyperparams()
        self.train_model(best_params)

        metric_fn = get_metric_fn(
            self.config.task, override=self.config.permutation_metric
        )

        shap_importance = compute_shap_importance(
            model=self.model,
            data=self.test_data,
            feature_names=self.feature_names,
            sample_size=self.config.shap_sample_size,
            random_seed=self.config.random_seed,
        )

        # Optionally compute SHAP importance on train+val for overfit/leakage
        # diagnostics (does not change the primary test-based evaluation).
        shap_importance_train = None
        shap_importance_delta = None
        if self.config.compute_train_importance:
            if self.train_data is None or self.val_data is None:
                raise RuntimeError(
                    "load_data() must be called before compute_train_importance"
                )
            train_val_data = self._train_val_data()
            shap_importance_train = compute_shap_importance(
                model=self.model,
                data=train_val_data,
                feature_names=self.feature_names,
                sample_size=self.config.shap_sample_size,
                random_seed=self.config.random_seed,
            )
            shap_importance_delta = compute_shap_importance_delta(
                train_importance=shap_importance_train,
                test_importance=shap_importance,
            )
            self.logger.info(
                "Train-vs-test SHAP importance delta computed (diagnostic only)"
            )

        perm_importance, perm_importance_std = compute_permutation_importance(
            model=self.model,
            data=self.test_data,
            target_column=self.config.target_column,
            feature_names=self.feature_names,
            metric_fn=metric_fn,
            use_predict_proba=(self.config.task == "binary"),
            n_repeats=self.config.n_permutations,
        )

        mi_score = compute_mutual_information(
            data=self.test_data,
            target_column=self.config.target_column,
            feature_names=self.feature_names,
            task=self.config.task,
            random_seed=self.config.random_seed,
        )

        detector = HarmDetector(
            shap_importance_threshold=self.config.shap_importance_threshold,
            perm_importance_threshold=self.config.permutation_importance_threshold,
        )
        if self.severe_class_imbalance:
            self.logger.warning(
                "Severe class imbalance detected -- permutation importance"
                " threshold is being skipped for harm detection; relying on"
                " SHAP importance alone for this run."
            )
        is_harmful, harm_reasons = detector.flag_harmful_features(
            shap_importance=shap_importance,
            perm_importance=perm_importance,
            severe_class_imbalance=self.severe_class_imbalance,
        )

        # Leakage detection looks at statistical properties (e.g. target
        # correlation) across the whole dataset, not held-out generalization
        # -- unlike importance evaluation above, using all three splits here
        # is correct and intentional.
        leakage_detector = LeakageDetector(
            self.config.leakage_config, screening_mode=self.config.leakage_screening
        )
        combined_data = pd.concat(
            [self.train_data, self.val_data, self.test_data], ignore_index=True
        )
        leakage_warnings = leakage_detector.detect(
            df=combined_data,
            target_column=self.config.target_column,
            feature_names=self.feature_names,
            shap_importance=shap_importance,
            task=self.config.task,
        )
        temporal_structure_warnings = leakage_detector.detect_temporal_structure(
            df=combined_data,
            feature_names=self.feature_names,
        )

        # Check for look-ahead leakage if chronological split is enabled
        lookahead_warnings: dict[str, list[str]] = {
            name: [] for name in self.feature_names
        }
        if self.config.time_column is not None:
            lookahead_warnings = leakage_detector.detect_lookahead_leakage(
                train_df=self.train_data,
                val_df=self.val_data,
                test_df=self.test_data,
                time_column=self.config.time_column,
                feature_names=self.feature_names,
            )

        residual_correlation = None
        residual_notes: dict[str, str] = {}
        if self.config.check_residuals:
            residuals = compute_residuals(
                model=self.model,
                data=self.test_data,
                target_column=self.config.target_column,
                feature_names=self.feature_names,
                task=self.config.task,
            )
            residual_correlation = compute_residual_correlation(
                data=self.test_data,
                feature_names=self.feature_names,
                residuals=residuals,
            )
            residual_notes = flag_residual_correlated_features(
                feature_names=self.feature_names,
                residual_correlation=residual_correlation,
                config=self.config.residual_config,
            )

        recommendations = []
        reasons = []
        for i, feature in enumerate(self.feature_names):
            lk_warnings = leakage_warnings[feature]
            residual_note = residual_notes.get(feature, "")
            temporal_note = " ".join(temporal_structure_warnings.get(feature, []))
            lookahead_note = " ".join(lookahead_warnings.get(feature, []))

            lookahead_list = [lookahead_note] if lookahead_note else []
            combined_warnings = [*lk_warnings, *lookahead_list]

            if is_harmful[i] and combined_warnings:
                recommendation = "Drop"
                all_reasons = [harm_reasons[i], *combined_warnings]
                numbered = " ".join(
                    f"({j}) {r}" for j, r in enumerate(all_reasons, start=1)
                )
                reason = f"{len(all_reasons)} issues found — " + numbered
            elif is_harmful[i]:
                recommendation = "Drop"
                reason = harm_reasons[i]
            elif combined_warnings:
                recommendation = "Caution"
                if len(combined_warnings) > 1:
                    numbered = " ".join(
                        f"({j}) {r}" for j, r in enumerate(combined_warnings, start=1)
                    )
                    reason = f"{len(combined_warnings)} issues found — " + numbered
                else:
                    reason = combined_warnings[0]
            else:
                recommendation = "Keep"
                reason = ""

            if recommendation == "Drop" and residual_note:
                # Residual evidence contradicts the Drop call (the feature
                # correlates with where the model is systematically wrong,
                # even though SHAP/permutation/MI rank it low). Downgrading
                # to Caution is more honest than keeping a "Drop" label that
                # the residual signal disputes -- it avoids over-trusting a
                # recommendation that has direct evidence against it, while
                # still telling the user the original harm/leakage reasoning.
                recommendation = "Caution"
                reason = f"{reason} {residual_note}".strip()
            elif residual_note:
                # Not a Drop call -- residual correlation is purely
                # informational here, surfaced via reason without changing
                # the recommendation tier.
                reason = (
                    f"{reason} {residual_note}".strip() if reason else residual_note
                )

            if temporal_note:
                # Temporal structure issues (e.g. a lag/rolling feature
                # whose leading-NaN count doesn't match its name) are a
                # distinct category from leakage -- a feature-engineering
                # correctness concern, not evidence of target derivation.
                # Surfaced via reason without changing the recommendation
                # tier, kept separate from leakage/residual reasoning.
                reason = (
                    f"{reason} {temporal_note}".strip() if reason else temporal_note
                )

            if (
                self.config.flag_leakage_importance_mismatch
                and not is_harmful[i]
                and lk_warnings
            ):
                # Dangerous case: feature passes the importance/harm check
                # (the model finds it useful -- not is_harmful[i]) but also
                # has active leakage warnings. Note this is exactly the
                # condition that produces the "Caution" tier above: leakage
                # warnings alone, without harm, always route to Caution
                # rather than Keep, so this can't be gated on the final
                # reconciled `recommendation == "Keep"` -- that branch is
                # unreachable whenever lk_warnings is truthy. Gating on the
                # underlying importance signal (not is_harmful[i]) instead
                # correctly captures "the model is using this feature
                # meaningfully, yet it's also statistically suspicious,"
                # which could indicate undetected leakage. Surfaced via
                # reason without changing the recommendation, since the
                # user explicitly enabled this flag.
                mismatch_note = (
                    "Feature is flagged with potential leakage concerns "
                    "despite high importance — review whether this indicates "
                    "undetected leakage."
                )
                reason = (
                    f"{reason} {mismatch_note}".strip() if reason else mismatch_note
                )

            recommendations.append(recommendation)
            reasons.append(reason)

        result_columns: dict[str, Any] = {
            "feature_name": self.feature_names,
            "shap_importance": shap_importance,
            "permutation_importance": perm_importance,
            "permutation_importance_std": perm_importance_std,
            "mutual_info_score": mi_score,
            "recommendation": recommendations,
            "reason": reasons,
        }

        bootstrap_importances = None
        if self.config.bootstrap_config.n_bootstrap > 0:
            bootstrap_importances = self._run_bootstrap_importance(best_params)
            result_columns["shap_importance_mean"] = bootstrap_importances.mean(axis=0)
            result_columns["shap_importance_std"] = bootstrap_importances.std(axis=0)
            # Apply stability gating if threshold is configured
            if (
                self.config.bootstrap_config.shap_importance_stability_threshold
                is not None
            ):
                recommendations, reasons = self._apply_stability_gating(
                    bootstrap_importances, recommendations, reasons
                )
                result_columns["recommendation"] = recommendations
                result_columns["reason"] = reasons

        if self.config.check_residuals:
            result_columns["residual_correlation"] = residual_correlation

        if self.config.compute_train_importance and shap_importance_delta is not None:
            result_columns["train_test_importance_delta"] = shap_importance_delta

        if self.config.compute_importance_divergence:
            result_columns["importance_divergence"] = compute_importance_divergence(
                shap_importance=shap_importance,
                perm_importance=perm_importance,
                mi_score=mi_score,
            )

        if self.config.use_boruta:
            result_columns["boruta_decision"] = run_boruta_shap(
                model=self.model,
                data=self.test_data,
                target_column=self.config.target_column,
                feature_names=self.feature_names,
                task=self.config.task,
                random_seed=self.config.random_seed,
            )

        if self.config.use_rfecv:
            train_val_data = self._train_val_data()
            result_columns["rfecv_ranking"] = run_rfecv(
                model=self.model,
                data=train_val_data,
                target_column=self.config.target_column,
                feature_names=self.feature_names,
                task=self.config.task,
                random_seed=self.config.random_seed,
            )

        if self.config.use_lightgbm:
            train_val_data = self._train_val_data()
            result_columns["lightgbm_importance"] = run_lightgbm_importance(
                data=train_val_data,
                target_column=self.config.target_column,
                feature_names=self.feature_names,
                task=self.config.task,
                random_seed=self.config.random_seed,
            )

        if self.config.use_catboost:
            result_columns["catboost_importance"] = run_catboost_importance(
                data=self.test_data,
                target_column=self.config.target_column,
                feature_names=self.feature_names,
                task=self.config.task,
                random_seed=self.config.random_seed,
            )

        if self.config.use_rf_baseline:
            train_val_data = self._train_val_data()
            result_columns["rf_importance"] = run_rf_baseline_importance(
                data=train_val_data,
                target_column=self.config.target_column,
                feature_names=self.feature_names,
                task=self.config.task,
                random_seed=self.config.random_seed,
            )

        if self.config.use_sklearn_permutation_importance:
            result_columns["sklearn_permutation_importance"] = (
                run_sklearn_permutation_importance(
                    model=self.model,
                    data=self.test_data,
                    target_column=self.config.target_column,
                    feature_names=self.feature_names,
                    task=self.config.task,
                    n_permutations=self.config.n_permutations,
                    random_seed=self.config.random_seed,
                )
            )

        self.results = (
            pd.DataFrame(result_columns)
            .sort_values("shap_importance", ascending=False)
            .reset_index(drop=True)
        )

        save_results(
            results=self.results,
            output_file=self.config.output_file,
            output_format=self.config.output_format,
        )

        groups = self.config.feature_groups
        if not groups and self.config.group_by_prefix:
            groups = derive_prefix_groups(self.feature_names)

        if groups:
            group_summary = build_group_summary(
                feature_groups=groups,
                feature_names=self.feature_names,
                shap_importance=shap_importance,
            )
            self.group_summary = group_summary
            stem = derive_output_stem(self.config.output_file)
            groups_path = f"{stem}_groups.csv"
            group_summary.to_csv(groups_path, index=False)
            self.logger.info(
                f"Group summary ({len(group_summary)} groups) saved to {groups_path}"
            )

        if self.config.compute_shap_interactions:
            self.interactions = compute_shap_interactions(
                model=self.model,
                data=self.test_data,
                feature_names=self.feature_names,
            )
            stem = derive_output_stem(self.config.output_file)
            interactions_file = f"{stem}_interactions.csv"
            self.interactions.to_csv(interactions_file, index=False)
            self.logger.info(f"SHAP interaction values saved to {interactions_file}")

            if self.config.synthetic_config.synthesize_from_interactions:
                self.suggested_features = suggest_synthetic_features(
                    data=self.test_data,
                    target_column=self.config.target_column,
                    interactions=self.interactions,
                    task=self.config.task,
                    random_seed=self.config.random_seed,
                    config=self.config.synthetic_config,
                )
                stem = derive_output_stem(self.config.output_file)
                suggestions_file = f"{stem}_suggested_features.csv"
                pd.DataFrame([vars(s) for s in self.suggested_features]).to_csv(
                    suggestions_file, index=False
                )
                self.logger.info(
                    f"{len(self.suggested_features)} synthetic feature"
                    f" suggestions saved to {suggestions_file}"
                )
        elif self.config.synthetic_config.synthesize_from_interactions:
            self.logger.warning(
                "synthetic_config.synthesize_from_interactions is True but"
                " compute_shap_interactions is False; skipping synthetic"
                " feature suggestion. Set compute_shap_interactions=True to"
                " enable it."
            )

        if self.config.use_mlflow:
            log_run_to_mlflow(
                config=self.config,
                results=self.results,
                experiment_name=self.config.mlflow_experiment_name,
            )
            self.logger.info(
                "Run logged to MLflow experiment"
                f" '{self.config.mlflow_experiment_name}'."
            )

        n_drop = (self.results["recommendation"] == "Drop").sum()
        n_caution = (self.results["recommendation"] == "Caution").sum()
        self.logger.info(
            f"Done. {len(self.results)} features evaluated:"
            f" {n_drop} to drop, {n_caution} flagged as caution."
        )
        return self.results

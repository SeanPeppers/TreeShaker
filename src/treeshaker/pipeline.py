"""Core feature selection pipeline orchestration."""

import dataclasses
import logging
from collections import defaultdict
from typing import Any, Literal, cast

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
from .synthetic_features import (
    SyntheticFeatureSuggestion,
    build_injection_frame,
    fit_interaction_encoders,
    suggest_synthetic_features,
)
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


def _validate_custom_model(model: Any) -> None:
    """Raise ValueError if `model` is missing required sklearn-API methods.

    Args:
        model: The Config.custom_model object to validate.

    Raises:
        ValueError: Listing any of "fit"/"predict" not present as callables
            on the object.
    """
    missing = [
        name for name in ("fit", "predict") if not callable(getattr(model, name, None))
    ]
    if missing:
        raise ValueError(
            "custom_model is missing required method(s): "
            f"{', '.join(missing)}. It must implement .fit(X, y) and"
            " .predict(X) (classifiers also need .predict_proba(X))."
        )


def _clone_custom_model(model: Any) -> Any:
    """Return an unfitted clone of a user-supplied custom_model.

    Tries `sklearn.base.clone` first (correct for sklearn-API estimators,
    including most wrapper adapters like skorch/scikeras); falls back to
    `copy.deepcopy` for objects clone() rejects (e.g. it doesn't implement
    get_params/set_params).

    Args:
        model: The Config.custom_model object to clone.

    Returns:
        A new, unfitted estimator instance.
    """
    from sklearn.base import clone

    try:
        return clone(model)
    except TypeError:
        import copy

        return copy.deepcopy(model)


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
    from scipy.stats import binomtest, false_discovery_control

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

    # Testing every feature against the same alpha independently inflates
    # false Confirmed/Rejected calls as feature count grows -- correct across
    # all of this run's per-feature p-values together (Benjamini-Hochberg)
    # before thresholding, rather than each feature's raw p-value alone.
    raw_pvalues = [binomtest(int(h), n_iterations, 0.5).pvalue for h in hits]
    corrected_pvalues = false_discovery_control(raw_pvalues)

    decisions = []
    for h, pvalue in zip(hits, corrected_pvalues, strict=True):
        hit_rate = h / n_iterations
        if pvalue < 0.05 and hit_rate > 0.5:
            decisions.append("Confirmed")
        elif pvalue < 0.05 and hit_rate < 0.5:
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


def _normalize_importances(importances: Any) -> list[float]:
    """Normalize raw feature importances to sum to 1.0.

    Shared by the alternative importance backends (LightGBM, CatBoost, RF
    baseline), all of which normalize their model's raw
    feature_importances_ the same way. Falls back to the raw values
    unnormalized if they sum to 0 (e.g. a model that assigned zero
    importance to every feature).
    """
    total = float(sum(importances))
    if total > 0:
        return [float(i / total) for i in importances]
    return [float(i) for i in importances]


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
    return _normalize_importances(importances)


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
    return _normalize_importances(importances)


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
    return _normalize_importances(importances)


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
        self._best_params: dict = {}
        # Set in run(), after load_data() determines severe_class_imbalance:
        # equal to config.n_shadow_probes/shadow_probe_percentile unless
        # severe imbalance is detected and the user didn't already opt into
        # shadow probes, in which case these are auto-set (see run()) so the
        # noise-floor cross-check activates for a population where the
        # fixed importance thresholds are already known to be unreliable.
        self._effective_n_shadow_probes: int = 0
        self._effective_shadow_probe_percentile: float = 100.0

    def _train_val_data(self) -> pd.DataFrame:
        """Concatenate train and validation splits for final-model fitting.

        Used by every step that should see all non-test data (final model
        training, bootstrap stability checks, and the opt-in alternative
        importance backends) without ever touching self.test_data.
        """
        if self.train_data is None or self.val_data is None:
            raise RuntimeError("load_data() must be called first")
        return pd.concat([self.train_data, self.val_data], ignore_index=True)

    @property
    def X_train(self) -> pd.DataFrame | None:
        """Return training features DataFrame."""
        if self.train_data is None:
            return None
        return self.train_data[self.feature_names]

    def _eval_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bound a frame to streaming_eval_sample_size rows when streaming.

        The explainability/diagnostic steps (SHAP, permutation importance,
        mutual information, leakage, residuals) conventionally evaluate on a
        held-out sample rather than every row, so under out-of-core streaming
        we cap them to a bounded in-memory sample -- the standard practical
        answer when the full split is too large to materialize comfortably.

        Returns df unchanged when streaming is off OR when df already fits
        within the cap, so the non-streaming path is byte-for-byte identical
        to before. The sample is drawn deterministically with random_seed so
        results are reproducible.
        """
        if not self.config.streaming:
            return df
        cap = self.config.streaming_eval_sample_size
        if len(df) <= cap:
            return df
        self.logger.info(
            "streaming: sampling %d of %d rows for explainability/diagnostics",
            cap,
            len(df),
        )
        return df.sample(n=cap, random_state=self.config.random_seed).reset_index(
            drop=True
        )

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
            streaming=self.config.streaming,
            streaming_chunk_size=self.config.streaming_chunk_size,
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
        Train the final model using the best hyperparameters.

        Combines train and validation data for the final fit, giving the
        model access to as much data as possible before importance analysis.

        If config.custom_model is set, fits a clone of that estimator
        instead of building an XGBClassifier/XGBRegressor; best_params is
        ignored in that case since it is XGBoost-specific (tuning is also
        skipped upstream for custom_model -- see run()).

        Args:
            best_params: Hyperparameter dict returned by tune_hyperparams().
                Ignored when config.custom_model is set.
        """
        self._best_params = best_params
        combined = self._train_val_data()
        X = combined[self.feature_names]
        y = combined[self.config.target_column]

        if self.config.custom_model is not None:
            _validate_custom_model(self.config.custom_model)
            self.model = _clone_custom_model(self.config.custom_model)
            self.model.fit(X, y)
            self.logger.info("Final custom_model trained on combined train+val data")
            return

        params = _build_xgb_fit_params(best_params, self.config)

        if self.config.streaming:
            # Out-of-core training: build the booster from on-disk chunks via
            # XGBoost's external-memory iterator rather than fitting the
            # sklearn wrapper on the full in-memory matrix. The returned
            # BoosterAdapter exposes the same predict/predict_proba surface the
            # downstream steps use. See streaming_train.py for the rationale.
            from .streaming_train import train_booster_streaming

            self.model = train_booster_streaming(
                X=X,
                y=y,
                task=self.config.task,
                params=params,
                feature_names=self.feature_names,
                chunk_size=self.config.streaming_chunk_size,
            )
            self.logger.info(
                "Final model trained out-of-core (streaming) over on-disk chunks"
            )
            return

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

    def _apply_collinearity_gating(
        self,
        recommendations: list[str],
        reasons: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Downgrade Drop to Caution when a low-importance feature is collinear
        with a feature the pipeline is keeping.

        Permutation importance is known to be misleading under high
        collinearity: shuffling one of two correlated-but-useful features
        barely hurts the model, since its partner still carries the signal --
        so both can end up with deflated importance and get flagged Drop,
        even though the redundancy (not uselessness) is what's driving the
        low score. When collinearity clustering is enabled and a Drop
        feature shares a cluster with a Keep feature, that's a real
        possibility, not evidence the Drop feature carries no signal on its
        own -- downgrade to Caution rather than Drop.

        Opt-in via collinearity_threshold (0 = disabled, the default); a
        no-op when disabled, matching cluster_correlated_features' own
        no-op-to-singletons behavior at threshold <= 0.

        Args:
            recommendations: Current list of recommendations (Drop/Caution/Keep).
            reasons: Current list of reason strings.

        Returns:
            Tuple of (updated_recommendations, updated_reasons).
        """
        if self.config.collinearity_threshold <= 0:
            return recommendations, reasons

        from .collinearity import cluster_correlated_features

        clusters = cluster_correlated_features(
            self.X_train if self.X_train is not None else pd.DataFrame(),
            self.feature_names,
            self.config.collinearity_threshold,
            cast(Literal["spearman", "pearson"], self.config.collinearity_method),
        )

        updated_recommendations = list(recommendations)
        updated_reasons = list(reasons)
        feature_index = {name: i for i, name in enumerate(self.feature_names)}

        for cluster in clusters:
            if len(cluster) < 2:
                continue
            cluster_indices = [feature_index[f] for f in cluster]
            keep_mates = [
                f
                for f, i in zip(cluster, cluster_indices, strict=True)
                if updated_recommendations[i] == "Keep"
            ]
            if not keep_mates:
                continue
            for feat, i in zip(cluster, cluster_indices, strict=True):
                if updated_recommendations[i] != "Drop":
                    continue
                updated_recommendations[i] = "Caution"
                mates = ", ".join(m for m in keep_mates if m != feat) or keep_mates[0]
                note = (
                    f"Correlated with kept feature(s) {mates} (collinearity "
                    f"threshold {self.config.collinearity_threshold:.2f}); low "
                    f"importance may reflect redundancy rather than "
                    f"uselessness, so this is no longer flagged as an "
                    f"outright Drop. (Note: if collinearity clustering below "
                    f"also marks this a CollinearDuplicate, that label takes "
                    f"precedence over Caution as the final recommendation.)"
                )
                updated_reasons[i] = (
                    f"{updated_reasons[i]} {note}".strip()
                    if updated_reasons[i]
                    else note
                )

        return updated_recommendations, updated_reasons

    def _apply_shadow_probe_gating(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Downgrade Drop to Caution when shadow probes dispute it.

        `shadow_probe_flag` (already computed on `results` by
        add_shadow_probe_warnings) is False when a feature's importance
        clears the empirical noise floor -- the importance level a
        genuinely uninformative, randomly-shuffled column reaches. That's
        an independent, distribution-relative check the fixed
        shap/permutation-importance thresholds don't have (see Boruta and
        similar shadow-feature methods in the feature-selection
        literature). When it disputes a Drop the fixed thresholds
        produced, that Drop isn't well-supported: downgrade to Caution
        rather than leaving it as an outright Drop.

        Only reachable when shadow probes are active -- either the user set
        n_shadow_probes > 0 (opt-in, 0 by default), or severe class
        imbalance was detected and probes were auto-enabled (see run() and
        _apply_severe_imbalance_noise_floor_gating). Default pipeline
        behavior on balanced data with n_shadow_probes unset is unaffected
        either way.

        Args:
            results: The results DataFrame, already carrying
                shadow_probe_flag (real feature rows only -- probe rows
                must already be stripped).

        Returns:
            results, with any disputed Drop rows downgraded to Caution and
            their reason updated in place.
        """
        results = results.copy()
        disputed_drop_mask = (results["recommendation"] == "Drop") & (
            ~results["shadow_probe_flag"]
        )
        for idx in results.index[disputed_drop_mask]:
            results.loc[idx, "recommendation"] = "Caution"
            pct = self._effective_shadow_probe_percentile
            note = (
                "Beats the shadow-probe noise floor "
                f"(shadow_probe_percentile={pct:.1f}) "
                "despite the fixed importance threshold flagging it Drop; "
                "the empirical noise-floor check disputes that call, so "
                "this is no longer flagged as an outright Drop."
            )
            existing_reason = results.loc[idx, "reason"]
            results.loc[idx, "reason"] = (
                f"{existing_reason} {note}".strip() if existing_reason else note
            )
        return results

    def _apply_severe_imbalance_noise_floor_gating(
        self, results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Under severe class imbalance, demote Keep/Caution to Drop when a
        feature is at or below the shadow-probe noise floor.

        Severe class imbalance inflates SHAP importance for genuinely
        uninformative features -- with few minority-class examples,
        boosting finds *some* apparent structure in noise columns purely
        from that small sample's own variance, pushing their SHAP
        importance above the fixed shap_importance_threshold (confirmed
        empirically: noise-feature SHAP can reach 0.02-0.36 under this
        condition, all above the default 0.01 constant). This isn't a
        class-weighting problem (scale_pos_weight doesn't change it,
        verified directly) -- it's the same "fixed absolute threshold
        breaks when the ambient noise level shifts" failure the collinearity
        and shadow-probe gating above already handle, just triggered by a
        different condition.

        A shadow probe's own SHAP importance inflates by the same
        mechanism, so it remains a valid relative noise floor even here
        (verified empirically: at n_shadow_probes=30, percentile=100.0, the
        floor correctly identifies close to 100% of true noise features).
        Unlike _apply_shadow_probe_gating (which only ever downgrades an
        existing Drop, and is unaffected by this method), this one is
        bidirectional -- it's the only way a noise feature gets flagged at
        all under severe imbalance, since the primary engine's fixed
        thresholds don't catch it in the first place.

        Only called from run() when self.severe_class_imbalance is True.

        Args:
            results: The results DataFrame, already carrying
                shadow_probe_flag (real feature rows only -- probe rows
                already stripped), after _apply_shadow_probe_gating has run.

        Returns:
            results, with any Keep/Caution feature at or below the noise
            floor demoted to Drop and its reason updated in place.
        """
        results = results.copy()
        at_floor_mask = results["shadow_probe_flag"] & (
            results["recommendation"] != "Drop"
        )
        for idx in results.index[at_floor_mask]:
            prior_recommendation = results.loc[idx, "recommendation"]
            results.loc[idx, "recommendation"] = "Drop"
            pct = self._effective_shadow_probe_percentile
            note = (
                "Severe class imbalance was detected for this dataset, "
                "under which the fixed importance thresholds are known to "
                "be unreliable (see check_class_imbalance); this feature's "
                "importance is at or below the shadow-probe noise floor "
                f"(shadow_probe_percentile={pct:.1f}), so it's flagged Drop "
                f"despite being {prior_recommendation!r} under the standard "
                "thresholds."
            )
            existing_reason = results.loc[idx, "reason"]
            results.loc[idx, "reason"] = (
                f"{existing_reason} {note}".strip() if existing_reason else note
            )
        return results

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
            candidate derived features (pairwise products and ratios, plus
            target-mean-encoded categorical interactions) are derived from the
            strongest SHAP interactions and scored by mutual information
            against the target (see synthetic_features.py). Candidates
            exceeding config.synthetic_config.mi_threshold are saved to
            `<output_stem>_suggested_features.csv` and made available on
            self.suggested_features. This is advisory only -- suggestions
            are never added to the model or training data. Setting
            synthesize_from_interactions=True without also setting
            compute_shap_interactions=True auto-enables the latter (see
            Config.__post_init__) rather than silently skipping synthesis.
        """
        self.load_data()
        if self.train_data is None or self.val_data is None or self.test_data is None:
            raise RuntimeError("load_data() did not populate train/val/test data")

        self._effective_n_shadow_probes = self.config.n_shadow_probes
        self._effective_shadow_probe_percentile = self.config.shadow_probe_percentile
        if self._effective_n_shadow_probes == 0 and self.severe_class_imbalance:
            # The fixed shap/permutation-importance thresholds are known to
            # be unreliable under this level of imbalance (see
            # check_class_imbalance's warning) -- auto-enable shadow probes
            # as a self-calibrating noise floor instead, at Boruta's own
            # convention (100th percentile) and enough probes (30) for a
            # stable estimate. Only activates when the user didn't already
            # set n_shadow_probes themselves; default behavior for balanced
            # data is completely unaffected.
            self._effective_n_shadow_probes = 30
            self._effective_shadow_probe_percentile = 100.0
            self.logger.info(
                "Severe class imbalance detected: auto-enabling 30 shadow"
                " probes (100th percentile noise floor) to cross-check"
                " Keep/Caution features against, since the fixed importance"
                " thresholds alone are unreliable at this level of"
                " imbalance."
            )

        if self._effective_n_shadow_probes > 0:
            from .shadow_probes import generate_shadow_probes

            train_probes = generate_shadow_probes(
                self.train_data[self.feature_names],
                self._effective_n_shadow_probes,
                self.config.random_seed,
            )
            val_probes = generate_shadow_probes(
                self.val_data[self.feature_names],
                self._effective_n_shadow_probes,
                self.config.random_seed,
            )
            test_probes = generate_shadow_probes(
                self.test_data[self.feature_names],
                self._effective_n_shadow_probes,
                self.config.random_seed,
            )
            self.train_data = pd.concat([self.train_data, train_probes], axis=1)
            self.val_data = pd.concat([self.val_data, val_probes], axis=1)
            self.test_data = pd.concat([self.test_data, test_probes], axis=1)
            self.feature_names.extend(list(train_probes.columns))

        if self.config.custom_model is not None:
            self.logger.info(
                "config.custom_model is set; skipping Optuna tuning (it is"
                " XGBoost-specific) and fitting the supplied estimator"
                " directly."
            )
            best_params = {}
        elif self.config.skip_tuning:
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

        # Under streaming, bound the whole-frame explainability/diagnostic
        # computations to an in-memory sample (no-op when streaming is off or
        # the split already fits the cap -- see _eval_sample). The held-out
        # test set drives importance/permutation/MI/residuals; the combined
        # frame drives leakage. Sampling is deterministic (random_seed).
        test_eval = self._eval_sample(self.test_data)

        shap_importance = compute_shap_importance(
            model=self.model,
            data=test_eval,
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
            train_val_data = self._eval_sample(self._train_val_data())
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
            data=test_eval,
            target_column=self.config.target_column,
            feature_names=self.feature_names,
            metric_fn=metric_fn,
            use_predict_proba=(self.config.task == "binary"),
            n_repeats=self.config.n_permutations,
            parallel=self.config.parallel_feature_evaluation,
            n_jobs=self.config.n_jobs,
        )

        mi_score = compute_mutual_information(
            data=test_eval,
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
        leakage_warnings: dict[str, list[str]] = {
            name: [] for name in self.feature_names
        }
        temporal_structure_warnings: dict[str, list[str]] = {
            name: [] for name in self.feature_names
        }
        lookahead_warnings: dict[str, list[str]] = {
            name: [] for name in self.feature_names
        }
        if not self.config.skip_leakage_check:
            leakage_detector = LeakageDetector(
                self.config.leakage_config, screening_mode=self.config.leakage_screening
            )
            combined_full = pd.concat(
                [self.train_data, self.val_data, self.test_data], ignore_index=True
            )
            # The target-correlation / proxy / multicollinearity checks in
            # detect() are order-independent statistics, so under streaming
            # they run on a bounded random sample. detect_temporal_structure,
            # by contrast, measures order-dependent autocorrelation -- a
            # random sample would be meaningless -- so it always sees the
            # full, ordered frame regardless of streaming.
            combined_data = self._eval_sample(combined_full)
            leakage_warnings = leakage_detector.detect(
                df=combined_data,
                target_column=self.config.target_column,
                feature_names=self.feature_names,
                shap_importance=shap_importance,
                task=self.config.task,
            )
            temporal_structure_warnings = leakage_detector.detect_temporal_structure(
                df=combined_full,
                feature_names=self.feature_names,
            )

            # Check for look-ahead leakage if chronological split is enabled
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
            # Use the same sampled rows for both residuals and their
            # correlation so the residual vector aligns with the feature rows.
            residuals = compute_residuals(
                model=self.model,
                data=test_eval,
                target_column=self.config.target_column,
                feature_names=self.feature_names,
                task=self.config.task,
            )
            residual_correlation = compute_residual_correlation(
                data=test_eval,
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

        if self.config.compute_confidence_scores:
            from .confidence import compute_confidence_scores

            result_columns["confidence_score"] = compute_confidence_scores(
                shap_importance=shap_importance,
                perm_importance=perm_importance,
                recommendations=result_columns["recommendation"],
                shap_threshold=self.config.shap_importance_threshold,
                perm_threshold=self.config.permutation_importance_threshold,
            )

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
                if (
                    self.config.compute_confidence_scores
                    and "confidence_score" in result_columns
                ):
                    for i, rec in enumerate(recommendations):
                        if rec == "Caution":
                            result_columns["confidence_score"][i] = 0.5

        # Collinearity gating runs after stability gating so it only
        # reconsiders whatever Drops remain once other adjustments have
        # already been applied.
        if self.config.collinearity_threshold > 0:
            recommendations, reasons = self._apply_collinearity_gating(
                recommendations, reasons
            )
            result_columns["recommendation"] = recommendations
            result_columns["reason"] = reasons
            if (
                self.config.compute_confidence_scores
                and "confidence_score" in result_columns
            ):
                for i, rec in enumerate(recommendations):
                    if rec == "Caution":
                        result_columns["confidence_score"][i] = 0.5

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
            train_val_data = self._train_val_data()
            result_columns["boruta_decision"] = run_boruta_shap(
                model=self.model,
                data=train_val_data,
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
            train_val_data = self._train_val_data()
            result_columns["catboost_importance"] = run_catboost_importance(
                data=train_val_data,
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

        if self.config.collinearity_threshold > 0:
            from .collinearity import (
                cluster_correlated_features,
                select_cluster_representative,
            )

            clusters = cluster_correlated_features(
                self.X_train if self.X_train is not None else pd.DataFrame(),
                self.feature_names,
                self.config.collinearity_threshold,
                cast(Literal["spearman", "pearson"], self.config.collinearity_method),
            )
            shap_map = dict(
                zip(
                    self.results["feature_name"],
                    self.results["shap_importance"],
                    strict=True,
                )
            )
            cluster_labels: dict[str, str] = {}
            duplicate_features: set[str] = set()
            for i, cluster in enumerate(clusters):
                label = "singleton" if len(cluster) == 1 else f"cluster_{i}"
                rep = select_cluster_representative(cluster, shap_map)
                for feat in cluster:
                    cluster_labels[feat] = label
                    if feat != rep and len(cluster) > 1:
                        duplicate_features.add(feat)
            self.results["collinearity_cluster"] = self.results["feature_name"].map(
                cluster_labels
            )
            mask = self.results["feature_name"].isin(duplicate_features)
            self.results.loc[mask, "recommendation"] = "CollinearDuplicate"

        if self.config.n_stability_runs > 0:
            seeds = self.config.stability_seeds or [
                self.config.random_seed + i + 1
                for i in range(self.config.n_stability_runs)
            ]
            # Collect per-feature recommendations across all runs, including main.
            all_recs: dict[str, list[str]] = {
                f: [r] for f, r in zip(self.feature_names, recommendations, strict=True)
            }
            self.logger.info(
                f"Running {self.config.n_stability_runs} additional seeds for"
                " recommendation stability analysis..."
            )
            import dataclasses

            for seed in seeds:
                seed_cfg = dataclasses.replace(
                    self.config,
                    random_seed=seed,
                    n_stability_runs=0,  # prevent recursion
                    stability_seeds=None,
                )
                seed_results = FeatureSelectionPipeline(seed_cfg).run()
                seed_rec_map = dict(
                    zip(
                        seed_results["feature_name"],
                        seed_results["recommendation"],
                        strict=True,
                    )
                )
                for feat in self.feature_names:
                    all_recs[feat].append(seed_rec_map.get(feat, "Unknown"))

            # Build stability_score + stability_votes aligned to results order.
            feature_order = self.results["feature_name"].tolist()
            stability_scores = []
            stability_votes_list = []
            for feat in feature_order:
                feat_recs = all_recs[feat]
                final_rec = self.results.loc[
                    self.results["feature_name"] == feat, "recommendation"
                ].iloc[0]
                agree = sum(1 for r in feat_recs if r == final_rec)
                score = agree / len(feat_recs)
                from collections import Counter

                vote_counts = Counter(feat_recs)
                votes_str = ",".join(f"{k}:{v}" for k, v in sorted(vote_counts.items()))
                stability_scores.append(round(score, 4))
                stability_votes_list.append(votes_str)

            self.results["stability_score"] = stability_scores
            self.results["stability_votes"] = stability_votes_list
            self.logger.info(
                f"Stability analysis done:"
                f" {sum(1 for s in stability_scores if s == 1.0)} of"
                f" {len(stability_scores)} features unanimous across all seeds."
            )

        full_feature_names = list(self.feature_names)
        if self._effective_n_shadow_probes > 0:
            from .shadow_probes import (
                add_shadow_probe_warnings,
                compute_probe_threshold,
            )

            probe_rows = self.results[
                self.results["feature_name"].str.startswith("__shadow_probe_")
            ]
            probe_importances = probe_rows["shap_importance"].to_numpy()
            probe_threshold = compute_probe_threshold(
                probe_importances, self._effective_shadow_probe_percentile
            )

            self.results = self.results[
                ~self.results["feature_name"].str.startswith("__shadow_probe_")
            ].reset_index(drop=True)
            self.results = add_shadow_probe_warnings(self.results, probe_threshold)
            self.results = self._apply_shadow_probe_gating(self.results)
            if self.severe_class_imbalance:
                self.results = self._apply_severe_imbalance_noise_floor_gating(
                    self.results
                )

            real_indices = [
                i
                for i, name in enumerate(self.feature_names)
                if not name.startswith("__shadow_probe_")
            ]
            self.feature_names = [self.feature_names[i] for i in real_indices]
            shap_importance = shap_importance[real_indices]

        save_results(
            results=self.results,
            output_file=self.config.output_file,
            output_format=self.config.output_format,
        )

        # Shared by every "<output_stem>_<suffix>.csv" derived-output path
        # below (group summary, SHAP interactions, synthetic suggestions).
        stem = derive_output_stem(self.config.output_file)

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
            groups_path = f"{stem}_groups.csv"
            group_summary.to_csv(groups_path, index=False)
            self.logger.info(
                f"Group summary ({len(group_summary)} groups) saved to {groups_path}"
            )

        if self.config.compute_shap_interactions:
            # Interaction discovery and candidate MI-screening both run on
            # train+val, not test: scoring candidates against the same rows
            # used later to validate them (in _evaluate_synthetic_replacements)
            # would let a synthetic feature "win" purely from fitting held-out
            # noise it was mined from. The real held-out check stays in
            # _evaluate_synthetic_replacements, which fits on train+val and
            # scores on self.test_data, untouched by this.
            train_val_data = self._train_val_data()
            self.interactions = compute_shap_interactions(
                model=self.model,
                data=train_val_data,
                feature_names=full_feature_names,
            )
            if self.interactions is not None and self._effective_n_shadow_probes > 0:
                self.interactions = self.interactions[
                    ~self.interactions["feature_a"].str.startswith("__shadow_probe_")
                    & ~self.interactions["feature_b"].str.startswith("__shadow_probe_")
                ].reset_index(drop=True)
            interactions_file = f"{stem}_interactions.csv"
            self.interactions.to_csv(interactions_file, index=False)
            self.logger.info(f"SHAP interaction values saved to {interactions_file}")

            if self.config.synthetic_config.synthesize_from_interactions:
                self.suggested_features = suggest_synthetic_features(
                    data=train_val_data,
                    target_column=self.config.target_column,
                    interactions=self.interactions,
                    task=self.config.task,
                    random_seed=self.config.random_seed,
                    config=self.config.synthetic_config,
                    # Categorical target-mean encoding is fit in-sample on
                    # train+val here (screening only) since data and
                    # encoder_source are now the same split -- there's no
                    # third split to fit it on separately. Real held-out
                    # validation happens in _evaluate_synthetic_replacements.
                    encoder_source=train_val_data,
                )
                suggestions_file = f"{stem}_suggested_features.csv"
                pd.DataFrame([vars(s) for s in self.suggested_features]).to_csv(
                    suggestions_file, index=False
                )
                self.logger.info(
                    f"{len(self.suggested_features)} synthetic feature"
                    f" suggestions saved to {suggestions_file}"
                )

                if (
                    self.config.synthetic_config.dfe_loop_feedback
                    and self.suggested_features
                ):
                    replaced, winners = self._evaluate_synthetic_replacements()
                    if replaced:
                        mask = self.results["feature_name"].isin(replaced)
                        self.results.loc[mask, "recommendation"] = "ReplacedBySynthetic"
                        self.results = self._append_synthetic_winner_rows(
                            self.results, winners
                        )
                        self.logger.info(
                            f"DFE loop feedback: {len(replaced)} raw features"
                            f" outperformed by a synthetic derivative, marked"
                            f" ReplacedBySynthetic: {sorted(replaced)};"
                            f" added {len(winners)} synthetic feature(s) to"
                            f" results: {sorted(winners)}"
                        )
        # No elif for synthesize_from_interactions-without-
        # compute_shap_interactions here: Config.__post_init__ now
        # guarantees compute_shap_interactions=True whenever
        # synthesize_from_interactions=True, for every caller (CLI, MCP,
        # direct Config construction) -- that combination can no longer
        # reach this point.

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

    def _evaluate_synthetic_replacements(
        self,
    ) -> tuple[set[str], dict[str, float]]:
        """Re-fit on the augmented feature set and return raw features that a
        synthetic derivative genuinely outperforms.

        DFE loop feedback should only demote a raw feature when its synthetic
        replacement actually wins on the model's own terms, not merely because
        a synthetic was derived from it. This runs a real competition:

        1. Build train+val and test frames augmented with the injected
           synthetics. Categorical constituents are target-mean encoded with a
           map fit on the training split only (via ``fit_interaction_encoders``)
           and applied unchanged to the test split, so a synthetic can't win
           through test-set target leakage.
        2. Fit a fresh model (reusing the tuned ``best_params``) on the
           augmented training features.
        3. Compute SHAP importance on the augmented test split for the raw
           features plus the synthetics together.
        4. Mark a raw constituent ``ReplacedBySynthetic`` only when its
           synthetic derivative's SHAP importance strictly exceeds its own.

        Returns ``(replaced, winners)`` where ``replaced`` is the set of raw
        feature names to demote and ``winners`` maps each winning synthetic
        feature name to its SHAP importance on the augmented test split (so the
        caller can surface the replacement in the results table). Both are empty
        when nothing was injectable or no synthetic beat its constituents.
        """
        if (
            self.test_data is None
            or self.results is None
            or not self.suggested_features
        ):
            return set(), {}

        cfg = self.config.synthetic_config
        target_column = self.config.target_column
        train_pool = self._train_val_data()

        # Fit categorical encoders on training data only, then apply the same
        # map to both splits so the encoding never sees held-out targets.
        encoders = fit_interaction_encoders(
            train_pool, target_column, self.suggested_features
        )
        augmented_train, injected_train = build_injection_frame(
            train_pool,
            target_column,
            self.suggested_features,
            cfg.dfe_max_inject,
            encoders,
        )
        augmented_test, injected_test = build_injection_frame(
            self.test_data,
            target_column,
            self.suggested_features,
            cfg.dfe_max_inject,
            encoders,
        )
        # Only compete synthetics that were buildable on both splits.
        injected = [name for name in injected_train if name in set(injected_test)]
        if not injected:
            return set(), {}

        self.logger.info(
            f"DFE loop feedback: competing {len(injected)} synthetic feature(s)"
            f" against their raw constituents: {injected}"
        )

        compete_features = self.feature_names + injected
        params = _build_xgb_fit_params(self._best_params, self.config)
        model = _make_xgb_model(self.config.task, params)
        model.fit(augmented_train[compete_features], augmented_train[target_column])

        importances = compute_shap_importance(
            model=model,
            data=augmented_test,
            feature_names=compete_features,
            sample_size=self.config.shap_sample_size,
            random_seed=self.config.random_seed,
        )
        imp_by_feature = dict(zip(compete_features, importances, strict=True))

        known = set(self.results["feature_name"])
        injected_set = set(injected)
        replaced: set[str] = set()
        winners: dict[str, float] = {}
        for suggestion in self.suggested_features[: cfg.dfe_max_inject]:
            name = suggestion.suggested_name
            if name not in injected_set:
                continue
            syn_imp = imp_by_feature.get(name, 0.0)
            for raw in (suggestion.feature_a, suggestion.feature_b):
                if raw in known and syn_imp > imp_by_feature.get(raw, 0.0):
                    replaced.add(raw)
                    winners[name] = syn_imp
        return replaced, winners

    def _append_synthetic_winner_rows(
        self, results: pd.DataFrame, winners: dict[str, float]
    ) -> pd.DataFrame:
        """Append winning DFE synthetic features to the results table as Keep rows.

        When loop feedback demotes raw features to ``ReplacedBySynthetic``, the
        synthetics that beat them would otherwise only appear in the
        ``_suggested_features.csv`` sidecar, leaving the main results table with
        no visible replacement. This adds one ``Keep`` row per winning synthetic
        (carrying its measured SHAP importance) so the recommended replacement
        is surfaced alongside the features it displaced. Columns not applicable
        to a synthesized feature (e.g. permutation importance) are left as NaN.
        """
        if not winners:
            return results

        new_rows = [
            {
                "feature_name": name,
                "shap_importance": importance,
                "recommendation": "Keep",
                "reason": (
                    "Synthetic feature engineered by DFE loop feedback;"
                    " outperformed its raw constituent(s) on held-out SHAP"
                    " importance."
                ),
            }
            for name, importance in winners.items()
        ]
        additions = pd.DataFrame(new_rows).reindex(columns=results.columns)
        return pd.concat([results, additions], ignore_index=True)

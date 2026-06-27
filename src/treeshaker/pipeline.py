"""Core feature selection pipeline orchestration."""

import logging
import os
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .harm_detection import HarmDetector
from .importance import (
    compute_mutual_information,
    compute_permutation_importance,
    compute_shap_importance,
    compute_shap_interactions,
)
from .leakage_detection import LeakageDetector
from .utils import check_class_imbalance, get_metric_fn, load_csv, save_results


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
        )
        self.feature_names = [
            col for col in self.train_data.columns if col != self.config.target_column
        ]
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
        xgb_cfg = self.config.xgboost_config
        task = self.config.task

        X_train = self.train_data[self.feature_names]
        y_train = self.train_data[self.config.target_column]
        X_val = self.val_data[self.feature_names]
        y_val = self.val_data[self.config.target_column]

        def objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "random_state": self.config.random_seed,
                "verbosity": 0,
            }
            params.update(xgb_cfg.extra_params)

            if task in ("binary", "multiclass"):
                objective_str = (
                    "binary:logistic" if task == "binary" else "multi:softprob"
                )
                model = xgb.XGBClassifier(objective=objective_str, **params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                # Binary and multiclass are both scored by log loss on the
                # predicted class probabilities.
                preds = model.predict_proba(X_val)
                return log_loss(y_val, preds)
            else:
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                preds = model.predict(X_val)
                return mean_squared_error(y_val, preds)

        sampler = optuna.samplers.TPESampler(seed=cfg.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
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
        import xgboost as xgb

        if self.train_data is None or self.val_data is None:
            raise RuntimeError("load_data() must be called before train_model()")

        combined = pd.concat([self.train_data, self.val_data], ignore_index=True)
        X = combined[self.feature_names]
        y = combined[self.config.target_column]

        params = {
            **best_params,
            "random_state": self.config.random_seed,
            "verbosity": 0,
        }
        params.update(self.config.xgboost_config.extra_params)

        if self.config.task in ("binary", "multiclass"):
            objective_str = (
                "binary:logistic" if self.config.task == "binary" else "multi:softprob"
            )
            self.model = xgb.XGBClassifier(objective=objective_str, **params)
        else:
            self.model = xgb.XGBRegressor(**params)

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
        import xgboost as xgb

        from .importance import compute_shap_importance

        if self.train_data is None or self.val_data is None or self.test_data is None:
            raise RuntimeError(
                "load_data() must be called before _run_bootstrap_importance()"
            )

        bootstrap_cfg = self.config.bootstrap_config
        combined = pd.concat([self.train_data, self.val_data], ignore_index=True)
        n_rows = len(combined)
        sample_size = max(1, int(round(n_rows * bootstrap_cfg.sample_frac)))

        params = {
            **best_params,
            "random_state": self.config.random_seed,
            "verbosity": 0,
        }
        params.update(self.config.xgboost_config.extra_params)

        results = np.zeros((bootstrap_cfg.n_bootstrap, len(self.feature_names)))
        for i in range(bootstrap_cfg.n_bootstrap):
            seed = self.config.random_seed + i
            rng = np.random.default_rng(seed=seed)
            sample_idx = rng.integers(0, n_rows, size=sample_size)
            resampled = combined.iloc[sample_idx].reset_index(drop=True)

            X_resampled = resampled[self.feature_names]
            y_resampled = resampled[self.config.target_column]

            iter_params = {**params, "random_state": seed}
            if self.config.task in ("binary", "multiclass"):
                objective_str = (
                    "binary:logistic"
                    if self.config.task == "binary"
                    else "multi:softprob"
                )
                bootstrap_model = xgb.XGBClassifier(
                    objective=objective_str, **iter_params
                )
            else:
                bootstrap_model = xgb.XGBRegressor(**iter_params)

            bootstrap_model.fit(X_resampled, y_resampled)

            results[i] = compute_shap_importance(
                model=bootstrap_model,
                data=self.test_data,
                feature_names=self.feature_names,
            )
            self.logger.info(
                f"Bootstrap importance iteration {i + 1}/{bootstrap_cfg.n_bootstrap}"
                " complete"
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
        is_harmful, harm_reasons = detector.flag_harmful_features(
            shap_importance=shap_importance,
            perm_importance=perm_importance,
        )

        # Leakage detection looks at statistical properties (e.g. target
        # correlation) across the whole dataset, not held-out generalization
        # -- unlike importance evaluation above, using all three splits here
        # is correct and intentional.
        leakage_detector = LeakageDetector(self.config.leakage_config)
        combined_data = pd.concat(
            [self.train_data, self.val_data, self.test_data], ignore_index=True
        )
        leakage_warnings = leakage_detector.detect(
            df=combined_data,
            target_column=self.config.target_column,
            feature_names=self.feature_names,
            shap_importance=shap_importance,
        )

        recommendations = []
        reasons = []
        for i, feature in enumerate(self.feature_names):
            lk_warnings = leakage_warnings[feature]
            lk_text = " | ".join(lk_warnings) if lk_warnings else ""

            if is_harmful[i] and lk_text:
                recommendations.append("Drop")
                reasons.append(harm_reasons[i] + " Also: " + lk_text)
            elif is_harmful[i]:
                recommendations.append("Drop")
                reasons.append(harm_reasons[i])
            elif lk_text:
                recommendations.append("Caution")
                reasons.append(lk_text)
            else:
                recommendations.append("Keep")
                reasons.append("")

        result_columns: dict[str, Any] = {
            "feature_name": self.feature_names,
            "shap_importance": shap_importance,
            "permutation_importance": perm_importance,
            "permutation_importance_std": perm_importance_std,
            "mutual_info_score": mi_score,
            "recommendation": recommendations,
            "reason": reasons,
        }

        if self.config.bootstrap_config.n_bootstrap > 0:
            bootstrap_importances = self._run_bootstrap_importance(best_params)
            result_columns["shap_importance_mean"] = bootstrap_importances.mean(axis=0)
            result_columns["shap_importance_std"] = bootstrap_importances.std(axis=0)

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
            stem, _ = os.path.splitext(self.config.output_file)
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
            stem, _ = os.path.splitext(self.config.output_file)
            interactions_file = f"{stem}_interactions.csv"
            self.interactions.to_csv(interactions_file, index=False)
            self.logger.info(f"SHAP interaction values saved to {interactions_file}")

        n_drop = (self.results["recommendation"] == "Drop").sum()
        n_caution = (self.results["recommendation"] == "Caution").sum()
        self.logger.info(
            f"Done. {len(self.results)} features evaluated:"
            f" {n_drop} to drop, {n_caution} flagged as caution."
        )
        return self.results

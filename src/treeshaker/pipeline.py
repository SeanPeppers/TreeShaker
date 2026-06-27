"""Core feature selection pipeline orchestration."""

import logging
import os
from typing import Any

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


class FeatureSelectionPipeline:
    """
    Orchestrates the full TreeShaker feature selection workflow.

    The pipeline runs in the following order:
    1. Load CSV data and split into train/validation sets.
    2. Tune XGBoost hyperparameters with Optuna.
    3. Train the final XGBoost model on tuned hyperparameters.
    4. Compute SHAP importance for each feature.
    5. Compute permutation importance for each feature.
    6. Compute mutual information between each feature and the target.
    7. Flag harmful features using both importance signals.
    8. Detect data leakage warnings.
    9. Save results to CSV or HTML.
    10. (Optional, opt-in) Compute pairwise SHAP interaction values and save
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
        self.model: Any = None
        self.feature_names: list[str] = []
        self.results: pd.DataFrame | None = None
        self.interactions: pd.DataFrame | None = None

    def load_data(self) -> None:
        """
        Load the CSV file and split into train and validation sets.

        Populates self.train_data, self.val_data, and self.feature_names.
        """
        self.train_data, self.val_data = load_csv(
            filepath=self.config.input_file,
            target_column=self.config.target_column,
            task=self.config.task,
            test_size=self.config.test_size,
            random_seed=self.config.random_seed,
        )
        self.feature_names = [
            col for col in self.train_data.columns if col != self.config.target_column
        ]
        self.logger.info(f"Features to evaluate: {len(self.feature_names)}")

        combined = pd.concat([self.train_data, self.val_data], ignore_index=True)
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

    def run(self) -> pd.DataFrame:
        """
        Execute the full feature selection pipeline.

        Returns:
            DataFrame with columns: feature_name, shap_importance,
            permutation_importance, mutual_info_score, recommendation, reason.
            recommendation is "Drop", "Caution", or "Keep".
            reason is a plain English explanation (empty string if "Keep").
            Sorted by shap_importance descending.

            If config.compute_shap_interactions is True, pairwise SHAP
            interaction values are also computed and saved to
            `<output_stem>_interactions.csv`, and made available on
            self.interactions. This does not change the shape or schema
            of the returned DataFrame.
        """
        self.load_data()
        if self.train_data is None or self.val_data is None:
            raise RuntimeError("load_data() did not populate train/val data")

        best_params = self.tune_hyperparams()
        self.train_model(best_params)

        metric_fn = get_metric_fn(self.config.task)

        shap_importance = compute_shap_importance(
            model=self.model,
            data=self.val_data,
            feature_names=self.feature_names,
        )

        perm_importance = compute_permutation_importance(
            model=self.model,
            data=self.val_data,
            target_column=self.config.target_column,
            feature_names=self.feature_names,
            metric_fn=metric_fn,
            use_predict_proba=(self.config.task == "binary"),
        )

        mi_score = compute_mutual_information(
            data=self.val_data,
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

        leakage_detector = LeakageDetector(self.config.leakage_config)
        combined_data = pd.concat([self.train_data, self.val_data], ignore_index=True)
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

        self.results = (
            pd.DataFrame(
                {
                    "feature_name": self.feature_names,
                    "shap_importance": shap_importance,
                    "permutation_importance": perm_importance,
                    "mutual_info_score": mi_score,
                    "recommendation": recommendations,
                    "reason": reasons,
                }
            )
            .sort_values("shap_importance", ascending=False)
            .reset_index(drop=True)
        )

        save_results(
            results=self.results,
            output_file=self.config.output_file,
            output_format=self.config.output_format,
        )

        if self.config.compute_shap_interactions:
            self.interactions = compute_shap_interactions(
                model=self.model,
                data=self.val_data,
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

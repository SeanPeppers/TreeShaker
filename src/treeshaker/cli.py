"""Command-line interface for TreeShaker."""

import argparse
import json
import logging

from .config import BootstrapConfig, Config, OptunaConfig, XGBoostConfig
from .exceptions import ConfigError, TreeShakerError
from .leakage_detection import LeakageConfig
from .pipeline import FeatureSelectionPipeline


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for TreeShaker.

    Returns:
        argparse.Namespace with all parsed arguments.
    """

    parser = argparse.ArgumentParser(
        prog="treeshaker",
        description="Automated feature selection using XGBoost, SHAP, and Optuna.",
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--target", required=True, help="Name of the target column.")
    parser.add_argument(
        "--task",
        required=True,
        choices=["binary", "multiclass", "regression"],
        help="Prediction task type.",
    )
    parser.add_argument(
        "--output",
        default="treeshaker_results.csv",
        help="Output file path. (default: treeshaker_results.csv)",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        default="csv",
        choices=["csv", "html"],
        help="Output format. (default: csv)",
    )
    parser.add_argument(
        "--xgboost-params",
        default="{}",
        help="Custom XGBoost params as a JSON string. (default: {})",
    )
    parser.add_argument(
        "--feature-groups",
        default="{}",
        help="Feature groups as a JSON string mapping group name to a list of"
        ' feature names, e.g. \'{"price": ["price_min", "price_max"]}\'.'
        " (default: {})",
    )
    parser.add_argument(
        "--group-by-prefix",
        action="store_true",
        help="Auto-derive feature groups from the common prefix before the"
        " last underscore (e.g. price_min/price_max -> group 'price')."
        " Ignored if --feature-groups is set.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=50,
        help="Number of Optuna trials. (default: 50)",
    )
    parser.add_argument(
        "--optuna-timeout",
        type=int,
        default=300,
        help="Optuna timeout in seconds. (default: 300)",
    )
    parser.add_argument(
        "--shap-threshold",
        type=float,
        default=0.01,
        help="Features with mean |SHAP| below this are flagged harmful."
        " (default: 0.01)",
    )
    parser.add_argument(
        "--perm-threshold",
        type=float,
        default=0.05,
        help="Features with permutation importance below this are flagged"
        " harmful. (default: 0.05)",
    )
    parser.add_argument(
        "--target-corr-threshold",
        type=float,
        default=0.95,
        help="Flag features with |correlation to target| above this. (default: 0.95)",
    )
    parser.add_argument(
        "--dominant-shap-fraction",
        type=float,
        default=0.7,
        help="Flag a feature that accounts for more than this fraction"
        " of total SHAP. (default: 0.7)",
    )
    parser.add_argument(
        "--skip-leakage-check",
        action="store_true",
        help="Disable leakage and logical error detection.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip Optuna hyperparameter tuning and train the final model"
        " directly using the configured (default or --xgboost-params)"
        " XGBoost parameters.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=0,
        help="Number of bootstrap resamples to refit and assess SHAP"
        " importance ranking stability. 0 disables bootstrap importance."
        " (default: 0)",
    )
    parser.add_argument(
        "--bootstrap-sample-frac",
        type=float,
        default=1.0,
        help="Fraction of combined train+val rows to sample (with"
        " replacement) per bootstrap iteration. (default: 1.0)",
    )
    parser.add_argument(
        "--shap-interactions",
        action="store_true",
        help="Compute pairwise SHAP interaction values (expensive, opt-in)."
        " Saves top interacting feature pairs to"
        " <output_stem>_interactions.csv.",
    )
    parser.add_argument(
        "--imbalance-threshold",
        type=float,
        default=0.1,
        help="Warn if the minority class fraction (binary/multiclass) falls"
        " below this. (default: 0.1)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for validation. (default: 0.2)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. (default: 42)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    """
    Entry point for the TreeShaker CLI.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    args = parse_arguments()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        xgb_params = json.loads(args.xgboost_params)
    except json.JSONDecodeError as exc:
        logger.error(f"Invalid --xgboost-params JSON: {exc}")
        return 1

    try:
        feature_groups = json.loads(args.feature_groups) or None
    except json.JSONDecodeError as exc:
        logger.error(f"Invalid --feature-groups JSON: {exc}")
        return 1

    leakage_config = LeakageConfig(
        target_corr_threshold=args.target_corr_threshold,
        dominant_shap_fraction=args.dominant_shap_fraction,
    )

    try:
        config = Config(
            input_file=args.input,
            target_column=args.target,
            task=args.task,
            output_file=args.output,
            output_format=args.output_format,
            test_size=args.test_size,
            random_seed=args.random_seed,
            optuna_config=OptunaConfig(
                n_trials=args.optuna_trials,
                timeout=args.optuna_timeout,
                seed=args.random_seed,
            ),
            xgboost_config=XGBoostConfig(extra_params=xgb_params),
            shap_importance_threshold=args.shap_threshold,
            permutation_importance_threshold=args.perm_threshold,
            leakage_config=leakage_config,
            feature_groups=feature_groups,
            group_by_prefix=args.group_by_prefix,
            bootstrap_config=BootstrapConfig(
                n_bootstrap=args.n_bootstrap,
                sample_frac=args.bootstrap_sample_frac,
            ),
            compute_shap_interactions=args.shap_interactions,
            imbalance_warning_threshold=args.imbalance_threshold,
            verbose=args.verbose,
            skip_tuning=args.skip_tuning,
        )
    except ConfigError as exc:
        logger.error(f"Configuration error: {exc}")
        return 1

    try:
        pipeline = FeatureSelectionPipeline(config)
        pipeline.run()
        print(f"Results saved to: {config.output_file}")
        return 0
    except TreeShakerError as exc:
        logger.error(f"Pipeline error: {exc}")
        return 1
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}", exc_info=args.verbose)
        return 1

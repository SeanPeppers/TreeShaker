"""Configuration dataclasses for TreeShaker."""

from dataclasses import asdict, dataclass, field

from .exceptions import ConfigError
from .leakage_detection import LeakageConfig
from .residual_diagnostics import ResidualConfig
from .utils import VALID_PERMUTATION_METRICS


@dataclass
class OptunaConfig:
    """Configuration for Optuna hyperparameter tuning."""

    n_trials: int = 50
    timeout: int = 300
    seed: int = 42
    show_progress_bar: bool = False


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost model hyperparameters."""

    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.0
    min_child_weight: int = 1
    extra_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return merged XGBoost params dict including extra_params."""
        d = asdict(self)
        d.pop("extra_params")
        d.update(self.extra_params)
        return d


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap ensemble importance stability analysis."""

    n_bootstrap: int = 0
    sample_frac: float = 1.0


@dataclass
class Config:
    """Top-level pipeline configuration."""

    input_file: str
    target_column: str
    task: str
    output_file: str = "treeshaker_results.csv"
    output_format: str = "csv"
    test_size: float = 0.2
    """Fraction of the FULL dataset held out as the final test set, used for
    importance evaluation and leakage statistics. Never used for fitting any
    model (not during tuning, not in the final combined train+val fit).

    Note: prior to the train/val/test split fix, this field meant "held out
    validation fraction" (the model's final fit *included* this data). It now
    means "held out final test fraction" (this data is never fit on at all).
    Existing callers that relied on the old single-split behavior should
    review their --test-size value; numeric importance results will also
    change since they're now computed on a genuinely held-out set."""
    val_size: float = 0.2
    """Fraction of the remaining train+val pool (i.e. after test_size is
    removed from the full dataset) used as the validation set for Optuna
    trial evaluation / early stopping during hyperparameter tuning."""
    random_seed: int = 42
    optuna_config: OptunaConfig = field(default_factory=OptunaConfig)
    xgboost_config: XGBoostConfig = field(default_factory=XGBoostConfig)
    shap_importance_threshold: float = 0.01
    permutation_importance_threshold: float = 0.05
    compute_shap_interactions: bool = False
    leakage_config: LeakageConfig = field(default_factory=LeakageConfig)
    feature_groups: dict[str, list[str]] | None = None
    group_by_prefix: bool = False
    bootstrap_config: BootstrapConfig = field(default_factory=BootstrapConfig)
    imbalance_warning_threshold: float = 0.1
    verbose: bool = False
    skip_tuning: bool = False
    check_residuals: bool = False
    """If True, compute per-feature correlation with model prediction
    residuals on the held-out test set and surface a warning when a
    feature flagged "Drop" correlates with residuals beyond
    residual_config.residual_corr_threshold. Off by default; adds a
    residual_correlation column to the results DataFrame when enabled.
    See residual_diagnostics.py for the full definition of "residual" per
    task type."""
    residual_config: ResidualConfig = field(default_factory=ResidualConfig)
    n_permutations: int = 5
    """Number of shuffles per feature for permutation importance, averaged
    to reduce single-shuffle noise. 5 matches sklearn's own
    permutation_importance default -- a reasonable balance between
    stability and runtime."""
    permutation_metric: str | None = None
    """Optional override for the metric used specifically for permutation
    importance (does not affect Optuna tuning). One of "roc_auc",
    "accuracy", "neg_mse", or "wape". None (default) preserves the
    automatic per-task default: ROC-AUC for binary, accuracy for
    multiclass, negative MSE for regression."""

    def __post_init__(self) -> None:
        """Validate configuration values."""
        valid_tasks = {"binary", "multiclass", "regression"}
        if self.task not in valid_tasks:
            raise ConfigError(f"task must be one of {valid_tasks}, got {self.task!r}")
        if not 0 < self.test_size < 1:
            raise ConfigError(f"test_size must be in (0, 1), got {self.test_size}")
        if not 0 < self.val_size < 1:
            raise ConfigError(f"val_size must be in (0, 1), got {self.val_size}")
        if self.test_size + self.val_size >= 1:
            raise ConfigError(
                "test_size + val_size must be < 1 so train has positive size,"
                f" got test_size={self.test_size} + val_size={self.val_size}"
                f" = {self.test_size + self.val_size}"
            )
        valid_formats = {"csv", "html"}
        if self.output_format not in valid_formats:
            raise ConfigError(
                f"output_format must be one of {valid_formats}, "
                f"got {self.output_format!r}"
            )
        if self.bootstrap_config.n_bootstrap < 0:
            raise ConfigError(
                "bootstrap_config.n_bootstrap must be >= 0, got"
                f" {self.bootstrap_config.n_bootstrap}"
            )
        if not 0 < self.bootstrap_config.sample_frac <= 1.0:
            raise ConfigError(
                "bootstrap_config.sample_frac must be in (0, 1], got"
                f" {self.bootstrap_config.sample_frac}"
            )
        if self.residual_config.residual_corr_threshold <= 0:
            raise ConfigError(
                "residual_config.residual_corr_threshold must be > 0, got"
                f" {self.residual_config.residual_corr_threshold}"
            )
        if self.n_permutations < 1:
            raise ConfigError(f"n_permutations must be >= 1, got {self.n_permutations}")
        if (
            self.permutation_metric is not None
            and self.permutation_metric not in VALID_PERMUTATION_METRICS
        ):
            raise ConfigError(
                "permutation_metric must be one of"
                f" {sorted(VALID_PERMUTATION_METRICS)}, got"
                f" {self.permutation_metric!r}"
            )

"""Configuration dataclasses for TreeShaker."""

from dataclasses import asdict, dataclass, field
from typing import Literal

from .exceptions import ConfigError
from .leakage_detection import LeakageConfig
from .residual_diagnostics import ResidualConfig
from .synthetic_features import SyntheticConfig
from .utils import VALID_PERMUTATION_METRICS, detect_cuda_available


def _require_at_least(name: str, value: float, minimum: float) -> None:
    """Raise ConfigError unless value >= minimum.

    Shared by the several Config.__post_init__ checks of the form "field
    must be >= N", which otherwise differ only in the field name, the
    bound, and the comparison operator -- this covers the ">=" (inclusive
    lower bound) cases; see `_require_greater_than` for the strict "> N"
    cases.

    Args:
        name: Dotted field name as it should appear in the error message,
            e.g. "bootstrap_config.n_bootstrap".
        value: The value being validated.
        minimum: The inclusive lower bound.

    Raises:
        ConfigError: If value < minimum.
    """
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")


def _require_greater_than(name: str, value: float, lower_bound: float) -> None:
    """Raise ConfigError unless value > lower_bound.

    See `_require_at_least` for the inclusive-bound counterpart.

    Args:
        name: Dotted field name as it should appear in the error message.
        value: The value being validated.
        lower_bound: The exclusive lower bound.

    Raises:
        ConfigError: If value <= lower_bound.
    """
    if value <= lower_bound:
        raise ConfigError(f"{name} must be > {lower_bound}, got {value}")


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
    device: Literal["auto", "cuda", "cpu"] = "auto"
    """Which device XGBoost should train on. "auto" (default) uses a CUDA
    GPU automatically when one is usable, silently falling back to CPU
    otherwise -- no flag needed for the common case. Explicit "cuda" or
    "cpu" overrides auto-detection and is respected even if detection
    would suggest otherwise; forcing "cuda" when no GPU is actually
    usable raises ConfigError instead of silently falling back, since
    that mismatch likely means the user expected GPU acceleration they
    aren't getting."""
    extra_params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate device and fail fast if "cuda" is forced but unusable."""
        valid_devices = {"auto", "cuda", "cpu"}
        if self.device not in valid_devices:
            raise ConfigError(
                f"device must be one of {sorted(valid_devices)}, got {self.device!r}"
            )
        if self.device == "cuda" and not detect_cuda_available():
            raise ConfigError(
                "device='cuda' was explicitly requested but no usable CUDA"
                " GPU was detected. Use device='auto' to fall back to CPU"
                " automatically, or device='cpu' to run on CPU."
            )

    def to_dict(self) -> dict:
        """Return merged XGBoost params dict including extra_params.

        The "device" field is resolved to a literal "cuda"/"cpu" value (via
        resolve_xgboost_device) rather than passed through as "auto", since
        XGBoost itself doesn't understand "auto".
        """
        from .utils import resolve_xgboost_device

        d = asdict(self)
        d.pop("extra_params")
        d["device"] = resolve_xgboost_device(self.device)
        d.update(self.extra_params)
        return d


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap ensemble importance stability analysis."""

    n_bootstrap: int = 0
    sample_frac: float = 1.0
    shap_importance_stability_threshold: float | None = None


@dataclass
class Config:
    """Top-level pipeline configuration."""

    input_file: str
    target_column: str
    task: str
    output_file: str = "treeshaker_results.csv"
    output_format: str = "csv"
    leakage_screening: Literal["exact", "approximate"] = "exact"
    """Which algorithm to use for near-duplicate feature screening in leakage
    detection. "exact" (default) uses pairwise correlation (O(p^2) where p is
    the feature count), which is always accurate but slow on wide datasets
    (thousands of features). "approximate" uses SimHash-based locality-sensitive
    hashing via an optional Rust extension for near-linear-in-p screening,
    significantly faster on wide data but may miss some true near-duplicates that
    have weak hash collisions. Requires the optional Rust extension (install with
    pip install treeshaker[fast]). Falls back to "exact" with a warning if the
    extension is unavailable."""
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
    """Configuration for bootstrap ensemble importance stability analysis,
    including the optional stability threshold to gate recommendations on
    ranking stability when bootstrap is enabled (n_bootstrap > 0)."""
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
    shap_sample_size: int | None = None
    """Optional row cap for SHAP importance computation. When set and
    smaller than the number of available rows, a random sample of this
    many rows (drawn with random_seed) is used to compute SHAP values
    instead of the full dataset, which can substantially speed up SHAP on
    large datasets. Importance is still reported over the full feature
    set; only the row count used to estimate it is reduced. None
    (default) uses all rows, matching prior behavior -- no change unless
    explicitly set."""
    synthetic_config: SyntheticConfig = field(default_factory=SyntheticConfig)
    """Configuration for SHAP-interaction-driven synthetic feature
    suggestion (see synthetic_features.py). Off by default
    (synthetic_config.synthesize_from_interactions=False); when enabled,
    also requires compute_shap_interactions=True."""
    flag_leakage_importance_mismatch: bool = False
    """If True, features that pass the importance threshold (Keep) but have
    active leakage warnings are flagged with a distinct note in the reason
    column. This highlights the potentially dangerous case of features that
    are high-importance yet suspicious, which may indicate undetected leakage.
    Off by default since most users follow the harm/leakage pipeline
    normally; enable to surface this edge case explicitly."""
    compute_train_importance: bool = False
    """If True, compute SHAP importance on the combined train+val dataset
    in addition to the held-out test set, enabling detection of large
    train-vs-test SHAP gaps (a diagnostic signal for overfit/leakage).
    Off by default; when enabled, optionally paired with
    train_test_importance_delta_threshold to flag gaps exceeding a
    specified threshold. The test-only evaluation remains the authoritative
    importance signal for feature selection — this diagnostic does not
    change the recommendation logic."""
    train_test_importance_delta_threshold: float | None = None
    """When compute_train_importance is True, flag features whose
    train-vs-test SHAP importance delta (relative to test importance)
    exceeds this threshold. The delta is computed as
    abs(train_importance - test_importance) / test_importance, guarding
    against division by zero following the same convention as WAPE.
    None (default) disables the threshold check and reports deltas
    informationally only. Must be >= 0 if set."""
    compute_importance_divergence: bool = False
    """If True, compute a per-feature divergence score quantifying how much
    SHAP importance, permutation importance, and mutual information
    disagree about a feature's rank. 0 means all three signals agree on the
    feature's relative rank; 1 means maximum disagreement. Off by default;
    this is a diagnostic signal only and does not change the recommendation
    logic."""
    time_column: str | None = None
    """Optional column name for chronological train/val/test splitting.
    When set (not None), the data is sorted by this column ascending and
    split chronologically rather than randomly, preserving test_size and
    val_size as proportions. When None (default), splits are random and
    behavior is unchanged from prior versions. The time_column must exist
    in the input data and must contain sortable (numeric or datetime)
    values."""

    def __post_init__(self) -> None:
        """Validate configuration values."""
        valid_tasks = {"binary", "multiclass", "regression"}
        if self.task not in valid_tasks:
            raise ConfigError(f"task must be one of {valid_tasks}, got {self.task!r}")
        valid_screenings = {"exact", "approximate"}
        if self.leakage_screening not in valid_screenings:
            raise ConfigError(
                f"leakage_screening must be one of {valid_screenings},"
                f" got {self.leakage_screening!r}"
            )
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
        _require_at_least(
            "bootstrap_config.n_bootstrap", self.bootstrap_config.n_bootstrap, 0
        )
        if not 0 < self.bootstrap_config.sample_frac <= 1.0:
            raise ConfigError(
                "bootstrap_config.sample_frac must be in (0, 1], got"
                f" {self.bootstrap_config.sample_frac}"
            )
        if self.bootstrap_config.shap_importance_stability_threshold is not None:
            _require_greater_than(
                "bootstrap_config.shap_importance_stability_threshold",
                self.bootstrap_config.shap_importance_stability_threshold,
                0,
            )
        _require_greater_than(
            "residual_config.residual_corr_threshold",
            self.residual_config.residual_corr_threshold,
            0,
        )
        _require_at_least("n_permutations", self.n_permutations, 1)
        if self.shap_sample_size is not None:
            _require_at_least("shap_sample_size", self.shap_sample_size, 1)
        _require_at_least(
            "synthetic_config.mi_threshold", self.synthetic_config.mi_threshold, 0
        )
        _require_at_least(
            "synthetic_config.interaction_top_n",
            self.synthetic_config.interaction_top_n,
            1,
        )
        if self.train_test_importance_delta_threshold is not None:
            _require_at_least(
                "train_test_importance_delta_threshold",
                self.train_test_importance_delta_threshold,
                0,
            )
        if (
            self.permutation_metric is not None
            and self.permutation_metric not in VALID_PERMUTATION_METRICS
        ):
            raise ConfigError(
                "permutation_metric must be one of"
                f" {sorted(VALID_PERMUTATION_METRICS)}, got"
                f" {self.permutation_metric!r}"
            )

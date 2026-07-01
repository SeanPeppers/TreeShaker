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
    """Which algorithm to use for near-duplicate *numeric* feature screening in
    leakage detection. "exact" (default) uses pairwise correlation (O(p^2) where p
    is the feature count), which is always accurate but slow on wide datasets
    (thousands of features). "approximate" uses SimHash-based locality-sensitive
    hashing via an optional Rust extension for near-linear-in-p screening,
    significantly faster on wide data but may miss some true near-duplicates that
    have weak hash collisions. Requires the optional Rust extension (install with
    pip install treeshaker[fast]). Falls back to "exact" with a warning if the
    extension is unavailable.

    Scope: this setting only controls the numeric near-duplicate/correlation
    check (`LeakageDetector._check_near_duplicates*`); it has no effect on the
    numeric target-correlation check or on categorical proxy-leakage detection.
    A categorical feature that is a near-perfect proxy for the target (e.g. a
    post-outcome field) is caught separately and unconditionally by the
    bias-corrected Cramer's V association check (see
    `LeakageConfig.categorical_assoc_threshold`), which always runs regardless
    of `leakage_screening`. Switching between "exact" and "approximate" never
    changes whether a categorical column is flagged."""
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
    skip_leakage_check: bool = False
    """If True, skip all leakage and logical-error detection (the seven
    checks in LeakageDetector, including look-ahead leakage). Harm
    detection (SHAP/permutation importance) still runs."""
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
    use_boruta: bool = False
    """If True, run Boruta feature selection and add a boruta_decision
    column ('Confirmed', 'Tentative', or 'Rejected') to the results DataFrame.
    Off by default."""

    use_rfecv: bool = False
    """If True, run sklearn RFECV feature selection and add rfecv_ranking column
    to the results DataFrame. Off by default."""

    use_lightgbm: bool = False
    """If True, compute LightGBM model feature importances as lightgbm_importance
    column to cross-validate against XGBoost. Off by default."""

    use_catboost: bool = False
    """If True, compute CatBoost model feature importances as catboost_importance
    column with native categorical handling. Off by default."""

    use_rf_baseline: bool = False
    """If True, compute sklearn Random Forest baseline feature importances as
    rf_importance column for cross-checking. Off by default."""

    use_sklearn_permutation_importance: bool = False
    """If True, compute model-agnostic sklearn permutation importance as
    sklearn_permutation_importance column on the existing trained model.
    Off by default."""

    use_gpu: bool = False
    """If True, configure XGBoost models to use GPU acceleration (sets
    device to 'cuda'). Off by default."""

    use_optuna_pruning: bool = False
    """If True, enable Optuna trial pruning and early stopping during hyperparameter
    tuning to skip plateauing trials. Off by default."""

    large_dataset_sample_threshold: int | None = None
    """Optional row count threshold for large-dataset sampling. When set and
    input row count exceeds this threshold, auto-stratified sampling is applied
    before train/val/test split. None (default) disables sampling."""

    large_dataset_sample_frac: float = 0.5
    """Fraction of rows to sample when large_dataset_sample_threshold is exceeded.
    Default 0.5 (50%). Must be in (0, 1]."""

    use_mlflow: bool = False
    """If True, log this run to MLflow for experiment tracking: Config
    parameters actually used, per-feature importance/metric columns from
    the results DataFrame, and the final results table as a CSV artifact.
    Off by default. Requires the optional 'mlflow' package (install with
    pip install treeshaker[mlflow]); raises ImportError with install
    instructions if missing while this flag is True. Does not affect
    feature selection in any way -- logging only."""

    mlflow_experiment_name: str = "treeshaker"
    """MLflow experiment name to log runs under when use_mlflow=True.
    Ignored otherwise. Default "treeshaker"."""

    custom_model: object | None = None
    """Optional pre-instantiated sklearn-API-compatible estimator (anything
    with `.fit(X, y)` and `.predict(X)`; classifiers also need
    `.predict_proba(X)`) to use as the primary model instead of building an
    XGBClassifier/XGBRegressor. Lets TreeShaker drive arbitrary ML or NN
    models via standard sklearn wrapper adapters (e.g. skorch, scikeras, or
    a user's own wrapper) without TreeShaker itself depending on any NN
    framework.

    Python-API-only: this is a live Python object, not a CLI-flag-able
    value, so there is intentionally no `--custom-model` CLI flag and no
    MCP tool parameter for it. Only settable when constructing `Config`
    directly in Python and passing it to `FeatureSelectionPipeline`. None
    (default) preserves the existing XGBoost-only behavior unchanged.

    The pipeline clones this estimator (via `sklearn.base.clone` when
    possible, else `copy.deepcopy`) before fitting, so the object you pass
    in is never mutated. Required methods are validated at pipeline-run
    time (not here), raising ValueError listing any missing methods."""

    parallel_feature_evaluation: bool = False
    """If True, run the per-feature permutation-importance loop across
    worker processes (via joblib's loky backend) instead of a single
    sequential Python loop. A process backend, not threads, is required
    here: XGBoost's Booster.predict is not safe to call concurrently from
    multiple threads on the same fitted model (confirmed via a real-dataset
    crash on Windows). Speeds up evaluation on datasets with many features.
    Results are numerically identical to the sequential path (same
    per-feature random seeds), just computed out of order. Off by default
    -- existing single-threaded behavior is unchanged unless this is
    explicitly opted into."""

    n_jobs: int = -1
    """Number of worker processes when parallel_feature_evaluation=True,
    following joblib/sklearn convention (-1 = use all available cores).
    Ignored when parallel_feature_evaluation=False. Default -1."""

    streaming: bool = False
    """If True, opt into the out-of-core / streaming data path for datasets
    too large to comfortably hold in RAM as a single pandas DataFrame.

    Stage 1 (this field's initial behavior): the input CSV is read in
    chunks of `streaming_chunk_size` rows and numeric dtypes are downcast
    per chunk (int64 -> smallest signed int that fits, float64 -> float32)
    before the chunks are concatenated, lowering peak memory versus a single
    `pd.read_csv` of the whole file. The subsequent train/val/test split and
    every downstream step are otherwise unchanged.

    Stage 2 (out-of-core training): XGBoost training is routed through its
    external-memory `ExtMemQuantileDMatrix` + a custom `DataIter` that
    streams on-disk chunks of the train+val data, so the booster fit never
    holds more than one chunk plus the quantile sketch in memory. The fitted
    booster is wrapped in a thin sklearn-style adapter so every downstream
    step (SHAP, permutation importance, residuals) is unchanged.

    CPU-only constraint: the external-memory training path runs on CPU.
    XGBoost's GPU external memory requires GPU-resident (cupy/RAPIDS) chunks,
    which TreeShaker does not depend on. If a GPU device is requested
    (use_gpu / xgboost_config.device="cuda") together with streaming, the
    streaming booster falls back to CPU with a warning (the GPU is still used
    for the in-memory path when streaming is off).

    Later stages additionally switch the explainability/diagnostic steps
    (SHAP, permutation importance, mutual information, leakage, residuals)
    onto a bounded in-memory sample (see `shap_sample_size` /
    `large_dataset_sample_threshold`) rather than the full dataset.

    Off by default -- the default in-memory path for small/medium datasets
    is completely unchanged unless this is explicitly enabled."""

    streaming_chunk_size: int = 100_000
    """Number of CSV rows read per chunk when streaming=True. Larger chunks
    use more memory but fewer iterations; smaller chunks lower peak memory at
    the cost of more overhead. Ignored when streaming=False. Must be >= 1.
    Default 100_000."""

    use_polars: bool = False
    """If True, use Polars instead of pandas to read the input CSV. Polars
    parses CSV significantly faster than pandas (Rust-based, multi-threaded)
    and uses less peak memory during the parse. The resulting DataFrame is
    converted to pandas after loading so the rest of the pipeline is
    unchanged. Off by default; requires ``pip install treeshaker[polars]``.
    Compatible with ``streaming=True``."""
    n_stability_runs: int = 0
    """Number of additional pipeline runs (with different random seeds) used to
    estimate per-feature recommendation stability. 0 = disabled (default).

    When > 0, the pipeline is re-run `n_stability_runs` times after the main
    run, each with a different random seed (derived from `random_seed` unless
    `stability_seeds` is set explicitly). The per-run recommendations are
    collected and two columns are appended to the results DataFrame:
    - `stability_score`: fraction of runs (including the main run) that agreed
      with the final recommendation (0.0–1.0). 1.0 = unanimous across all seeds.
    - `stability_votes`: e.g. "Keep:4,Drop:1" — vote tally across all seeds.

    Use with `skip_tuning=True` to keep total runtime reasonable; each
    additional run adds roughly one main-pipeline-run's worth of time."""

    stability_seeds: list[int] | None = None
    """Explicit list of random seeds for the additional stability runs. Must
    have exactly `n_stability_runs` elements when set. When None (default),
    seeds are auto-derived as [random_seed + 1, random_seed + 2, ...] up to
    `n_stability_runs`. Ignored when n_stability_runs=0."""

    compute_confidence_scores: bool = False
    """If True, compute calibrated confidence scores for each feature's
    importance and recommendation. Off by default."""

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.use_gpu and not detect_cuda_available():
            raise ConfigError(
                "use_gpu=True was explicitly requested but no usable CUDA"
                " GPU was detected."
            )
        valid_tasks = {"binary", "multiclass", "regression"}
        if self.task not in valid_tasks:
            raise ConfigError(f"task must be one of {valid_tasks}, got {self.task!r}")
        if self.large_dataset_sample_threshold is not None:
            _require_at_least(
                "large_dataset_sample_threshold",
                self.large_dataset_sample_threshold,
                1,
            )
        if not 0 < self.large_dataset_sample_frac <= 1.0:
            raise ConfigError(
                "large_dataset_sample_frac must be in (0, 1], got"
                f" {self.large_dataset_sample_frac}"
            )
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
        _require_at_least("streaming_chunk_size", self.streaming_chunk_size, 1)
        _require_at_least("n_stability_runs", self.n_stability_runs, 0)
        if self.stability_seeds is not None:
            if self.n_stability_runs == 0:
                raise ConfigError(
                    "stability_seeds has no effect when n_stability_runs=0."
                )
            if len(self.stability_seeds) != self.n_stability_runs:
                raise ConfigError(
                    f"stability_seeds must have exactly n_stability_runs="
                    f"{self.n_stability_runs} elements, got"
                    f" {len(self.stability_seeds)}."
                )
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

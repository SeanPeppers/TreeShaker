"""Configuration dataclasses for TreeShaker."""

from dataclasses import asdict, dataclass, field

from .exceptions import ConfigError
from .leakage_detection import LeakageConfig


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
class Config:
    """Top-level pipeline configuration."""

    input_file: str
    target_column: str
    task: str
    output_file: str = "treeshaker_results.csv"
    output_format: str = "csv"
    test_size: float = 0.2
    random_seed: int = 42
    optuna_config: OptunaConfig = field(default_factory=OptunaConfig)
    xgboost_config: XGBoostConfig = field(default_factory=XGBoostConfig)
    shap_importance_threshold: float = 0.01
    permutation_importance_threshold: float = 0.05
    leakage_config: LeakageConfig = field(default_factory=LeakageConfig)
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        valid_tasks = {"binary", "multiclass", "regression"}
        if self.task not in valid_tasks:
            raise ConfigError(f"task must be one of {valid_tasks}, got {self.task!r}")
        if not 0 < self.test_size < 1:
            raise ConfigError(f"test_size must be in (0, 1), got {self.test_size}")
        valid_formats = {"csv", "html"}
        if self.output_format not in valid_formats:
            raise ConfigError(
                f"output_format must be one of {valid_formats}, "
                f"got {self.output_format!r}"
            )

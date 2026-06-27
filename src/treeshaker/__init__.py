"""TreeShaker: Automated feature selection using XGBoost, SHAP, and Optuna."""

from .config import BootstrapConfig, Config, OptunaConfig, XGBoostConfig
from .exceptions import ConfigError, TreeShakerError, ValidationError
from .leakage_detection import LeakageConfig, LeakageDetector
from .pipeline import FeatureSelectionPipeline

__version__ = "0.1.0"

__all__ = [
    "FeatureSelectionPipeline",
    "Config",
    "OptunaConfig",
    "XGBoostConfig",
    "BootstrapConfig",
    "LeakageConfig",
    "LeakageDetector",
    "TreeShakerError",
    "ValidationError",
    "ConfigError",
]

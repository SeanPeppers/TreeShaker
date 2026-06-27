"""Exception classes for TreeShaker."""


class TreeShakerError(Exception):
    """Base exception for TreeShaker errors."""

    pass


class ValidationError(TreeShakerError):
    """Raised when input data or target column is invalid."""

    pass


class ConfigError(TreeShakerError):
    """Raised when configuration values are invalid."""

    pass

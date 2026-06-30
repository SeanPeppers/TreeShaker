"""Out-of-core XGBoost training via external memory (streaming Stage 2).

This module implements the actual out-of-core *training* path for
``Config.streaming=True``. XGBoost's scikit-learn API
(``XGBClassifier``/``XGBRegressor``) materializes the full feature matrix in
RAM when you call ``.fit(X, y)``; it has no external-memory entry point. The
only out-of-core training path XGBoost offers is the *native* API: build an
``ExtMemQuantileDMatrix`` (or ``DMatrix``) from a custom
``xgboost.DataIter`` that yields on-disk chunks, then call ``xgboost.train``.
The result is a native ``Booster`` rather than a scikit-learn estimator.

To keep every downstream consumer (permutation importance, residual
diagnostics, etc., which all call ``.predict``/``.predict_proba``) working
unchanged, the trained ``Booster`` is wrapped in :class:`BoosterAdapter`,
which exposes the small scikit-learn-style surface those callers rely on and
a ``get_booster()`` shim so ``shap.TreeExplainer`` keeps the fast tree path.

Why write chunks to disk: the in-memory split DataFrames would defeat the
purpose if we fed them to the iterator whole. Instead the train+val rows are
spilled once to a sequence of on-disk pickled chunk files; the iterator then
streams them back one at a time, so booster construction never holds more
than a single chunk plus XGBoost's (bounded) quantile sketch in memory. This
is the honest out-of-core property -- peak training memory is O(chunk), not
O(dataset).
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _spill_chunks(
    X: pd.DataFrame, y: pd.Series, chunk_size: int, cache_dir: str
) -> list[str]:
    """Write (X, y) to a sequence of on-disk pickled chunk files.

    Each chunk holds at most ``chunk_size`` rows. The label is stored as a
    reserved ``__label__`` column so a single file round-trips both features
    and target. Pickle is used (not Parquet/CSV) because it preserves the
    prepared pandas dtypes exactly -- in particular the ``category`` dtype
    XGBoost consumes natively -- without requiring an optional engine
    (pyarrow/fastparquet) to be installed. Returns the ordered chunk paths.
    """
    n = len(X)
    if n == 0:
        raise ValueError("Cannot train on an empty dataset")
    paths: list[str] = []
    frame = X.copy()
    # Reserved label column name kept deliberately unlikely to collide with a
    # real feature; validated by the caller via the feature_names list.
    frame["__label__"] = np.asarray(y)
    for start in range(0, n, chunk_size):
        chunk = frame.iloc[start : start + chunk_size]
        path = str(Path(cache_dir) / f"chunk_{start // chunk_size:06d}.pkl")
        chunk.to_pickle(path)
        paths.append(path)
    logger.info(
        "Spilled %d rows to %d on-disk chunk(s) for streaming train",
        n,
        len(paths),
    )
    return paths


class _ChunkDataIter:
    """xgboost.DataIter over on-disk pickled chunks.

    Subclasses ``xgboost.DataIter`` at construction time (the base class is
    imported lazily so importing this module never requires xgboost). Each
    ``next`` call loads exactly one chunk file, splits off the reserved
    ``__label__`` column, and hands (X, y) to XGBoost via the ``input_data``
    callback. Only one chunk is resident at a time.
    """

    def __init__(
        self,
        chunk_paths: list[str],
        feature_names: list[str],
        cache_prefix: str,
    ):
        self._chunk_paths = chunk_paths
        self._feature_names = feature_names
        self._it = 0
        # Initialize the xgboost.DataIter base (set dynamically in factory).

    def reset(self) -> None:
        """Reset iteration to the first chunk (called by XGBoost per pass)."""
        self._it = 0

    def next(self, input_data: Callable) -> bool:
        """Load the next chunk and feed it to XGBoost. Return False when done."""
        if self._it == len(self._chunk_paths):
            return False
        # noqa justification: these pickle files are written by _spill_chunks
        # in this same process into a private temp dir -- never untrusted
        # external input -- so unpickling them is safe.
        df = pd.read_pickle(self._chunk_paths[self._it])  # noqa: S301
        y = df["__label__"].to_numpy()
        X = df[self._feature_names]
        input_data(data=X, label=y)
        self._it += 1
        return True


def _make_iterator(
    chunk_paths: list[str], feature_names: list[str], cache_prefix: str
) -> Any:
    """Build a concrete xgboost.DataIter subclass instance over the chunks.

    xgboost.DataIter is imported here (not at module import) so this module
    stays importable without xgboost installed; the returned object is a
    genuine DataIter so ``ExtMemQuantileDMatrix`` accepts it.
    """
    import xgboost as xgb

    class _Iter(_ChunkDataIter, xgb.DataIter):
        def __init__(self, paths: list[str], feats: list[str], prefix: str):
            _ChunkDataIter.__init__(self, paths, feats, prefix)
            # cache_prefix lets XGBoost cache the binarized pages on disk too.
            xgb.DataIter.__init__(self, cache_prefix=prefix)

    return _Iter(chunk_paths, feature_names, cache_prefix)


class BoosterAdapter:
    """scikit-learn-style adapter around a natively-trained xgboost.Booster.

    Exposes just the surface TreeShaker's downstream steps use:
    ``predict`` / ``predict_proba`` (DataFrame in, ndarray out) and
    ``get_booster`` (so ``shap.TreeExplainer`` takes the fast tree path).
    This lets the out-of-core training path return something the rest of the
    pipeline treats exactly like the usual XGBClassifier/XGBRegressor.

    Args:
        booster: The trained native booster.
        task: One of "binary", "multiclass", "regression" -- governs the
            shape/meaning of predict/predict_proba outputs.
        feature_names: Ordered feature columns, used to build the prediction
            DMatrix consistently with training.
        n_classes: Number of classes for multiclass (ignored otherwise).
    """

    def __init__(
        self,
        booster: Any,
        task: str,
        feature_names: list[str],
        n_classes: int = 0,
    ):
        self._booster = booster
        self._task = task
        self._feature_names = feature_names
        self._n_classes = n_classes
        # Mirror sklearn's attribute so callers that introspect it still work.
        self.feature_names_in_ = np.asarray(feature_names)

    def get_booster(self) -> Any:
        """Return the underlying booster (SHAP TreeExplainer fast path)."""
        return self._booster

    def _dmatrix(self, X: pd.DataFrame) -> Any:
        import xgboost as xgb

        return xgb.DMatrix(
            X[self._feature_names],
            enable_categorical=True,
            feature_names=self._feature_names,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities (classification only).

        For binary, returns a (n, 2) array [P(0), P(1)] to match
        XGBClassifier.predict_proba so existing positive-class slicing
        (``[:, 1]``) is unchanged.
        """
        raw = np.asarray(self._booster.predict(self._dmatrix(X)))
        if self._task == "binary":
            return np.column_stack([1.0 - raw, raw])
        # multiclass: booster already returns (n, n_classes) softprob.
        return raw

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard labels (classification) or raw values (regression)."""
        raw = np.asarray(self._booster.predict(self._dmatrix(X)))
        if self._task == "binary":
            return (raw >= 0.5).astype(int)
        if self._task == "multiclass":
            return raw.argmax(axis=1)
        return raw


def train_booster_streaming(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    params: dict,
    feature_names: list[str],
    chunk_size: int,
    num_boost_round: int = 100,
) -> BoosterAdapter:
    """Train an XGBoost booster out-of-core over on-disk chunks of (X, y).

    Spills the rows to pickled chunks, streams them through an
    ``ExtMemQuantileDMatrix`` (falling back to ``DMatrix`` if the external
    quantile variant is unavailable in the installed xgboost), trains via the
    native ``xgboost.train``, and wraps the result in :class:`BoosterAdapter`.

    Args:
        X: Feature matrix (in-memory here, but only spilled chunks are ever
            held during the fit itself).
        y: Target series aligned with X.
        task: "binary", "multiclass", or "regression".
        params: XGBoost params (already device/objective-resolved by the
            caller via _build_xgb_fit_params + objective selection below).
        feature_names: Ordered feature columns.
        chunk_size: Rows per on-disk chunk.
        num_boost_round: Boosting rounds for xgboost.train.

    Returns:
        A BoosterAdapter wrapping the trained booster.
    """
    import xgboost as xgb

    train_params = dict(params)
    # tree_method="hist" is required for the external-memory quantile path.
    train_params.setdefault("tree_method", "hist")
    # The external-memory iterator feeds CPU (numpy/pandas) data. GPU external
    # memory would require the iterator to yield cupy arrays (RAPIDS/cuDF),
    # which TreeShaker intentionally does not depend on. So out-of-core
    # streaming training always runs on CPU; if a GPU device was resolved
    # upstream (use_gpu / device="cuda"), fall back to CPU here with a clear
    # warning rather than crashing on the CPU-data/GPU-DMatrix mismatch.
    if str(train_params.get("device", "cpu")).startswith("cuda"):
        logger.warning(
            "streaming=True forces CPU training: XGBoost external memory needs"
            " GPU-resident (cupy/RAPIDS) chunks, which TreeShaker does not"
            " depend on. Ignoring the requested CUDA device for the streaming"
            " booster. Disable streaming to use the GPU on an in-memory fit."
        )
    train_params["device"] = "cpu"
    n_classes = 0
    if task == "binary":
        train_params.setdefault("objective", "binary:logistic")
    elif task == "multiclass":
        n_classes = int(pd.Series(y).nunique())
        train_params.setdefault("objective", "multi:softprob")
        train_params.setdefault("num_class", n_classes)

    # ignore_cleanup_errors: on Windows, XGBoost keeps the external-memory
    # cache page files mmap'd for the booster's lifetime, so they can still be
    # open when this context exits -- tolerate the resulting unlink failures
    # rather than crashing after a successful train. The OS temp dir is
    # reclaimed eventually regardless.
    with tempfile.TemporaryDirectory(
        prefix="treeshaker_stream_", ignore_cleanup_errors=True
    ) as cache_dir:
        chunk_paths = _spill_chunks(X, y, chunk_size, cache_dir)
        cache_prefix = str(Path(cache_dir) / "xgb_cache")
        it = _make_iterator(chunk_paths, feature_names, cache_prefix)
        dmatrix: Any
        try:
            dmatrix = xgb.ExtMemQuantileDMatrix(it, enable_categorical=True)
        except (AttributeError, TypeError) as exc:
            # Older xgboost without ExtMemQuantileDMatrix: fall back to the
            # generic external-memory DMatrix, which is also iterator-backed.
            logger.warning(
                "ExtMemQuantileDMatrix unavailable (%s); falling back to"
                " external-memory DMatrix.",
                exc,
            )
            dmatrix = xgb.DMatrix(it, enable_categorical=True)
        booster = xgb.train(train_params, dmatrix, num_boost_round=num_boost_round)

    logger.info(
        "Out-of-core booster trained over %d on-disk chunk(s)", len(chunk_paths)
    )
    return BoosterAdapter(booster, task, feature_names, n_classes=n_classes)

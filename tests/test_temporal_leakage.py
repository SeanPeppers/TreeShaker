"""Tests for extended temporal leakage detection."""

import logging

import numpy as np
import pandas as pd

from treeshaker.leakage_detection import LeakageConfig, LeakageDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detector() -> LeakageDetector:
    return LeakageDetector(LeakageConfig())


# ---------------------------------------------------------------------------
# 1. Unsorted time column warning
# ---------------------------------------------------------------------------


def test_unsorted_time_column_emits_warning(tmp_path, caplog):
    n = 60
    rng = np.random.default_rng(42)
    t = np.arange(n, dtype=float)
    y = rng.standard_normal(n)
    df = pd.DataFrame({"t": t, "y": y, "feat": rng.standard_normal(n)})
    # shuffle so time is unsorted
    df = df.sample(frac=1, random_state=0)
    csv_path = tmp_path / "unsorted.csv"
    df.to_csv(csv_path, index=False)

    from treeshaker.utils import load_csv

    # caplog without logger= captures root logger (logging.warning goes there)
    with caplog.at_level(logging.WARNING):
        load_csv(
            str(csv_path),
            target_column="y",
            task="regression",
            time_column="t",
        )

    assert any("monotonically sorted" in msg for msg in caplog.messages)


def test_sorted_time_column_no_warning(tmp_path, caplog):
    n = 60
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "t": np.arange(n, dtype=float),
            "y": rng.standard_normal(n),
            "feat": rng.standard_normal(n),
        }
    )
    csv_path = tmp_path / "sorted.csv"
    df.to_csv(csv_path, index=False)

    from treeshaker.utils import load_csv

    with caplog.at_level(logging.WARNING):
        load_csv(
            str(csv_path),
            target_column="y",
            task="regression",
            time_column="t",
        )

    assert not any("monotonically sorted" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# 2. Random split detection
# ---------------------------------------------------------------------------


def test_random_split_on_timeseries_flagged():
    """Explicitly overlapping time ranges → overlap warning on time_column key."""
    # Construct splits where time ranges definitely overlap
    t = np.arange(120, dtype=float)
    rng = np.random.default_rng(0)
    y = rng.standard_normal(120)
    # train: t=0..79, val: t=40..59 (overlaps train), test: t=10..29 (overlaps both)
    train = pd.DataFrame({"t": t[:80], "y": y[:80]})
    val = pd.DataFrame({"t": t[40:60], "y": y[40:60]})
    test = pd.DataFrame({"t": t[10:30], "y": y[10:30]})

    detector = _make_detector()
    warnings = detector.detect_lookahead_leakage(
        train, val, test, time_column="t", feature_names=["t", "y"]
    )
    assert any(
        "random split" in w.lower() or "overlap" in w.lower() for w in warnings["t"]
    )


def test_chronological_split_no_random_split_warning():
    """Proper chronological split → no overlap warning."""
    t = np.arange(120, dtype=float)
    rng = np.random.default_rng(0)
    y = rng.standard_normal(120)
    train = pd.DataFrame({"t": t[:80], "y": y[:80]})
    val = pd.DataFrame({"t": t[80:100], "y": y[80:100]})
    test = pd.DataFrame({"t": t[100:], "y": y[100:]})

    detector = _make_detector()
    warnings = detector.detect_lookahead_leakage(
        train, val, test, time_column="t", feature_names=["t", "y"]
    )
    assert not any(
        "random split" in w.lower() or "overlap" in w.lower() for w in warnings["t"]
    )


# ---------------------------------------------------------------------------
# 3. Future-peeking rolling window
# ---------------------------------------------------------------------------


def _make_rolling_df(window: int, shift: int, n: int = 100) -> pd.DataFrame:
    """Build a DataFrame with a rolling_mean_{window} feature using given shift."""
    rng = np.random.default_rng(7)
    s = pd.Series(rng.standard_normal(n), name="x")
    col_name = f"rolling_mean_{window}"
    rolled = s.rolling(window).mean().shift(shift)
    return pd.DataFrame({"x": s, col_name: rolled})


def test_future_peeking_rolling_window_detected():
    """shift=0 on rolling_mean_5 → offset-1 leading NaNs → future-peeking warning."""
    window = 5
    df = _make_rolling_df(window=window, shift=0)
    col = f"rolling_mean_{window}"
    detector = _make_detector()
    warnings = detector.detect_temporal_structure(df, [col])
    assert any("future-peeking" in w.lower() for w in warnings[col])


def test_correct_rolling_window_no_peeking_warning():
    """shift=1 on rolling_mean_5 → correct offset NaNs → no future-peeking warning."""
    window = 5
    df = _make_rolling_df(window=window, shift=1)
    col = f"rolling_mean_{window}"
    detector = _make_detector()
    warnings = detector.detect_temporal_structure(df, [col])
    assert not any("future-peeking" in w.lower() for w in warnings[col])


def test_generic_mismatch_still_fires_for_non_rolling():
    """lag_5 feature with wrong NaN count gets generic mismatch (not future-peeking)."""
    rng = np.random.default_rng(3)
    s = pd.Series(rng.standard_normal(100))
    lagged = s.shift(3)  # only 3 NaNs but name says 5
    df = pd.DataFrame({"lag_5": lagged})
    detector = _make_detector()
    warnings = detector.detect_temporal_structure(df, ["lag_5"])
    msgs = warnings["lag_5"]
    assert any("mismatch" in w.lower() or "off-by-one" in w.lower() for w in msgs)
    assert not any("future-peeking" in w.lower() for w in msgs)

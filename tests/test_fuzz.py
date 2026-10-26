"""Property-based (fuzz) tests, also picked up by Scorecard's Fuzzing check."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from treeshaker.exceptions import ValidationError
from treeshaker.utils import compute_wape, derive_output_stem

finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)


values = st.lists(finite, min_size=1, max_size=50)


@given(values, values)
def test_wape_non_negative_or_validation_error(y_true, y_pred):
    n = min(len(y_true), len(y_pred))
    try:
        assert compute_wape(np.array(y_true[:n]), np.array(y_pred[:n])) >= 0
    except ValidationError:
        assert sum(abs(v) for v in y_true[:n]) == 0


@given(st.text())
def test_output_stem_never_longer_than_input(path):
    assert len(derive_output_stem(path)) <= len(path)

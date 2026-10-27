"""Atheris target for ClusterFuzzLite: feeds arbitrary bytes to the pure helpers."""

import sys

import atheris

with atheris.instrument_imports():
    import numpy as np

    from treeshaker.exceptions import ValidationError
    from treeshaker.utils import compute_wape, derive_output_stem


def test_one_input(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    derive_output_stem(fdp.ConsumeUnicodeNoSurrogates(64))
    n = fdp.ConsumeIntInRange(1, 20)
    y_true = np.array(fdp.ConsumeRegularFloatList(n))
    y_pred = np.array(fdp.ConsumeRegularFloatList(n))
    if len(y_true) != len(y_pred):
        return
    try:
        compute_wape(y_true, y_pred)
    except ValidationError:
        pass


if __name__ == "__main__":
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()

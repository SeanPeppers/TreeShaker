"""Tests for the treeshaker-text CLI (cli_text.py)."""

from unittest.mock import patch

import pandas as pd
import pytest

from treeshaker.cli_text import _load_dotted_path, _load_texts, main, parse_arguments


def _cli_argv(
    *extra: str, texts_file: str, predict_fn: str = "text_cli_fixtures:predict"
):
    return [
        "treeshaker-text",
        "--predict-fn",
        predict_fn,
        "--texts-file",
        texts_file,
        *extra,
    ]


def test_load_dotted_path_resolves_function():
    fn = _load_dotted_path("text_cli_fixtures:predict")
    assert callable(fn)


def test_load_dotted_path_missing_colon_raises():
    with pytest.raises(ValueError, match="module:attribute"):
        _load_dotted_path("text_cli_fixtures.predict")


def test_load_dotted_path_missing_attribute_raises():
    with pytest.raises(ValueError, match="no attribute"):
        _load_dotted_path("text_cli_fixtures:does_not_exist")


def test_load_texts_from_txt_file(tmp_path):
    path = tmp_path / "texts.txt"
    path.write_text("first line\nsecond line\n\nthird line\n")
    texts = _load_texts(str(path), text_column=None)
    assert texts == ["first line", "second line", "third line"]


def test_load_texts_from_csv_requires_text_column(tmp_path):
    path = tmp_path / "texts.csv"
    pd.DataFrame({"text": ["a", "b"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="--text-column"):
        _load_texts(str(path), text_column=None)


def test_load_texts_from_csv_with_column(tmp_path):
    path = tmp_path / "texts.csv"
    pd.DataFrame({"review": ["good movie", "bad movie"]}).to_csv(path, index=False)
    texts = _load_texts(str(path), text_column="review")
    assert texts == ["good movie", "bad movie"]


def test_parse_arguments_required(tmp_path):
    texts_file = str(tmp_path / "texts.txt")
    with patch("sys.argv", _cli_argv(texts_file=texts_file)):
        args = parse_arguments()
    assert args.predict_fn == "text_cli_fixtures:predict"
    assert args.texts_file == texts_file
    assert args.top_n == 20
    assert args.output == "text_results.csv"


def test_main_end_to_end(tmp_path):
    texts_file = tmp_path / "texts.txt"
    texts_file.write_text(
        "this movie is good\nthis movie is bad\na good story overall\n"
    )
    output = tmp_path / "results.csv"
    with patch(
        "sys.argv",
        _cli_argv("--output", str(output), "--top-n", "5", texts_file=str(texts_file)),
    ):
        result = main()
    assert result == 0
    df = pd.read_csv(output)
    assert "token" in df.columns
    assert df.iloc[0]["token"] == "good"


def test_main_bad_predict_fn_path_returns_error(tmp_path):
    texts_file = tmp_path / "texts.txt"
    texts_file.write_text("some text\n")
    with patch(
        "sys.argv",
        _cli_argv(
            texts_file=str(texts_file),
            predict_fn="not_a_real_module:predict",
        ),
    ):
        result = main()
    assert result == 1


def test_main_missing_texts_file_returns_error(tmp_path):
    with patch("sys.argv", _cli_argv(texts_file=str(tmp_path / "does_not_exist.txt"))):
        result = main()
    assert result == 1

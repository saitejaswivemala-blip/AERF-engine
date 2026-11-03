"""Unit tests for ingestion/text_cleaner.py."""

import pytest

from ingestion.text_cleaner import clean_text
from utils.exceptions import TextCleaningError


def test_clean_text_dehyphenates_line_break_word():
    raw = "a significant reduction in cardio-\nvascular events"
    cleaned = clean_text(raw)
    assert "cardiovascular" in cleaned
    assert "cardio-" not in cleaned


def test_clean_text_collapses_whitespace():
    raw = "Line one.\nLine   two.\n\n\nLine three."
    cleaned = clean_text(raw)
    assert "  " not in cleaned  # no double spaces
    assert "\n" not in cleaned


def test_clean_text_handles_empty_string():
    assert clean_text("") == ""


def test_clean_text_handles_none():
    assert clean_text(None) == ""


def test_clean_text_handles_whitespace_only_string():
    assert clean_text("   \n\n   ") == ""


def test_clean_text_dehyphenate_can_be_disabled():
    raw = "cardio-\nvascular"
    cleaned = clean_text(raw, dehyphenate=False)
    assert "cardio- vascular" == cleaned


def test_clean_text_raises_on_non_string_input():
    with pytest.raises(TextCleaningError):
        clean_text(12345)  # type: ignore[arg-type]


def test_clean_text_strips_leading_trailing_whitespace():
    assert clean_text("   hello world   ") == "hello world"

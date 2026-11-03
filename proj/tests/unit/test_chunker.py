"""Unit tests for ingestion/chunker.py."""

from ingestion.chunker import chunk_page
from ingestion.pdf_loader import PageText
from utils.exceptions import ChunkingError

import pytest


def _page(page_number: int = 1) -> PageText:
    # chunk_page only reads page.page_number; text is unused (cleaned_text
    # is passed separately), so a placeholder raw text is fine here.
    return PageText(page_number=page_number, text="placeholder")


def test_chunk_page_splits_multiple_sentences():
    cleaned = "Aspirin reduces cardiovascular risk significantly. Drug B has no measurable effect. Further studies are recommended by researchers."
    chunks = chunk_page(_page(), cleaned)
    assert len(chunks) == 3


def test_chunk_page_char_offsets_reconstruct_original_text():
    cleaned = "Aspirin reduces cardiovascular risk significantly. Drug B has no measurable effect."
    chunks = chunk_page(_page(), cleaned)
    for chunk in chunks:
        assert cleaned[chunk.char_start:chunk.char_end].strip() == chunk.text


def test_chunk_page_preserves_page_number():
    cleaned = "Aspirin reduces cardiovascular risk significantly. Drug B has no measurable effect."
    chunks = chunk_page(_page(page_number=7), cleaned)
    assert all(c.page_number == 7 for c in chunks)


def test_chunk_page_does_not_split_on_abbreviation():
    cleaned = "This finding was reported by Smith et al. in a large clinical trial. The sample size was adequate for this type of study."
    chunks = chunk_page(_page(), cleaned)
    assert len(chunks) == 2
    assert "et al." in chunks[0].text


def test_chunk_page_empty_text_returns_empty_list():
    assert chunk_page(_page(), "") == []


def test_chunk_page_raises_on_non_string_input():
    with pytest.raises(ChunkingError):
        chunk_page(_page(), 12345)  # type: ignore[arg-type]


def test_chunk_page_merges_fragments_shorter_than_min_length():
    # "Hi." (3 chars) is well below the default min_chunk_length (15),
    # so it should be merged into the following sentence rather than
    # kept as its own tiny chunk.
    cleaned = "Hi. This is a longer sentence that easily exceeds the minimum chunk length on its own."
    chunks = chunk_page(_page(), cleaned)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Hi.")


def test_chunk_page_force_splits_text_longer_than_max_length():
    # A single "sentence" with no punctuation at all, long enough to
    # exceed the default max_chunk_length (1000), must still be split
    # into more than one chunk (guards against reference lists / tables
    # producing one giant unsplittable chunk).
    long_text = ("word " * 400).strip()  # ~2000 chars, no sentence punctuation
    chunks = chunk_page(_page(), long_text)
    assert len(chunks) > 1
    assert all(len(c.text) <= 1000 for c in chunks)

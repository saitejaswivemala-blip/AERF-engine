"""
chunker.py

Standard ingestion component. Splits cleaned page text into sentence-level
chunks, each carrying its character offsets *within the cleaned page text*
(see text_cleaner.py's design note on why offsets are relative to cleaned,
not raw, text).

[IMPLEMENTATION DECISION]: sentence segmentation here is a deliberately
simple, dependency-free regex/heuristic splitter -- not a trained model.
The design doc keeps `nlp/` (Phase 2+, model-based) strictly separate from
`ingestion/` (Phase 1, no models), so this splitter intentionally does not
use spaCy or any other NLP library. It handles common abbreviations
(e.g., "et al.", "Fig.", "e.g.") to reduce false sentence breaks, but it
will still make mistakes on unusual scientific notation (e.g. decimal
numbers, unusual abbreviations) -- this limitation is expected and is
noted in the Phase 1 README/tests rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.config_loader import get_settings
from ingestion.pdf_loader import PageText
from utils.exceptions import ChunkingError
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Common scientific-writing abbreviations that should NOT be treated as
# sentence boundaries even though they end in a period.
_ABBREVIATIONS = {
    "e.g.", "i.e.", "et al.", "etc.", "fig.", "figs.", "eq.", "eqs.",
    "vs.", "cf.", "approx.", "no.", "nos.", "vol.", "vols.", "pp.", "p.",
    "dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "ph.d.", "resp.",
}


@dataclass(frozen=True)
class RawChunk:
    """A single sentence-level chunk before it is persisted (no IDs yet)."""
    text: str
    page_number: int
    char_start: int  # offset into the cleaned page text this chunk came from
    char_end: int


def _find_sentence_spans(text: str) -> list[tuple[int, int]]:
    """
    Return a list of (start, end) character spans for each sentence in
    `text`, using a punctuation + capitalization heuristic with an
    abbreviation guard. `text` is assumed already cleaned (single-spaced).
    """
    if not text:
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    n = len(text)
    i = 0

    while i < n:
        ch = text[i]
        if ch in ".!?":
            next_idx = i + 1
            boundary_candidate = next_idx >= n or text[next_idx] == " "
            if boundary_candidate:
                # Extract the word ending at this punctuation to check for abbreviations
                word_start = i
                while word_start > start and text[word_start - 1] != " ":
                    word_start -= 1
                word = text[word_start:i + 1].lower()

                is_abbreviation = word in _ABBREVIATIONS

                # Look ahead past the space(s) to the next non-space char
                j = next_idx
                while j < n and text[j] == " ":
                    j += 1
                next_starts_sentence = j >= n or text[j].isupper() or text[j].isdigit()

                if not is_abbreviation and next_starts_sentence:
                    end = i + 1
                    spans.append((start, end))
                    start = j
                    i = j
                    continue
        i += 1

    if start < n:
        spans.append((start, n))

    return spans


def _enforce_length_bounds(
    spans: list[tuple[int, int]],
    text: str,
    min_length: int,
    max_length: int,
) -> list[tuple[int, int]]:
    """
    Merge sentence spans shorter than min_length into the following span
    (or the preceding one if it's the last span), and force-split spans
    longer than max_length at the nearest space boundary. This guards
    against degenerate inputs -- e.g. a mis-split abbreviation producing
    a 2-character chunk, or a reference list / table with no punctuation
    producing one huge chunk.
    """
    # --- merge short spans ---
    merged: list[tuple[int, int]] = []
    i = 0
    while i < len(spans):
        s, e = spans[i]
        if (e - s) < min_length and i + 1 < len(spans):
            next_s, next_e = spans[i + 1]
            merged.append((s, next_e))
            i += 2
        elif (e - s) < min_length and merged:
            prev_s, _ = merged.pop()
            merged.append((prev_s, e))
            i += 1
        else:
            merged.append((s, e))
            i += 1

    # --- force-split long spans ---
    bounded: list[tuple[int, int]] = []
    for s, e in merged:
        span_len = e - s
        if span_len <= max_length:
            bounded.append((s, e))
            continue
        cursor = s
        while (e - cursor) > max_length:
            split_at = text.rfind(" ", cursor, cursor + max_length)
            if split_at == -1 or split_at <= cursor:
                split_at = cursor + max_length  # no space found; hard split
            bounded.append((cursor, split_at))
            cursor = split_at + 1 if split_at < e and text[split_at] == " " else split_at
        if cursor < e:
            bounded.append((cursor, e))

    return bounded


def chunk_page(page: PageText, cleaned_text: str) -> list[RawChunk]:
    """
    Segment one page's cleaned text into sentence-level RawChunks.

    Args:
        page: the original PageText (used only for page_number).
        cleaned_text: the output of text_cleaner.clean_text() for this page.

    Returns:
        List of RawChunk, in reading order, with char_start/char_end
        offsets into `cleaned_text`.

    Raises:
        ChunkingError: if cleaned_text is not a string.
    """
    if not isinstance(cleaned_text, str):
        raise ChunkingError(f"Expected str, got {type(cleaned_text).__name__}")

    if cleaned_text.strip() == "":
        return []

    settings = get_settings()
    prep_cfg = settings.get("preprocessing", {})
    min_length = prep_cfg.get("min_chunk_length", 15)
    max_length = prep_cfg.get("max_chunk_length", 1000)

    spans = _find_sentence_spans(cleaned_text)
    spans = _enforce_length_bounds(spans, cleaned_text, min_length, max_length)

    chunks = [
        RawChunk(
            text=cleaned_text[s:e].strip(),
            page_number=page.page_number,
            char_start=s,
            char_end=e,
        )
        for s, e in spans
        if cleaned_text[s:e].strip()  # drop any whitespace-only artifacts
    ]

    logger.debug("Page %d -> %d chunk(s)", page.page_number, len(chunks))
    return chunks

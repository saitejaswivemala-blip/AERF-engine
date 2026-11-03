"""
text_cleaner.py

Standard ingestion component. Normalizes raw per-page PDF text into a
single clean string per page, which the chunker then segments.

Deliberately does NOT do any NLP (no tokenization, no NER) -- this is
pure string normalization. What it does:
  1. De-hyphenates words split across a line break (e.g. "cardio-\\nvascular"
     -> "cardiovascular"), configurable via settings.yaml.
  2. Strips non-printable/control characters that PDF extraction sometimes
     introduces (ligature artifacts, form-feed characters, etc.).
  3. Collapses all whitespace (including newlines) to single spaces.

[DESIGN NOTE / IMPLEMENTATION DECISION]: step 3 means paragraph
boundaries are not preserved after cleaning -- Phase 1 chunks at
sentence granularity, not paragraph granularity, so this tradeoff is
acceptable here and is documented rather than silently made. All
provenance character offsets produced by the chunker are offsets into
this *cleaned* string, not the raw PDF text, since the raw text's own
line breaks are not semantically meaningful (they are PDF layout
artifacts, not sentence/paragraph structure).
"""

from __future__ import annotations

import re
import unicodedata

from config.config_loader import get_settings
from utils.exceptions import TextCleaningError
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Word-\nword -> Wordword  (hyphen immediately followed by a line break,
# between two word characters -- conservative pattern so we don't merge
# real hyphenated compounds like "well-known" that happen to wrap differently)
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")

# Any run of whitespace (space, tab, newline, etc.) -> single space
_WHITESPACE_RE = re.compile(r"\s+")

# Control characters (category Cc) other than the ones we've already
# handled via whitespace collapsing
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(raw_text: str, dehyphenate: bool | None = None) -> str:
    """
    Normalize a raw text string (typically one PDF page) into clean text.

    Args:
        raw_text: text as extracted by pdf_loader.
        dehyphenate: override for settings.yaml's
            preprocessing.dehyphenate_line_breaks. If None, reads from config.

    Returns:
        Cleaned, single-spaced text. Returns "" for None/empty input rather
        than raising, since empty pages are valid (see pdf_loader).

    Raises:
        TextCleaningError: if raw_text is not a string.
    """
    if raw_text is None:
        return ""
    if not isinstance(raw_text, str):
        raise TextCleaningError(f"Expected str, got {type(raw_text).__name__}")
    if raw_text.strip() == "":
        return ""

    if dehyphenate is None:
        settings = get_settings()
        dehyphenate = settings.get("preprocessing", {}).get(
            "dehyphenate_line_breaks", True
        )

    text = raw_text

    # Normalize unicode (e.g. ligatures like "ﬁ" -> "fi") before anything else
    text = unicodedata.normalize("NFKC", text)

    if dehyphenate:
        text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

    text = _CONTROL_CHAR_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = text.strip()

    return text

"""
pdf_loader.py

Standard NLP/ingestion component (Section 3 of the design doc).
Responsible for exactly one thing: turning a PDF file on disk into a list
of (page_number, raw_text) pairs. No cleaning, no chunking, no NLP happens
here -- this module's only job is to preserve a faithful, page-numbered
extraction so that every later stage can trace text back to a page.

Uses pdfplumber (pure-Python, no external binary dependency), which keeps
the Phase 1 install simple on Windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from utils.exceptions import PDFLoadError
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PageText:
    """Raw extracted text for a single PDF page (1-indexed page numbers)."""
    page_number: int
    text: str


def load_pdf_pages(pdf_path: str | Path) -> list[PageText]:
    """
    Extract text from every page of a PDF.

    Args:
        pdf_path: path to a .pdf file.

    Returns:
        List of PageText, one entry per page, in page order, 1-indexed.
        Pages with no extractable text (e.g. a blank page, or a scanned
        image page with no text layer) are still included with an empty
        string, so page numbering stays correct -- callers decide whether
        to skip empty pages.

    Raises:
        PDFLoadError: if the file does not exist, is not a valid PDF,
            is password-protected, or yields zero pages.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise PDFLoadError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise PDFLoadError(f"Expected a .pdf file, got: {path.name}")

    logger.info("Opening PDF: %s", path)

    try:
        with pdfplumber.open(path) as pdf:
            if len(pdf.pages) == 0:
                raise PDFLoadError(f"PDF has zero pages: {path}")

            pages: list[PageText] = []
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as exc:  # pdfplumber can raise various parsing errors per-page
                    logger.warning(
                        "Failed to extract text from page %d of %s: %s. "
                        "Treating page as empty.",
                        i, path.name, exc,
                    )
                    text = ""
                pages.append(PageText(page_number=i, text=text))

    except PDFLoadError:
        raise
    except Exception as exc:
        # Covers pdfplumber/pdfminer errors: encrypted PDFs, corrupt files, etc.
        raise PDFLoadError(f"Could not open/parse PDF '{path}': {exc}") from exc

    non_empty_pages = sum(1 for p in pages if p.text.strip())
    if non_empty_pages == 0:
        raise PDFLoadError(
            f"PDF '{path.name}' produced no extractable text on any page. "
            "It may be a scanned/image-only PDF (no text layer), which is "
            "out of scope for Phase 1 (no OCR is implemented)."
        )

    logger.info(
        "Extracted %d page(s) from %s (%d with text).",
        len(pages), path.name, non_empty_pages,
    )
    return pages

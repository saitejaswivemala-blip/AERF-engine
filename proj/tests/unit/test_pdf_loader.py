"""
Unit tests for ingestion/pdf_loader.py.

Uses the real fixture PDF (tests/fixtures/sample_paper.pdf) rather than
mocking pdfplumber, since PDF parsing behavior is exactly what we need
confidence in for Phase 1.
"""

from pathlib import Path

import pytest

from ingestion.pdf_loader import load_pdf_pages
from utils.exceptions import PDFLoadError

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "sample_paper.pdf"


def test_load_pdf_pages_returns_correct_page_count():
    pages = load_pdf_pages(FIXTURE_PATH)
    assert len(pages) == 2


def test_load_pdf_pages_preserves_page_numbers_1_indexed():
    pages = load_pdf_pages(FIXTURE_PATH)
    assert [p.page_number for p in pages] == [1, 2]


def test_load_pdf_pages_extracts_known_text():
    pages = load_pdf_pages(FIXTURE_PATH)
    assert "Aspirin reduces cardiovascular risk" in pages[0].text
    assert "Drug A has no effect on recovery" in pages[1].text


def test_load_pdf_pages_raises_on_missing_file():
    with pytest.raises(PDFLoadError):
        load_pdf_pages("tests/fixtures/does_not_exist.pdf")


def test_load_pdf_pages_raises_on_non_pdf_extension(tmp_path):
    fake_file = tmp_path / "not_a_pdf.txt"
    fake_file.write_text("hello")
    with pytest.raises(PDFLoadError):
        load_pdf_pages(fake_file)

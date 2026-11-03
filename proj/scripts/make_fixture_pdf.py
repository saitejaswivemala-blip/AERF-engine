"""
make_fixture_pdf.py

Generates tests/fixtures/sample_paper.pdf — a small, deterministic 2-page
PDF used by the Phase 1 unit/integration tests. The fixture is checked
into the repo already (so `pip install -r requirements.txt && pytest`
works out of the box), but this script exists so you can regenerate it,
inspect exactly what it contains, or build additional fixtures the same
way.

The content is deliberately constructed to exercise:
  - multi-page extraction (2 pages, so page_number provenance is testable)
  - a word hyphenated across a line break ("cardio-" / "vascular"), to
    test text_cleaner's de-hyphenation
  - an abbreviation ("et al.") that should NOT be treated as a sentence
    boundary, to test chunker's abbreviation guard
  - multiple distinct sentences per page, to test sentence segmentation

Usage:
    python scripts/make_fixture_pdf.py
"""

import sys
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "tests" / "fixtures" / "sample_paper.pdf"

PAGE_1_LINES = [
    "Aspirin and Cardiovascular Risk: A Clinical Study",
    "",
    "Aspirin reduces cardiovascular risk in patients with a history of",
    "heart disease. This finding was reported by Smith et al. in a large",
    "clinical trial. The study included over 10,000 participants.",
    "Researchers observed a significant reduction in cardio-",
    "vascular events among the treatment group.",
]

PAGE_2_LINES = [
    "Contradicting Evidence Section",
    "",
    "A separate cohort study found that Drug A has no effect on recovery",
    "time. This contradicts earlier reports of a positive effect. Further",
    "randomized controlled trials are needed to resolve this disagreement.",
]


def _draw_page(c: canvas.Canvas, lines: list[str]) -> None:
    text_obj = c.beginText(72, 720)  # 1 inch margin, near top of US Letter page
    text_obj.setFont("Helvetica", 11)
    for line in lines:
        text_obj.textLine(line)
    c.drawText(text_obj)
    c.showPage()


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT_PATH), pagesize=LETTER)
    _draw_page(c, PAGE_1_LINES)
    _draw_page(c, PAGE_2_LINES)
    c.save()
    print(f"Wrote fixture PDF: {OUTPUT_PATH}")


if __name__ == "__main__":
    sys.exit(main() or 0)

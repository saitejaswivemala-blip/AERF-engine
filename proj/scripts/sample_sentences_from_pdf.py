"""
scripts/sample_sentences_from_pdf.py

Reproducible, blind sentence sampler for the AERF Engine independent evaluation.
Extracts candidate sentences from an unseen research PDF staged in data/independent_test_corpus/
WITHOUT running the claim extractor or inspecting model predictions.

Workflow:
1. Loads PDF text page-by-page.
2. Identifies structural sections (Abstract, Introduction, Methods, Results, Discussion).
3. Segments sentences using standard sentence boundary rules.
4. Filters out noise (captions, formulas, references, headers).
5. Samples exactly 15-20 sentences (default: 16) across sections using a fixed random seed.
6. Writes blind annotation rows to annotator sheets.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import re
import sys
from typing import Dict, List, Tuple


SECTION_PATTERNS = {
    "Abstract": re.compile(r"^\s*(?:abstract|summary)\b", re.I),
    "Introduction": re.compile(r"^\s*(?:\d+\.?\s*)?(?:introduction|background)\b", re.I),
    "Methods": re.compile(r"^\s*(?:\d+\.?\s*)?(?:methods?|methodology|experimental(?: setup| procedures)?|materials and methods)\b", re.I),
    "Results": re.compile(r"^\s*(?:\d+\.?\s*)?(?:results(?: and discussion)?|findings)\b", re.I),
    "Discussion": re.compile(r"^\s*(?:\d+\.?\s*)?(?:discussion|conclusions?|limitations?)\b", re.I),
    "References": re.compile(r"^\s*(?:\d+\.?\s*)?(?:references|bibliography|literature cited)\b", re.I),
}

def normalize_text_for_verify(text: str) -> str:
    text = text.replace("-\n", "").replace("­", "")
    text = re.sub(r"[–—−]", "-", text)
    text = re.sub(r"[\u2018\u2019`]", "'", text)
    text = re.sub(r'[\u201c\u201d"]', '"', text)
    text = re.sub(r"\s+", " ", text).lower()
    return text.strip()


NOISE_PATTERNS = [
    re.compile(r"^(?:table|figure|fig\.)\s*\d", re.I),
    re.compile(r"\b(?:figure|fig\.|table)\s*\d", re.I),
    re.compile(r"(?:@\w+\.\w+|https?://|www\.|doi:)", re.I),
    re.compile(r"\[\d+\]"),
    re.compile(r"\b(?:pp\.\s*\d+|vol\.\s*\d+|journal of|in proceedings of)\b", re.I),
    re.compile(r"\bet al\.,?\s*\d{4}", re.I),
    re.compile(r"\bet al\.:", re.I),
    re.compile(r"(?:preprint submitted|page \d+ of \d+)", re.I),
    re.compile(r"^\s*(?:\d+\.?\s*){1,4}$"),  # standalone section numbers
    re.compile(r"\b[a-zA-Z]{25,}\b"),  # glued words
    re.compile(r"\b\w+-\s+\w+\b"),  # broken hyphenation across line breaks
    re.compile(r"=\s*\(?\d+/\d+\)?"),  # raw math fractions
]


def is_valid_sentence(sent: str) -> bool:
    sent = sent.strip()
    if len(sent) < 40 or len(sent) > 500:
        return False
    words = sent.split()
    if len(words) < 7:
        return False
    # Check noise
    for np in NOISE_PATTERNS:
        if np.search(sent):
            return False
    # Must have alphabetic content and end with valid sentence punctuation
    if not re.search(r"[a-zA-Z]", sent):
        return False
    if not sent[-1] in ".!?":
        return False
    # Must start with a capital letter or quote
    if not (sent[0].isupper() or sent[0] in "\"'“‘("):
        return False
    return True


def segment_sentences(text: str) -> List[str]:
    # Basic sentence splitter that handles common scientific abbreviations
    abbrevs = r"(?<!\bet al)(?<!\bi\.e)(?<!\be\.g)(?<!\bFig)(?<!\bEq)(?<!\bRef)(?<!\bNo)(?<!\bDr)(?<!\bvs)"
    raw_sents = re.split(rf"{abbrevs}\.\s+(?=[A-Z0-9])", text)
    cleaned = []
    for s in raw_sents:
        s = s.strip()
        if not s.endswith("."):
            s = s + "."
        # clean internal line-break hyphens
        s = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", s)
        s = " ".join(s.split())
        if is_valid_sentence(s):
            cleaned.append(s)
    return cleaned


def extract_and_sample_sentences(
    pdf_path: Path,
    paper_id: str,
    doi_or_url: str,
    target_count: int = 16,
    seed: int = 42,
) -> List[Dict]:
    """
    Extracts text from PDF, segments by section, and pseudo-randomly samples sentences.
    Zero interaction with claim extractor model.
    """
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for p_num, page in enumerate(pdf.pages, start=1):
                txt = page.extract_text(x_tolerance=1.5) or ""
                pages.append((p_num, txt))
    except ImportError:
        # Fallback to PyPDF2 or basic text loader
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        pages = [(idx + 1, page.extract_text() or "") for idx, page in enumerate(reader.pages)]

    # Categorize sentences into sections
    current_section = "Introduction"  # default opening section if before abstract/intro
    section_sentences: Dict[str, List[Tuple[int, str]]] = {
        "Abstract": [],
        "Introduction": [],
        "Methods": [],
        "Results": [],
        "Discussion": [],
    }

    HEADER_FOOTER_PAT = re.compile(
        r"(?:page\s+\d+\s+of\s+\d+|preprint\s+submitted|arxiv:\d{4}\.\d+|^\d+\s*$|^[a-z\s]+et\s+al\.:)",
        re.I
    )

    for page_num, text in pages:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        page_buffer = []
        
        # Discard clear header lines at the top of the page (first 2 lines)
        start_idx = 0
        while start_idx < min(2, len(lines)) and HEADER_FOOTER_PAT.search(lines[start_idx]):
            start_idx += 1
            
        # Discard clear footer lines at the bottom (last 2 lines)
        end_idx = len(lines)
        while end_idx > max(start_idx, len(lines) - 2) and HEADER_FOOTER_PAT.search(lines[end_idx - 1]):
            end_idx -= 1

        for line in lines[start_idx:end_idx]:
            # Check section header transition
            for sec_name, pat in SECTION_PATTERNS.items():
                if pat.search(line) and len(line) < 50:
                    current_section = sec_name
                    break
            page_buffer.append(line)

        full_page_text = " ".join(page_buffer)
        sents = segment_sentences(full_page_text)
        for s in sents:
            if current_section != "References" and current_section in section_sentences:
                section_sentences[current_section].append((page_num, s))

    # Allocation: Abstract: 3, Intro: 3, Methods: 3, Results: 4, Discussion: 3 (Total = 16)
    allocation = {
        "Abstract": 3,
        "Introduction": 3,
        "Methods": 3,
        "Results": 4,
        "Discussion": 3,
    }

    rng = random.Random(seed)
    sampled = []
    seen_in_paper = set()
    sample_idx = 1

    for sec, count in allocation.items():
        raw_pool = section_sentences.get(sec, [])
        if not raw_pool:
            # Fallback to any available non-reference sentences if specific section was not parsed
            raw_pool = [item for s_name, s_list in section_sentences.items() if s_name != "References" for item in s_list]

        # Filter out anything already chosen, duplicates, or sentences not cleanly verifiable
        full_doc_clean = " ".join([normalize_text_for_verify(t) for _, t in pages])
        unique_pool = []
        for p_num, s_text in raw_pool:
            norm = " ".join(re.sub(r"[^\w\s]", "", s_text.lower()).split())
            if norm not in seen_in_paper:
                # Strictly verify that s_text can be matched in the document text
                s_norm = normalize_text_for_verify(s_text)
                if s_norm in full_doc_clean:
                    unique_pool.append((p_num, s_text))

        k = min(count, len(unique_pool))
        chosen = rng.sample(unique_pool, k) if len(unique_pool) >= k else unique_pool

        for page_num, sent in chosen:
            norm = " ".join(re.sub(r"[^\w\s]", "", sent.lower()).split())
            seen_in_paper.add(norm)
            sampled.append({
                "sample_id": f"{paper_id}_{sample_idx:02d}",
                "paper_id": paper_id,
                "sentence": sent,
                "source_section": sec,
                "page_number": page_num,
                "doi_or_url": doi_or_url,
                "annotator_label": "",
                "annotator_rationale": "",
            })
            sample_idx += 1

    return sampled


def main():
    parser = argparse.ArgumentParser(description="Sample blind sentences from an unseen research PDF.")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to PDF in data/independent_test_corpus/")
    parser.add_argument("--id", type=str, required=True, help="Paper ID (e.g. INDEP_01_MATSCI)")
    parser.add_argument("--doi", type=str, required=True, help="Paper DOI or URL")
    parser.add_argument("--count", type=int, default=16, help="Target sentences count (default: 16)")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed for sampling reproducibility")
    parser.add_argument(
        "--append-sheets",
        action="store_true",
        help="If set, appends rows to tests/data/independent_eval/annotator_1_sheet.csv and annotator_2_sheet.csv",
    )

    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"[ERROR] PDF not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    samples = extract_and_sample_sentences(
        pdf_path=args.pdf,
        paper_id=args.id,
        doi_or_url=args.doi,
        target_count=args.count,
        seed=args.seed,
    )

    print(f"\nExtracted {len(samples)} blind sentences from {args.pdf.name} across sections:")
    for s in samples:
        print(f"  [{s['source_section']} p.{s['page_number']}] {s['sentence'][:70]}...")

    if args.append_sheets:
        for sheet_path in [
            Path("tests/data/independent_eval/annotator_1_sheet.csv"),
            Path("tests/data/independent_eval/annotator_2_sheet.csv"),
        ]:
            file_exists = sheet_path.exists()
            with open(sheet_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["sample_id", "paper_id", "sentence", "source_section", "page_number", "doi_or_url", "annotator_label", "annotator_rationale"],
                )
                if not file_exists or sheet_path.stat().st_size == 0:
                    writer.writeheader()
                writer.writerows(samples)
            print(f"Appended {len(samples)} candidate rows to {sheet_path}")


if __name__ == "__main__":
    main()

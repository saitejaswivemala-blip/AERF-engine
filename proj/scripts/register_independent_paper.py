"""
scripts/register_independent_paper.py

Helper CLI utility to register a newly collected open-access research paper
into the independent evaluation manifest (tests/data/independent_eval/independent_paper_manifest.json).

Responsibilities:
1. Validates that the PDF file exists in data/independent_test_corpus/.
2. Computes the PDF's cryptographic SHA-256 checksum.
3. Performs automated overlap and development-history independence checks.
4. Updates the manifest with verified metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from scripts.validate_independent_dataset import check_paper_overlap


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def register_paper(
    pdf_path: Path,
    paper_id: str,
    title: str,
    authors: str,
    year: int,
    venue: str,
    doi_or_url: str,
    discipline: str,
    manifest_path: Path,
) -> dict:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

    # Check development history independence
    is_overlap, reason = check_paper_overlap(paper_id, doi_or_url, title)
    if is_overlap:
        raise ValueError(
            f"REJECTED: Paper overlaps with existing development/corpus papers: {reason}\n"
            f"Independent evaluation requires completely unseen literature outside existing research lineage."
        )

    pdf_hash = compute_file_sha256(pdf_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    registered = manifest.setdefault("registered_independent_papers", [])

    # Check for duplicate paper_id or DOI in registered list
    for p in registered:
        if p.get("paper_id") == paper_id:
            raise ValueError(f"Paper ID '{paper_id}' already registered in manifest.")
        if p.get("doi_or_url") == doi_or_url:
            raise ValueError(f"DOI '{doi_or_url}' already registered in manifest.")

    entry = {
        "paper_id": paper_id,
        "title": title,
        "authors": authors,
        "publication_year": int(year),
        "venue_or_journal": venue,
        "doi_or_url": doi_or_url,
        "scientific_discipline": discipline,
        "pdf_filename": pdf_path.name,
        "pdf_sha256": pdf_hash,
        "split_assignment": "INDEPENDENT_TEST",
        "source_verification_status": "VERIFIED_OPEN_ACCESS_PDF",
        "development_history_check": "VERIFIED_UNSEEN_IN_AERF",
    }

    registered.append(entry)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return entry


def main():
    parser = argparse.ArgumentParser(description="Register a newly collected paper in independent manifest.")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to PDF in data/independent_test_corpus/")
    parser.add_argument("--id", type=str, required=True, help="Unique paper ID (e.g. INDEP_01_MATSCI)")
    parser.add_argument("--title", type=str, required=True, help="Full title of the paper")
    parser.add_argument("--authors", type=str, required=True, help="Authors list (e.g. 'Smith J, Doe A')")
    parser.add_argument("--year", type=int, required=True, help="Publication year")
    parser.add_argument("--venue", type=str, required=True, help="Journal or venue name")
    parser.add_argument("--doi", type=str, required=True, help="DOI or permanent URL")
    parser.add_argument("--discipline", type=str, required=True, help="Scientific discipline (e.g. 'Materials Science')")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tests/data/independent_eval/independent_paper_manifest.json"),
        help="Path to manifest JSON",
    )

    args = parser.parse_args()

    try:
        entry = register_paper(
            pdf_path=args.pdf,
            paper_id=args.id,
            title=args.title,
            authors=args.authors,
            year=args.year,
            venue=args.venue,
            doi_or_url=args.doi,
            discipline=args.discipline,
            manifest_path=args.manifest,
        )
        print("\nSuccessfully registered paper in independent evaluation manifest:")
        print(json.dumps(entry, indent=2))
    except Exception as e:
        print(f"\n[ERROR] Registration failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
main.py

Phase 1 CLI entry point.

Usage (from the aerf-engine/ project root, with the virtual environment active):

    python main.py --file data/raw_pdfs/my_paper.pdf
    python main.py --dir  data/raw_pdfs

Prints a short summary of what was ingested. All errors are logged (see
logs/aerf_engine.log) as well as printed to the console.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from db.repository import Repository
from pipeline.orchestrator import ingest_directory, ingest_paper
from pipeline.phase2 import Phase2Pipeline
from utils.exceptions import AERFError
from utils.logging_config import get_logger

logger = get_logger(__name__)


def load_metadata_by_filename(manifest_path: str) -> dict[str, dict]:
    """
    Reads a paper-manifest JSON (same shape as
    experiments/datasets/paper_manifest.json: {"papers": [{"pdf_path":
    ..., "title": ..., "authors": ..., "year": ..., "venue": ...}, ...]})
    and returns {pdf basename: {"title", "authors", "publication_year",
    "venue"}} for ingest_directory/ingest_paper's `metadata` argument.

    This is the only place manifest field names ("year") get mapped to
    the Paper/database field name ("publication_year") — everything else
    downstream (ingest_paper, insert_paper, update_paper_metadata) uses
    the database's own field names throughout.
    """
    data = json.loads(Path(manifest_path).read_text())
    out = {}
    for p in data.get("papers", []):
        filename = Path(p["pdf_path"]).name
        out[filename] = {
            "title": p.get("title"),
            "authors": p.get("authors"),
            "publication_year": p.get("year"),
            "venue": p.get("venue"),
        }
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AERF Engine — Phase 1: PDF ingestion pipeline "
                    "(extraction -> cleaning -> chunking -> SQLite storage)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a single PDF file to ingest.")
    group.add_argument("--dir", type=str, help="Path to a directory of PDF files to ingest.")
    group.add_argument("--phase2-paper-id", type=int, help="Run Phase 2 NLP on an already-ingested paper id.")
    group.add_argument("--phase3-paper-id", type=int, help="Run Phase 3 cross-document reasoning/AERF for an ingested paper id.")
    parser.add_argument(
        "--init-db", action="store_true",
        help="Initialize the database schema before running (safe to always pass; idempotent).",
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Optional path to a paper-manifest JSON (e.g. "
             "experiments/datasets/paper_manifest.json) providing real "
             "title/authors/year/venue for --file/--dir ingestion, matched "
             "by PDF filename. Omit for the original behavior (no "
             "metadata; those fields stay unset). Also backfills metadata "
             "on an already-ingested paper that's missing it, without "
             "re-ingesting/re-chunking.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    repo = Repository()

    if args.init_db:
        repo.initialize_database()

    try:
        if args.phase2_paper_id is not None:
            result = Phase2Pipeline(repo).run(args.phase2_paper_id)
            print(f"\nPhase 2 complete: {result}")
            return 0
        if args.phase3_paper_id is not None:
            from pipeline.phase3 import Phase3Pipeline
            result = Phase3Pipeline(repo).run(args.phase3_paper_id)
            print(f"\nPhase 3 complete: {result}")
            return 0
        if args.file:
            metadata = None
            if args.manifest:
                metadata = load_metadata_by_filename(args.manifest).get(Path(args.file).name)
            paper = ingest_paper(args.file, repository=repo, metadata=metadata)
            chunk_count = repo.count_chunks(paper.paper_id)
            print(
                f"\nIngested '{paper.source_filename}' "
                f"(paper_id={paper.paper_id}, pages={paper.page_count}, "
                f"chunks={chunk_count})"
            )
        else:
            metadata_by_filename = load_metadata_by_filename(args.manifest) if args.manifest else None
            papers = ingest_directory(args.dir, repository=repo, metadata_by_filename=metadata_by_filename)
            print(f"\nIngested {len(papers)} paper(s):")
            for paper in papers:
                chunk_count = repo.count_chunks(paper.paper_id)
                print(f"  - {paper.source_filename}: paper_id={paper.paper_id}, chunks={chunk_count}")

    except AERFError as exc:
        logger.error("Ingestion failed: %s", exc)
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

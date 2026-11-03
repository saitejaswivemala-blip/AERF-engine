"""
orchestrator.py

Phase 1 pipeline: PDF -> extraction -> cleaning -> provenance-preserving
chunks -> SQLite storage. Deliberately stops here — no NER, no claim
extraction, no embeddings, no NLI, no AERF, per the Phase 1 scope. Later
phases will extend this same orchestrator with additional steps rather
than replacing it (see `pipeline/orchestrator.py` in the design doc's
Section 4 data-flow, which this implements the first stage of).

Idempotent per paper: re-ingesting a file with the same filename does not
duplicate rows; it logs and returns the existing paper.
"""

from __future__ import annotations

from pathlib import Path

from db.models import Chunk, Paper
from db.repository import Repository
from ingestion.chunker import chunk_page
from ingestion.pdf_loader import load_pdf_pages
from ingestion.text_cleaner import clean_text
from utils.exceptions import AERFError
from utils.logging_config import get_logger

logger = get_logger(__name__)


def ingest_paper(
    pdf_path: str | Path,
    repository: Repository | None = None,
    metadata: dict | None = None,
) -> Paper:
    """
    Run the full Phase 1 pipeline on a single PDF: load -> clean -> chunk
    -> persist. Returns the persisted Paper (with paper_id set).

    Args:
        pdf_path: path to the PDF file to ingest.
        repository: optional Repository instance (mainly for tests, to
            point at a temporary database). Defaults to the configured
            database in settings.yaml.
        metadata: optional dict with any of "title", "authors",
            "publication_year", "venue" — bibliographic metadata for this
            paper, supplied by the caller (e.g. from a paper manifest).
            Phase 1 itself extracts no metadata from the PDF (see this
            module's docstring: text extraction only), so without this
            argument every paper's title/authors/publication_year/venue
            stay None, exactly as before this parameter existed — this is
            purely additive and backward compatible. When the paper is
            already ingested (idempotent path below), any fields present
            here are still applied via update_paper_metadata() if the
            existing row is missing them, so metadata can be backfilled
            for already-ingested papers without re-ingesting/re-chunking.

    Raises:
        PDFLoadError, TextCleaningError, ChunkingError, DatabaseError:
            propagated from the underlying stage with no swallowing —
            the caller (CLI/tests) decides how to report failures.
    """
    repo = repository or Repository()
    path = Path(pdf_path)
    metadata = metadata or {}

    if repo.paper_exists(path.name):
        existing = repo.get_paper_by_filename(path.name)
        logger.info(
            "Paper '%s' already ingested (paper_id=%s). Skipping re-ingestion.",
            path.name, existing.paper_id,
        )
        if metadata:
            missing = {
                k: v for k, v in metadata.items()
                if v is not None and getattr(existing, k, None) is None
            }
            if missing:
                logger.info(
                    "Backfilling metadata for already-ingested paper '%s' (paper_id=%s): %s",
                    path.name, existing.paper_id, sorted(missing),
                )
                existing = repo.update_paper_metadata(existing.paper_id, **missing)
        return existing

    logger.info("Starting Phase 1 ingestion for: %s", path)

    # Step 1: PDF -> raw per-page text
    pages = load_pdf_pages(path)

    # Step 2 + 3: clean each page, then chunk it, collecting all chunks
    # for this paper before touching the database (so a single failed
    # page doesn't leave a partially-inserted paper behind).
    all_raw_chunks = []
    for page in pages:
        cleaned = clean_text(page.text)
        page_chunks = chunk_page(page, cleaned)
        all_raw_chunks.extend(page_chunks)

    if not all_raw_chunks:
        # load_pdf_pages already guards against zero extractable text,
        # but a document that is all header/footer noise below
        # min_chunk_length could still end up empty here.
        logger.warning(
            "Paper '%s' produced 0 chunks after cleaning/chunking "
            "(all extracted text was below min_chunk_length).", path.name
        )

    # Step 4: persist paper row first (chunks need its paper_id)
    paper = Paper(
        source_filename=path.name,
        source_path=str(path.resolve()),
        page_count=len(pages),
        title=metadata.get("title"),
        authors=metadata.get("authors"),
        publication_year=metadata.get("publication_year"),
        venue=metadata.get("venue"),
    )
    paper = repo.insert_paper(paper)

    # Step 5: attach paper_id and persist chunks
    chunks = [
        Chunk(
            paper_id=paper.paper_id,
            page_number=rc.page_number,
            char_start=rc.char_start,
            char_end=rc.char_end,
            text=rc.text,
        )
        for rc in all_raw_chunks
    ]
    repo.insert_chunks(chunks)

    logger.info(
        "Finished ingesting '%s': paper_id=%s, %d page(s), %d chunk(s).",
        path.name, paper.paper_id, paper.page_count, len(chunks),
    )
    return paper


def ingest_directory(
    directory: str | Path,
    repository: Repository | None = None,
    metadata_by_filename: dict[str, dict] | None = None,
) -> list[Paper]:
    """
    Ingest every .pdf file in a directory (non-recursive). Continues past
    individual file failures (logs them) rather than aborting the whole
    batch — returns only the papers that succeeded.

    Args:
        metadata_by_filename: optional {pdf basename: metadata dict}
            (same shape as ingest_paper's `metadata` argument), matched
            against each file's Path(...).name. A file with no matching
            key gets no metadata (identical to today's behavior) — this
            parameter is purely additive.
    """
    repo = repository or Repository()
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise AERFError(f"Not a directory: {dir_path}")

    pdf_files = sorted(dir_path.glob("*.pdf"))
    logger.info("Found %d PDF(s) in %s", len(pdf_files), dir_path)
    metadata_by_filename = metadata_by_filename or {}

    results: list[Paper] = []
    for pdf_file in pdf_files:
        try:
            results.append(ingest_paper(
                pdf_file, repository=repo,
                metadata=metadata_by_filename.get(pdf_file.name),
            ))
        except AERFError as exc:
            logger.error("Failed to ingest '%s': %s", pdf_file.name, exc)

    return results

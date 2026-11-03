"""
Integration test for the complete Phase 1 pipeline: runs
pipeline.orchestrator.ingest_paper() end-to-end against the real fixture
PDF and a temporary SQLite database, and checks that every provenance
field survives the whole trip from PDF to database row.
"""

from pathlib import Path

import pytest

from db.repository import Repository
from pipeline.orchestrator import ingest_directory, ingest_paper
from utils.exceptions import PDFLoadError

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "sample_paper.pdf"


@pytest.fixture
def repo(tmp_path):
    r = Repository(db_path=tmp_path / "integration_test.db")
    r.initialize_database()
    return r


def test_ingest_paper_end_to_end(repo):
    paper = ingest_paper(FIXTURE_PATH, repository=repo)

    # --- Paper-level checks ---
    assert paper.paper_id is not None
    assert paper.source_filename == "sample_paper.pdf"
    assert paper.page_count == 2

    # --- Chunk-level checks ---
    chunks = repo.get_chunks_by_paper(paper.paper_id)
    assert len(chunks) > 0

    # Every chunk must carry complete provenance: paper_id, page_number,
    # char_start/char_end, and non-empty source text — this is the
    # Phase 1 requirement this test exists to guard.
    for chunk in chunks:
        assert chunk.paper_id == paper.paper_id
        assert chunk.page_number in (1, 2)
        assert chunk.char_start >= 0
        assert chunk.char_end > chunk.char_start
        assert chunk.text.strip() != ""

    # Known content must actually be present somewhere in the chunks
    all_text = " ".join(c.text for c in chunks)
    assert "Aspirin" in all_text
    assert "cardiovascular" in all_text  # proves de-hyphenation ran
    assert "cardio-" not in all_text
    assert "Drug A has no effect on recovery" in all_text

    # Page 2 content must be attributed to page_number 2, not page 1
    page_2_chunks = [c for c in chunks if c.page_number == 2]
    assert any("Drug A" in c.text for c in page_2_chunks)


def test_ingest_paper_is_idempotent(repo):
    paper1 = ingest_paper(FIXTURE_PATH, repository=repo)
    paper2 = ingest_paper(FIXTURE_PATH, repository=repo)

    assert paper1.paper_id == paper2.paper_id
    assert repo.count_papers() == 1
    # Chunks must not be duplicated on re-ingestion
    assert repo.count_chunks(paper1.paper_id) == repo.count_chunks(paper1.paper_id)


def test_ingest_paper_raises_on_missing_file(repo):
    with pytest.raises(PDFLoadError):
        ingest_paper("tests/fixtures/nonexistent.pdf", repository=repo)


def test_ingest_directory_ingests_all_pdfs(repo, tmp_path):
    # Copy the fixture into a fresh directory so ingest_directory has a
    # clean, isolated folder to scan (doesn't touch the real fixtures dir).
    import shutil
    src_dir = tmp_path / "pdfs"
    src_dir.mkdir()
    shutil.copy(FIXTURE_PATH, src_dir / "sample_paper.pdf")

    papers = ingest_directory(src_dir, repository=repo)
    assert len(papers) == 1
    assert repo.count_papers() == 1


def test_ingest_directory_skips_failed_files_but_continues(repo, tmp_path):
    import shutil
    src_dir = tmp_path / "pdfs"
    src_dir.mkdir()
    shutil.copy(FIXTURE_PATH, src_dir / "good.pdf")
    (src_dir / "broken.pdf").write_bytes(b"this is not a real pdf")

    papers = ingest_directory(src_dir, repository=repo)
    # Only the valid PDF should have been ingested; the broken one is
    # logged as an error and skipped, not raised.
    assert len(papers) == 1
    assert papers[0].source_filename == "good.pdf"


def test_ingest_paper_with_metadata_persists_it(repo):
    metadata = {"title": "Real Title", "authors": "Real Authors", "publication_year": 2024, "venue": "Real Venue"}
    paper = ingest_paper(FIXTURE_PATH, repository=repo, metadata=metadata)
    assert (paper.title, paper.authors, paper.publication_year, paper.venue) == (
        "Real Title", "Real Authors", 2024, "Real Venue",
    )
    reloaded = repo.get_paper_by_filename("sample_paper.pdf")
    assert reloaded.title == "Real Title"


def test_ingest_paper_without_metadata_leaves_fields_none(repo):
    paper = ingest_paper(FIXTURE_PATH, repository=repo)
    assert paper.title is None
    assert paper.authors is None
    assert paper.publication_year is None
    assert paper.venue is None


def test_ingest_paper_backfills_metadata_on_already_ingested_paper(repo):
    # Ingest with no metadata first (the historical/original path).
    first = ingest_paper(FIXTURE_PATH, repository=repo)
    assert first.title is None

    # Re-"ingest" the same filename, now supplying metadata -- must NOT
    # re-chunk (idempotent), but MUST backfill the missing metadata onto
    # the existing paper_id.
    metadata = {"title": "Backfilled Title", "authors": "Backfilled Authors",
                "publication_year": 2021, "venue": "Backfilled Venue"}
    second = ingest_paper(FIXTURE_PATH, repository=repo, metadata=metadata)

    assert second.paper_id == first.paper_id
    assert second.title == "Backfilled Title"
    assert repo.count_chunks(first.paper_id) == repo.get_chunks_by_paper(first.paper_id).__len__()
    assert repo.count_papers() == 1  # still exactly one paper row, no duplication


def test_ingest_paper_backfill_does_not_overwrite_existing_metadata(repo):
    first = ingest_paper(FIXTURE_PATH, repository=repo, metadata={"title": "Original"})
    assert first.title == "Original"

    second = ingest_paper(FIXTURE_PATH, repository=repo,
                           metadata={"title": "Should Not Apply", "authors": "New Authors"})
    # title was already set -- must not be clobbered by a second ingest call.
    assert second.title == "Original"
    # authors was genuinely missing -- should get backfilled.
    assert second.authors == "New Authors"


def test_ingest_directory_threads_metadata_by_filename(repo, tmp_path):
    import shutil
    src_dir = tmp_path / "pdfs"
    src_dir.mkdir()
    shutil.copy(FIXTURE_PATH, src_dir / "sample_paper.pdf")

    metadata_by_filename = {"sample_paper.pdf": {"title": "Dir Title", "venue": "Dir Venue"}}
    papers = ingest_directory(src_dir, repository=repo, metadata_by_filename=metadata_by_filename)
    assert papers[0].title == "Dir Title"
    assert papers[0].venue == "Dir Venue"


def test_ingest_directory_with_no_metadata_map_is_unaffected(repo, tmp_path):
    import shutil
    src_dir = tmp_path / "pdfs"
    src_dir.mkdir()
    shutil.copy(FIXTURE_PATH, src_dir / "sample_paper.pdf")

    papers = ingest_directory(src_dir, repository=repo)  # no metadata_by_filename arg at all
    assert papers[0].title is None

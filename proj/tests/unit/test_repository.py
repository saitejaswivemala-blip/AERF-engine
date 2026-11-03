"""
Unit tests for db/repository.py.

Every test gets its own temporary SQLite file (via pytest's tmp_path
fixture) so tests never touch the real data/aerf.db and can run in any
order without interfering with each other.
"""

import pytest

from db.models import Chunk, Paper
from db.repository import Repository
from utils.exceptions import DatabaseError


@pytest.fixture
def repo(tmp_path):
    r = Repository(db_path=tmp_path / "test.db")
    r.initialize_database()
    return r


def test_initialize_database_creates_papers_table(repo):
    assert repo.count_papers() == 0


def test_insert_paper_assigns_id(repo):
    paper = Paper(source_filename="a.pdf", source_path="/tmp/a.pdf", page_count=3)
    inserted = repo.insert_paper(paper)
    assert inserted.paper_id is not None
    assert inserted.paper_id > 0


def test_insert_paper_then_get_by_filename(repo):
    repo.insert_paper(Paper(source_filename="a.pdf", source_path="/tmp/a.pdf", page_count=3))
    fetched = repo.get_paper_by_filename("a.pdf")
    assert fetched is not None
    assert fetched.source_filename == "a.pdf"
    assert fetched.page_count == 3


def test_get_paper_by_filename_returns_none_when_absent(repo):
    assert repo.get_paper_by_filename("does_not_exist.pdf") is None


def test_paper_exists_true_and_false(repo):
    assert repo.paper_exists("a.pdf") is False
    repo.insert_paper(Paper(source_filename="a.pdf", source_path="/tmp/a.pdf", page_count=1))
    assert repo.paper_exists("a.pdf") is True


def test_insert_duplicate_filename_raises(repo):
    repo.insert_paper(Paper(source_filename="a.pdf", source_path="/tmp/a.pdf", page_count=1))
    with pytest.raises(DatabaseError):
        repo.insert_paper(Paper(source_filename="a.pdf", source_path="/tmp/a.pdf", page_count=1))


def test_insert_and_get_chunks(repo):
    paper = repo.insert_paper(Paper(source_filename="a.pdf", source_path="/tmp/a.pdf", page_count=1))
    chunks = [
        Chunk(paper_id=paper.paper_id, page_number=1, char_start=0, char_end=10, text="First one."),
        Chunk(paper_id=paper.paper_id, page_number=1, char_start=11, char_end=25, text="Second chunk."),
    ]
    repo.insert_chunks(chunks)

    fetched = repo.get_chunks_by_paper(paper.paper_id)
    assert len(fetched) == 2
    assert all(c.chunk_id is not None for c in fetched)
    assert fetched[0].text == "First one."


def test_insert_chunks_empty_list_is_noop(repo):
    assert repo.insert_chunks([]) == []


def test_count_chunks_filters_by_paper(repo):
    paper1 = repo.insert_paper(Paper(source_filename="a.pdf", source_path="/tmp/a.pdf", page_count=1))
    paper2 = repo.insert_paper(Paper(source_filename="b.pdf", source_path="/tmp/b.pdf", page_count=1))
    repo.insert_chunks([Chunk(paper_id=paper1.paper_id, page_number=1, char_start=0, char_end=5, text="hello")])
    repo.insert_chunks([
        Chunk(paper_id=paper2.paper_id, page_number=1, char_start=0, char_end=5, text="world"),
        Chunk(paper_id=paper2.paper_id, page_number=1, char_start=6, char_end=11, text="again"),
    ])
    assert repo.count_chunks(paper1.paper_id) == 1
    assert repo.count_chunks(paper2.paper_id) == 2
    assert repo.count_chunks() == 3


def test_chunks_cascade_delete_when_paper_deleted(repo):
    paper = repo.insert_paper(Paper(source_filename="a.pdf", source_path="/tmp/a.pdf", page_count=1))
    repo.insert_chunks([Chunk(paper_id=paper.paper_id, page_number=1, char_start=0, char_end=5, text="hello")])
    with repo._connect() as conn:
        conn.execute("DELETE FROM papers WHERE paper_id = ?", (paper.paper_id,))
    assert repo.count_chunks(paper.paper_id) == 0


def test_update_paper_metadata_sets_only_provided_fields(repo):
    paper = repo.insert_paper(Paper(source_filename="a.pdf", source_path="/tmp/a.pdf", page_count=1))
    assert paper.title is None and paper.venue is None

    updated = repo.update_paper_metadata(paper.paper_id, title="Real Title", publication_year=2024)
    assert updated.title == "Real Title"
    assert updated.publication_year == 2024
    # authors/venue were not passed -- must stay None, not get clobbered.
    assert updated.authors is None
    assert updated.venue is None


def test_update_paper_metadata_does_not_overwrite_existing_value_with_none(repo):
    paper = repo.insert_paper(Paper(
        source_filename="a.pdf", source_path="/tmp/a.pdf", page_count=1,
        title="Original Title",
    ))
    updated = repo.update_paper_metadata(paper.paper_id, authors="Real Authors")
    assert updated.authors == "Real Authors"
    assert updated.title == "Original Title"  # untouched, since title=None wasn't passed


def test_update_paper_metadata_persists_across_reload(repo):
    paper = repo.insert_paper(Paper(source_filename="a.pdf", source_path="/tmp/a.pdf", page_count=1))
    repo.update_paper_metadata(paper.paper_id, title="X", authors="Y", publication_year=2020, venue="Z")
    reloaded = repo.get_paper_by_filename("a.pdf")
    assert (reloaded.title, reloaded.authors, reloaded.publication_year, reloaded.venue) == ("X", "Y", 2020, "Z")


def test_update_paper_metadata_unknown_paper_id_returns_none(repo):
    assert repo.update_paper_metadata(99999, title="X") is None


def test_update_paper_metadata_no_fields_is_noop(repo):
    paper = repo.insert_paper(Paper(source_filename="a.pdf", source_path="/tmp/a.pdf", page_count=1, title="Keep Me"))
    result = repo.update_paper_metadata(paper.paper_id)
    assert result.title == "Keep Me"


def test_delete_claim_pairs_by_matching_method_removes_only_matching_rows(repo):
    p1 = repo.insert_paper(Paper(source_filename="a.pdf", source_path="/tmp/a.pdf", page_count=1))
    p2 = repo.insert_paper(Paper(source_filename="b.pdf", source_path="/tmp/b.pdf", page_count=1))
    repo.ensure_phase2_schema()
    repo.ensure_phase3_schema()

    class C:
        pass
    c1 = C(); c1.text = "Claim one."; c1.start = 0; c1.end = 10
    c2 = C(); c2.text = "Claim two."; c2.start = 0; c2.end = 10
    repo.insert_chunks([Chunk(p1.paper_id, 1, 0, 10, "Claim one.")])
    repo.insert_chunks([Chunk(p2.paper_id, 1, 0, 10, "Claim two.")])
    claim_a = repo.insert_claim_p2(p1.paper_id, 1, c1)
    claim_b = repo.insert_claim_p2(p2.paper_id, 2, c2)

    repo.upsert_claim_pair({
        "claim_a_id": claim_a, "claim_b_id": claim_b, "cosine_similarity": 0.9,
        "nli_label": "neutral", "nli_confidence": 0.7, "entity_overlap_score": 0.1,
        "validated_as": None, "matching_method": "curated_real_model_evaluation",
    })
    other_pair_claim = repo.insert_claim_p2(p1.paper_id, 1, c1)
    repo.upsert_claim_pair({
        "claim_a_id": claim_a, "claim_b_id": other_pair_claim, "cosine_similarity": 0.9,
        "nli_label": "entailment", "nli_confidence": 0.9, "entity_overlap_score": 0.5,
        "validated_as": "supports", "matching_method": "embedding→entity-alignment→bidirectional-NLI",
    })

    with repo._connect() as conn:
        before = conn.execute("SELECT COUNT(*) n FROM claim_pairs").fetchone()["n"]
    assert before == 2

    deleted = repo.delete_claim_pairs_by_matching_method("curated_real_model_evaluation")
    assert deleted == 1

    with repo._connect() as conn:
        remaining = conn.execute("SELECT matching_method FROM claim_pairs").fetchall()
    assert [r["matching_method"] for r in remaining] == ["embedding→entity-alignment→bidirectional-NLI"]

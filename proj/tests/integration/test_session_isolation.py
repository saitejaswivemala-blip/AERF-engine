from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

import pytest

from db.models import Chunk, Paper
from db.repository import Repository
from pipeline.session_manager import (
    create_session,
    generate_session_id,
    get_session,
    get_session_repository,
    get_session_results,
    list_sessions,
    run_session_analysis,
)


ORIGINAL_DB_PATH = Path(__file__).resolve().parent.parent.parent / "experiments" / "runs" / "experiment_FINAL_2026-09-02.db"
SAMPLE_PDF_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "sample_paper.pdf"


def _get_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestSessionIsolation:
    """Automated tests proving session-based multi-paper isolation and integrity."""

    def test_session_id_generation_unique(self):
        """Verify unique session IDs with timestamp and randomness."""
        sid1 = generate_session_id()
        sid2 = generate_session_id()
        assert sid1.startswith("session_")
        assert sid2.startswith("session_")
        assert sid1 != sid2

    def test_original_database_remains_unchanged(self):
        """Verify that the existing 11-paper corpus database is untouched."""
        if not ORIGINAL_DB_PATH.exists():
            pytest.skip("Original database not present.")

        initial_hash = _get_file_hash(ORIGINAL_DB_PATH)

        # Create and run an isolated session in a temporary database
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_db = Path(tmpdir) / "test_session.db"
            sid, repo = create_session("test_sess_orig_check", custom_db_path=tmp_db)

            # Insert a dummy paper in the test session
            paper = repo.insert_paper(
                Paper(
                    source_filename="new_isolated_paper.pdf",
                    source_path=str(Path(tmpdir) / "new_isolated_paper.pdf"),
                    page_count=5,
                )
            )
            repo.link_session_paper(sid, paper.paper_id, paper.source_filename)

            # Check original DB
            after_hash = _get_file_hash(ORIGINAL_DB_PATH)
            assert initial_hash == after_hash, "Original database file was modified!"

            # Verify original DB paper count is still exactly 11
            orig_repo = Repository(ORIGINAL_DB_PATH)
            assert orig_repo.count_papers() == 11

    def test_two_paper_session_isolation_and_no_leakage(self):
        """
        Prove that a session containing two new PDFs displays only those papers
        and their results, and that records from the original database do not appear.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_db = Path(tmpdir) / "session_two_papers.db"
            sid, repo = create_session("session_two_papers_test", custom_db_path=tmp_db)

            # 1. Ingest 2 new papers into the session repository
            p1 = repo.insert_paper(
                Paper(
                    title="Subglacial Drainage Dynamics",
                    authors="Smith et al.",
                    publication_year=2024,
                    source_filename="session_paper_1.pdf",
                    source_path=str(Path(tmpdir) / "session_paper_1.pdf"),
                    page_count=10,
                )
            )
            p2 = repo.insert_paper(
                Paper(
                    title="Glacial Water Pressure Analysis",
                    authors="Jones et al.",
                    publication_year=2025,
                    source_filename="session_paper_2.pdf",
                    source_path=str(Path(tmpdir) / "session_paper_2.pdf"),
                    page_count=8,
                )
            )
            repo.link_session_paper(sid, p1.paper_id, p1.source_filename)
            repo.link_session_paper(sid, p2.paper_id, p2.source_filename)

            # 2. Add chunks and claims for both papers
            ch1 = repo.insert_chunks([
                Chunk(paper_id=p1.paper_id, page_number=1, char_start=0, char_end=100, text="Subglacial water pressure fluctuates seasonally.")
            ])[0]
            ch2 = repo.insert_chunks([
                Chunk(paper_id=p2.paper_id, page_number=1, char_start=0, char_end=100, text="Water pressure measurements confirmed winter decreases.")
            ])[0]

            c1_id = repo.insert_claim_p2(
                p1.paper_id,
                ch1.chunk_id,
                type("Claim", (), {"text": "Subglacial water pressure fluctuates seasonally.", "start": 0, "end": 48})(),
            )
            c2_id = repo.insert_claim_p2(
                p2.paper_id,
                ch2.chunk_id,
                type("Claim", (), {"text": "Water pressure measurements confirmed winter decreases.", "start": 0, "end": 56})(),
            )

            # Add an intra-session claim pair
            repo.upsert_claim_pair({
                "claim_a_id": c1_id,
                "claim_b_id": c2_id,
                "cosine_similarity": 0.88,
                "nli_label": "entailment",
                "nli_confidence": 0.92,
                "entity_overlap_score": 0.40,
                "validated_as": "supports",
                "matching_method": "cosine_nli",
            })

            # Add AERF score for both claims
            repo.store_aerf_score(c1_id, 0.75, {"corroboration": 0.6, "source": 0.8}, "v1")
            repo.store_aerf_score(c2_id, 0.80, {"corroboration": 0.7, "source": 0.8}, "v1")

            # Link graph edges
            repo.upsert_graph_edge("claim", c1_id, "claim", c2_id, "supports", 0.92, 1)

            # 3. Retrieve session results
            results = get_session_results(sid, repo=repo)

            # 4. Assertions on Session Results
            paper_ids = results["paper_ids"]
            assert sorted(paper_ids) == sorted([p1.paper_id, p2.paper_id])
            assert len(results["papers"]) == 2
            filenames = {p["source_filename"] for p in results["papers"]}
            assert filenames == {"session_paper_1.pdf", "session_paper_2.pdf"}

            # Verify claims scoping
            assert len(results["claims"]) == 2
            for c in results["claims"]:
                assert c["paper_id"] in paper_ids
                assert c["source_filename"] in filenames

            # Verify pairs scoping
            assert len(results["pairs"]) == 1
            pair = results["pairs"][0]
            assert pair["claim_a_id"] == c1_id
            assert pair["claim_b_id"] == c2_id
            assert pair["validated_as"] == "supports"

            # Verify no original database papers appear
            original_filenames = {
                "P1_jolliffe2024_ari_meta_analysis.pdf",
                "P2_bolland2022_secondary_analysis.pdf",
                "P3_coronavit2022_rct.pdf",
            }
            for fn in original_filenames:
                assert fn not in filenames

            # Verify summary metrics
            summary = results["summary"]
            assert summary["total_papers"] == 2
            assert summary["total_claims"] == 2
            assert summary["total_pairs"] == 1
            assert summary["supports_count"] == 1
            assert summary["contradicts_count"] == 0
            assert summary["total_scores"] == 2
            assert summary["total_graph_edges"] == 1

    def test_single_paper_session_no_cross_comparisons(self):
        """
        Verify that a single PDF upload session processes successfully,
        extracts claims and baseline reliability, without requiring cross-paper comparisons.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_db = Path(tmpdir) / "session_single_paper.db"
            sid, repo = create_session("session_single_test", custom_db_path=tmp_db)

            # Ingest single paper
            p = repo.insert_paper(
                Paper(
                    title="Isolated Study on Plant Immunity",
                    source_filename="single_paper.pdf",
                    source_path=str(Path(tmpdir) / "single_paper.pdf"),
                    page_count=6,
                )
            )
            repo.link_session_paper(sid, p.paper_id, p.source_filename)

            ch = repo.insert_chunks([
                Chunk(paper_id=p.paper_id, page_number=1, char_start=0, char_end=80, text="Stomatal closure is triggered by elevated flg22 peptide concentration.")
            ])[0]

            cid = repo.insert_claim_p2(
                p.paper_id,
                ch.chunk_id,
                type("Claim", (), {"text": "Stomatal closure is triggered by elevated flg22 peptide concentration.", "start": 0, "end": 70})(),
            )
            repo.store_aerf_score(cid, 0.65, {"source": 0.5, "methodology": 0.6}, "v1")

            results = get_session_results(sid, repo=repo)

            assert len(results["papers"]) == 1
            assert results["papers"][0]["source_filename"] == "single_paper.pdf"
            assert len(results["claims"]) == 1
            # 0 cross-paper pairs expected for single paper
            assert len(results["pairs"]) == 0
            assert results["summary"]["total_pairs"] == 0
            assert len(results["scores"]) == 1
            assert results["summary"]["total_scores"] == 1

    def test_strict_query_filtering_in_shared_repository(self):
        """
        Prove that even if multiple papers exist in the same database,
        the session view queries strictly filter by session paper IDs.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_db = Path(tmpdir) / "shared_multi_paper.db"
            repo = Repository(tmp_db)
            repo.initialize_database()
            repo.ensure_phase2_schema()
            repo.ensure_phase3_schema()
            repo.ensure_session_schema()

            # Insert Paper 1 & 2 (belongs to Session A)
            p1 = repo.insert_paper(Paper(source_filename="sessA_p1.pdf", source_path="dummy1", page_count=2))
            p2 = repo.insert_paper(Paper(source_filename="sessA_p2.pdf", source_path="dummy2", page_count=3))

            # Insert Paper 3 (belongs to Session B / outside corpus)
            p3 = repo.insert_paper(Paper(source_filename="outside_corpus_p3.pdf", source_path="dummy3", page_count=4))

            repo.create_session("sess_A")
            repo.link_session_paper("sess_A", p1.paper_id, p1.source_filename)
            repo.link_session_paper("sess_A", p2.paper_id, p2.source_filename)

            # Insert chunks and claims for all 3
            ch1 = repo.insert_chunks([Chunk(paper_id=p1.paper_id, page_number=1, char_start=0, char_end=10, text="Claim text A1")])[0]
            ch2 = repo.insert_chunks([Chunk(paper_id=p2.paper_id, page_number=1, char_start=0, char_end=10, text="Claim text A2")])[0]
            ch3 = repo.insert_chunks([Chunk(paper_id=p3.paper_id, page_number=1, char_start=0, char_end=10, text="Outside claim text")])[0]

            cA1 = repo.insert_claim_p2(p1.paper_id, ch1.chunk_id, type("C", (), {"text": "Claim text A1", "start": 0, "end": 10})())
            cA2 = repo.insert_claim_p2(p2.paper_id, ch2.chunk_id, type("C", (), {"text": "Claim text A2", "start": 0, "end": 10})())
            cOut = repo.insert_claim_p2(p3.paper_id, ch3.chunk_id, type("C", (), {"text": "Outside claim text", "start": 0, "end": 10})())

            # Add pairs
            repo.upsert_claim_pair({
                "claim_a_id": cA1, "claim_b_id": cA2,
                "cosine_similarity": 0.8, "nli_label": "entailment",
                "nli_confidence": 0.9, "entity_overlap_score": 0.5,
                "validated_as": "supports",
            })
            repo.upsert_claim_pair({
                "claim_a_id": cA1, "claim_b_id": cOut,
                "cosine_similarity": 0.85, "nli_label": "contradiction",
                "nli_confidence": 0.85, "entity_overlap_score": 0.4,
                "validated_as": "contradicts",
            })

            # Query Session A results
            results_A = get_session_results("sess_A", repo=repo)

            # Assertions
            session_pids = {p1.paper_id, p2.paper_id}
            assert set(results_A["paper_ids"]) == session_pids
            for claim in results_A["claims"]:
                assert claim["paper_id"] in session_pids
                assert claim["claim_id"] != cOut

            # Claim pair involving outside paper must be filtered out
            for pair in results_A["pairs"]:
                assert pair["claim_a_id"] in {cA1, cA2}
                assert pair["claim_b_id"] in {cA1, cA2}
                assert pair["claim_a_id"] != cOut and pair["claim_b_id"] != cOut

            assert len(results_A["pairs"]) == 1

    def test_session_listing_and_discovery(self):
        """Verify list_sessions retrieves registered and filesystem sessions."""
        sessions = list_sessions()
        assert isinstance(sessions, list)
        for s in sessions:
            assert "session_id" in s
            assert "created_at" in s
            assert "status" in s

    def test_run_session_analysis_workflow_with_real_pdf(self):
        """
        Verify the complete upload-to-results execution workflow using run_session_analysis
        with an isolated database and real sample PDF.
        """
        if not SAMPLE_PDF_PATH.exists():
            pytest.skip("sample_paper.pdf fixture not available.")

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf1 = Path(tmpdir) / "session_doc1.pdf"
            pdf2 = Path(tmpdir) / "session_doc2.pdf"
            shutil.copyfile(SAMPLE_PDF_PATH, pdf1)
            shutil.copyfile(SAMPLE_PDF_PATH, pdf2)

            sid = generate_session_id()
            custom_db = Path(tmpdir) / f"{sid}.db"
            _, repo = create_session(session_id=sid, custom_db_path=custom_db)

            # Mock phase2 & phase3 for fast and deterministic unit testing
            class MockPhase2:
                def __init__(self, r): self.repo = r
                def run(self, paper_id, force=False):
                    chunks = self.repo.get_chunks_by_paper(paper_id)
                    if chunks:
                        cid = self.repo.insert_claim_p2(
                            paper_id, chunks[0].chunk_id,
                            type("Claim", (), {"text": f"Sample claim for paper {paper_id}", "start": 0, "end": 20})()
                        )
                        return {"paper_id": paper_id, "claims": 1}
                    return {"paper_id": paper_id, "claims": 0}

            class MockPhase3:
                def __init__(self, r):
                    self.repo = r
                    self.scorer = type("S", (), {"score": lambda self, cid: type("R", (), {"score": 0.72, "factors": {"source": 0.8}, "weights_version": "v1"})()})()
                    self.updater = type("U", (), {"recompute": lambda *args, **kwargs: []})()
                def run(self, paper_ids=None, paper_id=None):
                    claims = self.repo.get_claims_by_paper_ids(paper_ids)
                    if len(claims) >= 2:
                        self.repo.upsert_claim_pair({
                            "claim_a_id": claims[0]["claim_id"],
                            "claim_b_id": claims[1]["claim_id"],
                            "cosine_similarity": 0.85,
                            "nli_label": "entailment",
                            "nli_confidence": 0.90,
                            "entity_overlap_score": 0.50,
                            "validated_as": "supports",
                            "matching_method": "cosine_nli",
                        })
                    for c in claims:
                        self.repo.store_aerf_score(c["claim_id"], 0.72, {"source": 0.8}, "v1")
                    return {"status": "ok"}

            results = run_session_analysis(
                session_id=sid,
                pdf_paths=[pdf1, pdf2],
                phase2_pipeline=MockPhase2(repo),
                phase3_pipeline=MockPhase3(repo),
                repo=repo,
            )

            assert results["session_id"] == sid
            assert results["summary"]["total_papers"] == 2
            assert results["summary"]["total_claims"] == 2
            assert results["summary"]["total_pairs"] == 1
            assert results["summary"]["supports_count"] == 1
            assert results["summary"]["total_scores"] == 2
            assert results["summary"]["mean_score"] == 0.72


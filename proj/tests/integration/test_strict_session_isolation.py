from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
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

ORIGINAL_DB_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "experiments"
    / "runs"
    / "experiment_FINAL_2026-09-02.db"
)
SAMPLE_PDF_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "sample_paper.pdf"
)


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestStrictSessionIsolation:
    """
    Comprehensive regression test suite proving strict session isolation:
      - Test A: Two-paper isolation (results contain only Paper A and B).
      - Test B: No global-data leakage (claims/papers from 11-paper DB never leak into session).
      - Test C: Session separation (Session 1 with A, B vs Session 2 with C, D - no cross-leakage).
      - Test D: Session switching (switching active session displays only that session's data).
      - Test E: Database integrity (original 11-paper DB is completely unchanged before and after).
      - Test F: Dependency & UI smoke test (st.dataframe serialization & NumPy/PyArrow compatibility).
    """

    def test_a_two_paper_isolation(self):
        """
        Test A: Verify that uploading and analyzing Paper A and Paper B in a new session
        produces results that contain ONLY Paper A and Paper B.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            session_db_path = Path(tmpdir) / "session_ab.db"
            sid, repo = create_session("session_ab_test", custom_db_path=session_db_path)

            p_a = repo.insert_paper(
                Paper(
                    title="Paper A: Cryosphere Dynamics",
                    authors="Alpha et al.",
                    publication_year=2024,
                    source_filename="Paper_A.pdf",
                    source_path=str(Path(tmpdir) / "Paper_A.pdf"),
                    page_count=5,
                )
            )
            p_b = repo.insert_paper(
                Paper(
                    title="Paper B: Glacial Hydrology",
                    authors="Beta et al.",
                    publication_year=2024,
                    source_filename="Paper_B.pdf",
                    source_path=str(Path(tmpdir) / "Paper_B.pdf"),
                    page_count=6,
                )
            )
            repo.link_session_paper(sid, p_a.paper_id, p_a.source_filename)
            repo.link_session_paper(sid, p_b.paper_id, p_b.source_filename)

            ch_a = repo.insert_chunks([
                Chunk(paper_id=p_a.paper_id, page_number=1, char_start=0, char_end=60, text="Subglacial drainage accelerates ice flow during summer melting.")
            ])[0]
            ch_b = repo.insert_chunks([
                Chunk(paper_id=p_b.paper_id, page_number=2, char_start=0, char_end=65, text="Elevated water pressure increases basal sliding velocity in glaciers.")
            ])[0]

            cid_a = repo.insert_claim_p2(
                p_a.paper_id,
                ch_a.chunk_id,
                type("Claim", (), {"text": "Subglacial drainage accelerates ice flow during summer melting.", "start": 0, "end": 60})(),
            )
            cid_b = repo.insert_claim_p2(
                p_b.paper_id,
                ch_b.chunk_id,
                type("Claim", (), {"text": "Elevated water pressure increases basal sliding velocity in glaciers.", "start": 0, "end": 65})(),
            )

            repo.upsert_claim_pair({
                "claim_a_id": cid_a,
                "claim_b_id": cid_b,
                "cosine_similarity": 0.82,
                "nli_label": "entailment",
                "nli_confidence": 0.89,
                "entity_overlap_score": 0.45,
                "validated_as": "supports",
                "matching_method": "cosine_nli",
            })
            repo.store_aerf_score(cid_a, 0.78, {"source": 0.8, "corroboration": 0.7}, "v1")
            repo.store_aerf_score(cid_b, 0.81, {"source": 0.85, "corroboration": 0.75}, "v1")
            repo.upsert_graph_edge("claim", cid_a, "claim", cid_b, "supports", 0.89, 1)

            results = get_session_results(sid, repo=repo)

            # Confirm papers
            assert len(results["papers"]) == 2
            assert {p["source_filename"] for p in results["papers"]} == {"Paper_A.pdf", "Paper_B.pdf"}
            assert set(results["paper_ids"]) == {p_a.paper_id, p_b.paper_id}

            # Confirm claims
            assert len(results["claims"]) == 2
            for c in results["claims"]:
                assert c["paper_id"] in {p_a.paper_id, p_b.paper_id}
                assert c["source_filename"] in {"Paper_A.pdf", "Paper_B.pdf"}

            # Confirm cross-paper pair
            assert len(results["pairs"]) == 1
            pair = results["pairs"][0]
            assert {pair["paper_a_filename"], pair["paper_b_filename"]} == {"Paper_A.pdf", "Paper_B.pdf"}
            assert pair["validated_as"] == "supports"

            # Confirm scores and graph edges
            assert len(results["scores"]) == 2
            assert len(results["graph_edges"]) == 1

    def test_b_no_global_data_leakage(self):
        """
        Test B: Verify that claims and historical data from the global 11-paper corpus
        never appear in or pollute an isolated session, even if a new record is added globally.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            global_db_path = Path(tmpdir) / "global_corpus.db"
            session_db_path = Path(tmpdir) / "session_isolated.db"

            # 1. Setup global DB with 11 papers
            global_repo = Repository(global_db_path)
            global_repo.initialize_database()
            global_repo.ensure_phase2_schema()
            global_repo.ensure_phase3_schema()
            global_repo.ensure_session_schema()

            global_pids = []
            for i in range(1, 12):
                p = global_repo.insert_paper(
                    Paper(
                        title=f"Historical Paper {i}",
                        source_filename=f"historical_paper_{i}.pdf",
                        source_path=f"/tmp/historical_paper_{i}.pdf",
                        page_count=10,
                    )
                )
                global_pids.append(p.paper_id)
                ch = global_repo.insert_chunks([
                    Chunk(paper_id=p.paper_id, page_number=1, char_start=0, char_end=50, text=f"Historical claim from paper {i}")
                ])[0]
                cid = global_repo.insert_claim_p2(
                    p.paper_id, ch.chunk_id,
                    type("Claim", (), {"text": f"Historical claim from paper {i}", "start": 0, "end": 50})()
                )
                global_repo.store_aerf_score(cid, 0.70, {"source": 0.7}, "v1")

            assert global_repo.count_papers() == 11
            assert len(global_repo.get_all_claims()) == 11

            # 2. Setup isolated session DB
            sid, sess_repo = create_session("sess_no_leak", custom_db_path=session_db_path)
            sess_p1 = sess_repo.insert_paper(Paper(source_filename="new_paper_1.pdf", source_path="/tmp/new_paper_1.pdf", page_count=4))
            sess_p2 = sess_repo.insert_paper(Paper(source_filename="new_paper_2.pdf", source_path="/tmp/new_paper_2.pdf", page_count=4))
            sess_repo.link_session_paper(sid, sess_p1.paper_id, sess_p1.source_filename)
            sess_repo.link_session_paper(sid, sess_p2.paper_id, sess_p2.source_filename)

            ch1 = sess_repo.insert_chunks([Chunk(paper_id=sess_p1.paper_id, page_number=1, char_start=0, char_end=30, text="New session claim 1")])[0]
            ch2 = sess_repo.insert_chunks([Chunk(paper_id=sess_p2.paper_id, page_number=1, char_start=0, char_end=30, text="New session claim 2")])[0]
            c1 = sess_repo.insert_claim_p2(sess_p1.paper_id, ch1.chunk_id, type("C", (), {"text": "New session claim 1", "start": 0, "end": 20})())
            c2 = sess_repo.insert_claim_p2(sess_p2.paper_id, ch2.chunk_id, type("C", (), {"text": "New session claim 2", "start": 0, "end": 20})())

            # Add an extra claim to the global database to test potential leakage
            global_leak_chunk = global_repo.insert_chunks([Chunk(paper_id=global_pids[0], page_number=2, char_start=0, char_end=50, text="Leaked claim should never appear in session")])[0]
            global_repo.insert_claim_p2(global_pids[0], global_leak_chunk.chunk_id, type("C", (), {"text": "Leaked claim should never appear in session", "start": 0, "end": 45})())

            # 3. Query session results
            results = get_session_results(sid, repo=sess_repo)

            # Assert: Zero historical papers or claims in session results
            assert len(results["papers"]) == 2
            assert {p["source_filename"] for p in results["papers"]} == {"new_paper_1.pdf", "new_paper_2.pdf"}
            for p in results["papers"]:
                assert "historical" not in p["source_filename"]

            assert len(results["claims"]) == 2
            for c in results["claims"]:
                assert "historical" not in c["text"].lower()
                assert "leaked" not in c["text"].lower()

    def test_c_session_separation(self):
        """
        Test C: Verify that Session 1 (Papers A & B) and Session 2 (Papers C & D)
        are completely isolated from each other with zero cross-session leakage.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            s1_db = Path(tmpdir) / "session_1.db"
            s2_db = Path(tmpdir) / "session_2.db"

            s1_id, r1 = create_session("sess_1", custom_db_path=s1_db)
            s2_id, r2 = create_session("sess_2", custom_db_path=s2_db)

            # Ingest A & B in Session 1
            pA = r1.insert_paper(Paper(source_filename="Paper_A.pdf", source_path="/tmp/Paper_A.pdf", page_count=3))
            pB = r1.insert_paper(Paper(source_filename="Paper_B.pdf", source_path="/tmp/Paper_B.pdf", page_count=3))
            r1.link_session_paper(s1_id, pA.paper_id, pA.source_filename)
            r1.link_session_paper(s1_id, pB.paper_id, pB.source_filename)

            chA = r1.insert_chunks([Chunk(paper_id=pA.paper_id, page_number=1, char_start=0, char_end=20, text="Claim from Paper A")])[0]
            chB = r1.insert_chunks([Chunk(paper_id=pB.paper_id, page_number=1, char_start=0, char_end=20, text="Claim from Paper B")])[0]
            r1.insert_claim_p2(pA.paper_id, chA.chunk_id, type("C", (), {"text": "Claim from Paper A", "start": 0, "end": 18})())
            r1.insert_claim_p2(pB.paper_id, chB.chunk_id, type("C", (), {"text": "Claim from Paper B", "start": 0, "end": 18})())

            # Ingest C & D in Session 2
            pC = r2.insert_paper(Paper(source_filename="Paper_C.pdf", source_path="/tmp/Paper_C.pdf", page_count=3))
            pD = r2.insert_paper(Paper(source_filename="Paper_D.pdf", source_path="/tmp/Paper_D.pdf", page_count=3))
            r2.link_session_paper(s2_id, pC.paper_id, pC.source_filename)
            r2.link_session_paper(s2_id, pD.paper_id, pD.source_filename)

            chC = r2.insert_chunks([Chunk(paper_id=pC.paper_id, page_number=1, char_start=0, char_end=20, text="Claim from Paper C")])[0]
            chD = r2.insert_chunks([Chunk(paper_id=pD.paper_id, page_number=1, char_start=0, char_end=20, text="Claim from Paper D")])[0]
            r2.insert_claim_p2(pC.paper_id, chC.chunk_id, type("C", (), {"text": "Claim from Paper C", "start": 0, "end": 18})())
            r2.insert_claim_p2(pD.paper_id, chD.chunk_id, type("C", (), {"text": "Claim from Paper D", "start": 0, "end": 18})())

            res1 = get_session_results(s1_id, repo=r1)
            res2 = get_session_results(s2_id, repo=r2)

            # Assert Session 1 has only A & B
            assert {p["source_filename"] for p in res1["papers"]} == {"Paper_A.pdf", "Paper_B.pdf"}
            for c in res1["claims"]:
                assert "Paper C" not in c["text"] and "Paper D" not in c["text"]

            # Assert Session 2 has only C & D
            assert {p["source_filename"] for p in res2["papers"]} == {"Paper_C.pdf", "Paper_D.pdf"}
            for c in res2["claims"]:
                assert "Paper A" not in c["text"] and "Paper B" not in c["text"]

    def test_d_session_switching(self):
        """
        Test D: Verify that switching between sessions dynamically updates
        the active context and returns only data belonging to the switched-to session.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            s1_db = Path(tmpdir) / "sess_alpha.db"
            s2_db = Path(tmpdir) / "sess_beta.db"

            s1_id, r1 = create_session("sess_alpha", custom_db_path=s1_db)
            s2_id, r2 = create_session("sess_beta", custom_db_path=s2_db)

            p1 = r1.insert_paper(Paper(source_filename="Alpha_Paper.pdf", source_path="/tmp/Alpha_Paper.pdf", page_count=2))
            r1.link_session_paper(s1_id, p1.paper_id, p1.source_filename)
            ch1 = r1.insert_chunks([Chunk(paper_id=p1.paper_id, page_number=1, char_start=0, char_end=15, text="Alpha claim 1")])[0]
            r1.insert_claim_p2(p1.paper_id, ch1.chunk_id, type("C", (), {"text": "Alpha claim 1", "start": 0, "end": 13})())

            p2 = r2.insert_paper(Paper(source_filename="Beta_Paper.pdf", source_path="/tmp/Beta_Paper.pdf", page_count=2))
            r2.link_session_paper(s2_id, p2.paper_id, p2.source_filename)
            ch2 = r2.insert_chunks([Chunk(paper_id=p2.paper_id, page_number=1, char_start=0, char_end=15, text="Beta claim 1")])[0]
            r2.insert_claim_p2(p2.paper_id, ch2.chunk_id, type("C", (), {"text": "Beta claim 1", "start": 0, "end": 12})())

            # Simulate UI state switching
            session_state = {"active_session_id": s1_id, "session_view_mode": "view"}
            active_res = get_session_results(session_state["active_session_id"], repo=r1)
            assert active_res["papers"][0]["source_filename"] == "Alpha_Paper.pdf"

            # Switch to Beta
            session_state["active_session_id"] = s2_id
            active_res_switched = get_session_results(session_state["active_session_id"], repo=r2)
            assert active_res_switched["papers"][0]["source_filename"] == "Beta_Paper.pdf"
            assert "Alpha_Paper.pdf" not in [p["source_filename"] for p in active_res_switched["papers"]]

            # Switch to New Session mode (unselected / empty state)
            session_state["active_session_id"] = None
            session_state["session_view_mode"] = "new"
            assert session_state["active_session_id"] is None

    def test_e_database_integrity(self):
        """
        Test E: Verify that the original 11-paper database file is bit-for-bit identical
        and unmodified before and after session creation, ingestion, and analysis.
        """
        if not ORIGINAL_DB_PATH.exists():
            pytest.skip("Original experiment database not present.")

        before_hash = _compute_sha256(ORIGINAL_DB_PATH)
        orig_repo = Repository(ORIGINAL_DB_PATH)
        before_paper_count = orig_repo.count_papers()
        before_claim_count = len(orig_repo.get_all_claims())
        assert before_paper_count == 11

        # Perform a complete session workflow
        with tempfile.TemporaryDirectory() as tmpdir:
            test_db = Path(tmpdir) / "integrity_test_session.db"
            sid, repo = create_session("integrity_test", custom_db_path=test_db)

            p1 = repo.insert_paper(Paper(source_filename="test1.pdf", source_path="/tmp/test1.pdf", page_count=2))
            p2 = repo.insert_paper(Paper(source_filename="test2.pdf", source_path="/tmp/test2.pdf", page_count=2))
            repo.link_session_paper(sid, p1.paper_id, p1.source_filename)
            repo.link_session_paper(sid, p2.paper_id, p2.source_filename)

            ch1 = repo.insert_chunks([Chunk(paper_id=p1.paper_id, page_number=1, char_start=0, char_end=20, text="Integrity test claim")])[0]
            repo.insert_claim_p2(p1.paper_id, ch1.chunk_id, type("C", (), {"text": "Integrity test claim", "start": 0, "end": 20})())
            results = get_session_results(sid, repo=repo)
            assert len(results["papers"]) == 2

        # Verify original database integrity
        after_hash = _compute_sha256(ORIGINAL_DB_PATH)
        assert before_hash == after_hash, "CRITICAL ERROR: Original 11-paper database hash changed!"

        after_orig_repo = Repository(ORIGINAL_DB_PATH)
        assert after_orig_repo.count_papers() == before_paper_count == 11
        assert len(after_orig_repo.get_all_claims()) == before_claim_count

    def test_f_dependency_and_ui_smoke_test(self):
        """
        Test F: Dependency & UI smoke test ensuring that Pandas DataFrame serialization
        to Arrow bytes in Streamlit runs without the 'ImportError: pyarrow requires NumPy 2.0 or newer' error.
        """
        import numpy as np
        import pyarrow as pa
        import streamlit as st
        from streamlit.dataframe_util import convert_anything_to_arrow_bytes

        # Verify versions
        assert np.__version__ == "1.26.4", f"Expected NumPy 1.26.4, got {np.__version__}"
        assert pa.__version__.startswith("17."), f"Expected PyArrow 17.x, got {pa.__version__}"

        # Test serializing DataFrames resembling session comparison tables
        df_papers = pd.DataFrame([
            {"Paper ID": 1, "Filename": "Paper_A.pdf", "Title": "Title A", "Extracted Claims": 5},
            {"Paper ID": 2, "Filename": "Paper_B.pdf", "Title": "Title B", "Extracted Claims": 7},
        ])
        arrow_bytes_papers = convert_anything_to_arrow_bytes(df_papers)
        assert len(arrow_bytes_papers) > 0

        df_claims = pd.DataFrame([
            {"Claim ID": "CLM01", "Paper": "Paper_A.pdf", "Page": 1, "Claim Text": "Glacial ice velocity increases."},
            {"Claim ID": "CLM02", "Paper": "Paper_B.pdf", "Page": 3, "Claim Text": "Basal lubrication drives speedup."},
        ])
        arrow_bytes_claims = convert_anything_to_arrow_bytes(df_claims)
        assert len(arrow_bytes_claims) > 0

        df_scores = pd.DataFrame([
            {
                "Claim ID": "CLM01",
                "AERF Score": "0.7820",
                "Source": "0.80",
                "Methodology": "0.75",
                "Corroboration": "0.60",
                "Contradiction": "0.10",
                "Temporal": "0.50",
                "Semantic": "0.85",
                "Provenance": "1.00",
            }
        ])
        arrow_bytes_scores = convert_anything_to_arrow_bytes(df_scores)
        assert len(arrow_bytes_scores) > 0

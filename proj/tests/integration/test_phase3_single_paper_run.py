"""
Regression test for pipeline/phase3.py's Phase3Pipeline.run(paper_id=...).

Bug this guards against: the candidate-generation loop restricted `a` to
the requested paper_id but still only paired it with `b` in
all_claims[i+1:] (claims with a HIGHER claim_id / later in the list).
Since claim_id order follows ingestion/Phase-2 order, running
run(paper_id=<a paper ingested/Phase-2'd EARLIER than others>) silently
dropped every comparison against papers with LOWER claim_ids -- and
running it for whichever paper happens to have been processed LAST
produced zero candidate pairs at all, even when genuine cross-document
matches existed. This is exactly the single-paper re-run path
main.py --phase3-paper-id N and FINAL_RUN_GUIDE.md's documented workflow
both use.

Uses a monkeypatched NLIClassifier._load (same technique as this
project's other tests that need Phase3Pipeline's real, unmodified
candidate-generation/matching logic without a network-dependent model
download -- Phase3Pipeline always constructs its own real NLIClassifier
internally, so there is no constructor-injection point to use instead).
The fake always returns a confident 'entailment' so gate-passing pairs are
unambiguously counted; it does not change any threshold or the matching
decision logic itself.
"""
import numpy as np
import pytest

from db.repository import Repository
from db.models import Paper, Chunk
import nlp.nli.nli_classifier as nli_module
from pipeline.phase3 import Phase3Pipeline


class _FakeHFPipeline:
    def __call__(self, premise, text_pair, top_k=None):
        return [[{"label": "LABEL_1", "score": 0.95},
                 {"label": "LABEL_0", "score": 0.03},
                 {"label": "LABEL_2", "score": 0.02}]]


@pytest.fixture
def fake_nli(monkeypatch):
    def _load(self):
        self._classifier = _FakeHFPipeline()
        return self._classifier
    monkeypatch.setattr(nli_module.NLIClassifier, "_load", _load)


def _make_three_paper_repo(tmp_path):
    """Papers P1, P2, P3 ingested and Phase-2'd in that order, so their
    claim_ids increase in the same order (P1 lowest, P3 highest) -- this
    ordering is what the bug depended on. All three claims get identical
    embeddings and no extracted entities, so every cross-pair clears both
    the similarity gate and the no-entities similarity fallback for
    "aligned", isolating the test from anything to do with entity overlap
    or embedding quality."""
    r = Repository(tmp_path / "a.db")
    r.initialize_database()
    r.ensure_phase2_schema()
    r.ensure_phase3_schema()

    class C:
        pass

    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    claim_ids = {}
    for label in ("P1", "P2", "P3"):
        paper = r.insert_paper(Paper(f"{label}.pdf", f"{label}.pdf", 1, publication_year=2024, venue="J"))
        text = f"Vitamin D reduces risk according to {label}."
        r.insert_chunks([Chunk(paper.paper_id, 1, 0, len(text), text)])
        c = C(); c.text = text; c.start = 0; c.end = len(text)
        claim_id = r.insert_claim_p2(paper.paper_id, 1, c)
        r.upsert_embedding_p2(claim_id, "fake", vec)
        claim_ids[label] = (paper.paper_id, claim_id)
    return r, claim_ids


def test_run_for_the_last_ingested_paper_still_finds_earlier_cross_pairs(tmp_path, fake_nli):
    repo, claim_ids = _make_three_paper_repo(tmp_path)
    p3_paper_id, p3_claim_id = claim_ids["P3"]

    result = Phase3Pipeline(repo).run(paper_id=p3_paper_id)

    # P3 has the HIGHEST claim_id of the three -- this is exactly the case
    # the bug silently produced zero pairs for.
    assert result["candidate_relationships_persisted"] == 2, (
        "Expected P3 to be matched against both P1 and P2 (both have lower "
        "claim_ids); a result of 0 means the pre-fix bug has regressed."
    )

    with repo._connect() as conn:
        rows = conn.execute(
            "SELECT claim_a_id, claim_b_id FROM claim_pairs WHERE claim_a_id=? OR claim_b_id=?",
            (p3_claim_id, p3_claim_id),
        ).fetchall()
    partners = {r["claim_a_id"] if r["claim_b_id"] == p3_claim_id else r["claim_b_id"] for r in rows}
    assert partners == {claim_ids["P1"][1], claim_ids["P2"][1]}


def test_run_for_the_first_ingested_paper_still_finds_later_cross_pairs(tmp_path, fake_nli):
    """Sanity check in the other direction -- this direction worked even
    before the fix, so this test mainly guards against a fix that
    accidentally breaks it."""
    repo, claim_ids = _make_three_paper_repo(tmp_path)
    p1_paper_id, p1_claim_id = claim_ids["P1"]

    result = Phase3Pipeline(repo).run(paper_id=p1_paper_id)

    assert result["candidate_relationships_persisted"] == 2
    with repo._connect() as conn:
        rows = conn.execute(
            "SELECT claim_a_id, claim_b_id FROM claim_pairs WHERE claim_a_id=? OR claim_b_id=?",
            (p1_claim_id, p1_claim_id),
        ).fetchall()
    partners = {r["claim_a_id"] if r["claim_b_id"] == p1_claim_id else r["claim_b_id"] for r in rows}
    assert partners == {claim_ids["P2"][1], claim_ids["P3"][1]}


def test_single_paper_run_never_pairs_claims_from_the_same_paper(tmp_path, fake_nli):
    repo, claim_ids = _make_three_paper_repo(tmp_path)
    p3_paper_id = claim_ids["P3"][0]
    Phase3Pipeline(repo).run(paper_id=p3_paper_id)

    with repo._connect() as conn:
        rows = conn.execute(
            "SELECT cp.claim_a_id, cp.claim_b_id, ca.paper_id AS pa, cb.paper_id AS pb "
            "FROM claim_pairs cp "
            "JOIN claims ca ON ca.claim_id = cp.claim_a_id "
            "JOIN claims cb ON cb.claim_id = cp.claim_b_id"
        ).fetchall()
    assert all(r["pa"] != r["pb"] for r in rows)


def test_corpus_wide_run_still_finds_every_cross_pair_exactly_once(tmp_path, fake_nli):
    """Preserves existing corpus-wide (paper_id=None) behavior: with 3
    papers there are exactly 3 unordered cross-paper pairs
    (P1-P2, P1-P3, P2-P3), each persisted exactly once."""
    repo, claim_ids = _make_three_paper_repo(tmp_path)

    result = Phase3Pipeline(repo).run(paper_id=None)

    assert result["candidate_relationships_persisted"] == 3
    with repo._connect() as conn:
        n = conn.execute("SELECT COUNT(*) n FROM claim_pairs").fetchone()["n"]
    assert n == 3  # no duplicate rows for the same unordered pair


def test_corpus_wide_run_matches_the_union_of_all_single_paper_runs(tmp_path, fake_nli):
    """The two call styles (one corpus-wide run vs. one run per paper, as
    main.py --phase3-paper-id N is used for each paper in
    FINAL_RUN_GUIDE.md) must produce the same set of validated pairs."""
    repo, claim_ids = _make_three_paper_repo(tmp_path)
    for label in ("P1", "P2", "P3"):
        Phase3Pipeline(repo).run(paper_id=claim_ids[label][0])

    with repo._connect() as conn:
        pairs_from_individual_runs = {
            tuple(sorted((r["claim_a_id"], r["claim_b_id"])))
            for r in conn.execute("SELECT claim_a_id, claim_b_id FROM claim_pairs").fetchall()
        }
    assert pairs_from_individual_runs == {
        tuple(sorted((claim_ids["P1"][1], claim_ids["P2"][1]))),
        tuple(sorted((claim_ids["P1"][1], claim_ids["P3"][1]))),
        tuple(sorted((claim_ids["P2"][1], claim_ids["P3"][1]))),
    }


def test_running_both_papers_in_either_order_leaves_exactly_one_logical_pair(tmp_path, fake_nli):
    """Regression test for the orientation-duplication bug: Phase3Pipeline
    always puts the paper_id-restricted paper's claim in claim_a_id (see
    pipeline/phase3.py's target_claims/other_claims split), so
    run(paper_id=P1) stores (P1_claim, P2_claim) while a LATER
    run(paper_id=P2) computes the same logical relationship as
    (P2_claim, P1_claim) -- the reverse orientation. Without an unordered
    dedup in Repository.upsert_claim_pair(), that reverse call inserted a
    second row instead of replacing the first, since claim_pairs has no
    unordered uniqueness constraint (see db/schema.sql) and the delete
    used to match only the exact orientation.

    Runs P1 then P2, and separately P2 then P1, and asserts exactly one
    claim_pairs row exists for that unordered pair in both cases."""
    for first, second in (("P1", "P2"), ("P2", "P1")):
        # Each iteration needs its own fresh database -- _make_three_paper_repo
        # uses a fixed filename ("a.db") and fixed paper filenames, so reusing
        # tmp_path directly across iterations would hit the papers table's
        # UNIQUE(source_filename) constraint on the second iteration. That's a
        # test-isolation detail, unrelated to the bug under test.
        sub = tmp_path / f"{first}_then_{second}"
        sub.mkdir()
        repo, claim_ids = _make_three_paper_repo(sub)
        # Only P1 and P2 matter for this pair; run() also touches P3, which
        # is irrelevant to the specific (P1,P2) orientation being tested.
        Phase3Pipeline(repo).run(paper_id=claim_ids[first][0])
        Phase3Pipeline(repo).run(paper_id=claim_ids[second][0])

        p1_claim, p2_claim = claim_ids["P1"][1], claim_ids["P2"][1]
        with repo._connect() as conn:
            rows = conn.execute(
                "SELECT pair_id, claim_a_id, claim_b_id FROM claim_pairs "
                "WHERE (claim_a_id=? AND claim_b_id=?) OR (claim_a_id=? AND claim_b_id=?)",
                (p1_claim, p2_claim, p2_claim, p1_claim),
            ).fetchall()

        assert len(rows) == 1, (
            f"Running {first} then {second} left {len(rows)} row(s) for the P1/P2 pair "
            f"(expected exactly 1) -- orientation-based duplication has regressed: {rows}"
        )


def test_rerunning_the_same_paper_is_idempotent_for_logical_pairs(tmp_path, fake_nli):
    """The simpler idempotency case: calling run(paper_id=X) twice in a row
    (same orientation both times) must not duplicate rows either -- this
    already worked via the exact-orientation delete, but is asserted here
    explicitly since it's part of the same idempotency requirement."""
    repo, claim_ids = _make_three_paper_repo(tmp_path)
    p3_paper_id = claim_ids["P3"][0]

    Phase3Pipeline(repo).run(paper_id=p3_paper_id)
    Phase3Pipeline(repo).run(paper_id=p3_paper_id)

    with repo._connect() as conn:
        n = conn.execute("SELECT COUNT(*) n FROM claim_pairs").fetchone()["n"]
    assert n == 2  # P3-P1 and P3-P2, each exactly once

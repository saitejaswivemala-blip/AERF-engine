"""
Regression test for scripts/evaluate_curated_mapping.py's evidence
persistence step.

Bug this guards against: evaluate_configuration() used to hardcode
"validated_as": None when writing real NLI evidence into claim_pairs,
which made that evidence permanently invisible to
AERFScorer.get_validated_neighbors() (corroboration/contradiction always
computed as 0.0 regardless of how confident the real NLI result was).
Fixed to use record.predicted_label -- the already-computed, correctly
similarity/entity-alignment/NLI-threshold-gated classification -- instead.

Uses a FakeNLI (same pattern as tests/experiments/test_pipeline_runner.py)
since this test must not require network/model access; it exercises the
real, unmodified run_matching() -> evaluate_configuration() ->
AERFScorer chain around that fake, which is exactly the chain the bug
lived in.
"""
import numpy as np

from db.repository import Repository
from db.models import Paper, Chunk
from nlp.nli.nli_classifier import NLIResult
from aerf.scorer import AERFScorer
from experiments.lib.experiment_config import FULL_AERF

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "evaluate_curated_mapping",
    Path(__file__).resolve().parents[2] / "scripts" / "evaluate_curated_mapping.py",
)
evaluate_curated_mapping = importlib.util.module_from_spec(_SPEC)
sys.modules["evaluate_curated_mapping"] = evaluate_curated_mapping
_SPEC.loader.exec_module(evaluate_curated_mapping)
evaluate_configuration = evaluate_curated_mapping.evaluate_configuration


class FakeNLI:
    """Same pattern as tests/experiments/test_pipeline_runner.py's FakeNLI."""
    def __init__(self, label="entailment", confidence=0.9):
        self.label = label
        self.confidence = confidence
        self.model_name = "fake-nli-for-test"

    def predict_bidirectional(self, a, b):
        e = c = n = 0.0
        if self.label == "entailment":
            e = self.confidence
        elif self.label == "contradiction":
            c = self.confidence
        else:
            n = self.confidence
        return NLIResult(self.label, e, c, n, self.confidence, "fake-nli")


def make_repo_with_confidently_supporting_pair(tmp_path):
    """Two claims with no extracted entities (so the entity-overlap gate
    falls back to its no-entities similarity path) and identical dummy
    embeddings (cosine similarity 1.0) -- guaranteed to pass FULL_AERF's
    similarity and entity-alignment gates, so a confident FakeNLI result
    should end up as a real 'supports' classification, not 'uncertain'."""
    r = Repository(tmp_path / "a.db")
    r.initialize_database()
    r.ensure_phase2_schema()
    r.ensure_phase3_schema()
    p1 = r.insert_paper(Paper("a.pdf", "a.pdf", 1, publication_year=2024, venue="J"))
    p2 = r.insert_paper(Paper("b.pdf", "b.pdf", 1, publication_year=2024, venue="J"))
    r.insert_chunks([Chunk(p1.paper_id, 1, 0, 20, "Vitamin D reduces risk.")])
    r.insert_chunks([Chunk(p2.paper_id, 1, 0, 20, "Vitamin D lowers risk.")])

    class C:
        pass
    c1 = C(); c1.text = "Vitamin D reduces risk."; c1.start = 0; c1.end = len(c1.text)
    c2 = C(); c2.text = "Vitamin D lowers risk."; c2.start = 0; c2.end = len(c2.text)
    a = r.insert_claim_p2(p1.paper_id, 1, c1)
    b = r.insert_claim_p2(p2.paper_id, 2, c2)
    for cid in (a, b):
        r.upsert_embedding_p2(cid, "fake", np.array([1, 0, 0], dtype=np.float32))
    return r, a, b


def _entry_and_pair(repo, a, b):
    claims = {c["claim_id"]: c for c in repo.get_all_claims()}
    entry = {"pair_id": "test_pair", "paper_a": "P1", "paper_b": "P2",
              "claim_a_id": a, "claim_b_id": b, "human_label": "supports"}
    return [entry], [(claims[a], claims[b])]


def test_evaluate_configuration_persists_real_validated_as_not_none(tmp_path):
    repo, a, b = make_repo_with_confidently_supporting_pair(tmp_path)
    entries, pairs = _entry_and_pair(repo, a, b)
    nli = FakeNLI("entailment", 0.9)

    result = evaluate_configuration(repo, FULL_AERF, nli, entries, pairs)

    assert result["predictions"][0]["system_prediction"] == "supports"

    with repo._connect() as conn:
        row = conn.execute(
            "SELECT validated_as FROM claim_pairs WHERE claim_a_id=? AND claim_b_id=?", (a, b)
        ).fetchone()
    assert row is not None, "evaluate_configuration must persist a claim_pairs row for a confident match"
    assert row["validated_as"] == "supports"  # NOT None -- this is the bug this test guards against


def test_evaluate_configuration_persisted_evidence_is_visible_to_aerf_scoring(tmp_path):
    """End-to-end regression: after evaluate_configuration() runs, a fresh
    AERFScorer.factors() call on claim `a` must see nonzero corroboration
    -- proving the persisted validated_as is not just present in the row
    but actually consumed by get_validated_neighbors()."""
    repo, a, b = make_repo_with_confidently_supporting_pair(tmp_path)
    entries, pairs = _entry_and_pair(repo, a, b)
    nli = FakeNLI("entailment", 0.9)

    evaluate_configuration(repo, FULL_AERF, nli, entries, pairs)

    scorer = AERFScorer(repo)
    factors = scorer.factors(a)
    assert factors["corroboration"] > 0.0, (
        "Real, confident NLI evidence was persisted but AERFScorer still computed "
        "corroboration=0.0 -- validated_as is not reaching get_validated_neighbors()."
    )


def test_evaluate_configuration_does_not_write_human_label_into_validated_as(tmp_path):
    """Guards the actual distinction the original (buggy) code was trying,
    incorrectly, to protect: validated_as must be the system's own
    classification, and must never be set to (or derived from) the human
    ground-truth label, which is a completely separate field."""
    repo, a, b = make_repo_with_confidently_supporting_pair(tmp_path)
    entries, pairs = _entry_and_pair(repo, a, b)
    entries[0]["human_label"] = "contradicts"  # deliberately different from what NLI will say
    nli = FakeNLI("entailment", 0.9)  # NLI says supports

    evaluate_configuration(repo, FULL_AERF, nli, entries, pairs)

    with repo._connect() as conn:
        row = conn.execute(
            "SELECT validated_as FROM claim_pairs WHERE claim_a_id=? AND claim_b_id=?", (a, b)
        ).fetchone()
    # validated_as must reflect the system's real classification (supports,
    # from NLI), never the human_label (contradicts) -- proving no leakage
    # of ground truth into the evidence field in either direction.
    assert row["validated_as"] == "supports"

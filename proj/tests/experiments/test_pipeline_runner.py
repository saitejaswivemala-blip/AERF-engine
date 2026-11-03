import numpy as np
import pytest

from db.repository import Repository
from db.models import Paper, Chunk
from nlp.nli.nli_classifier import NLIResult
from experiments.lib.experiment_config import (
    BASELINE_A_SIMILARITY_ONLY, BASELINE_B_SIMILARITY_NLI,
    BASELINE_C_SIMILARITY_ENTITY_NLI, FULL_AERF,
    ABLATION_WITHOUT_ENTITY_ALIGNMENT,
)
from experiments.lib.pipeline_runner import run_matching, score_claims_with_config


class FakeNLI:
    """Same pattern as tests/unit/test_phase3_reasoning.py's FakeNLI."""
    def __init__(self, label="contradiction", confidence=0.9):
        self.label = label
        self.confidence = confidence

    def predict_bidirectional(self, a, b):
        e = c = n = 0.0
        if self.label == "entailment":
            e = self.confidence
        elif self.label == "contradiction":
            c = self.confidence
        else:
            n = self.confidence
        return NLIResult(self.label, e, c, n, self.confidence, "fake-nli")


def make_repo(tmp_path, entities_overlap=True):
    r = Repository(tmp_path / "a.db")
    r.initialize_database()
    r.ensure_phase2_schema()
    r.ensure_phase3_schema()
    p1 = r.insert_paper(Paper("a.pdf", "a.pdf", 1, publication_year=2024, venue="J"))
    p2 = r.insert_paper(Paper("b.pdf", "b.pdf", 1, publication_year=2024, venue="J"))
    r.insert_chunks([Chunk(p1.paper_id, 1, 0, 20, "A increases X strongly.")])
    r.insert_chunks([Chunk(p2.paper_id, 1, 0, 20, "A decreases X strongly.")])

    class C:
        pass
    c1 = C(); c1.text = "A increases X strongly."; c1.start = 0; c1.end = len(c1.text)
    c2 = C(); c2.text = "A decreases X strongly."; c2.start = 0; c2.end = len(c2.text)
    a = r.insert_claim_p2(p1.paper_id, 1, c1)
    b = r.insert_claim_p2(p2.paper_id, 2, c2)
    for cid in (a, b):
        r.upsert_embedding_p2(cid, "fake", np.array([1, 0, 0], dtype=np.float32))
    return r, a, b


def get_claim_dicts(repo, a, b):
    claims = {c["claim_id"]: c for c in repo.get_all_claims()}
    return claims[a], claims[b]


def test_baseline_a_labels_purely_from_similarity_no_nli_called(tmp_path):
    repo, a, b = make_repo(tmp_path)
    ca, cb = get_claim_dicts(repo, a, b)

    class ExplodingNLI:
        def predict_bidirectional(self, x, y):
            raise AssertionError("Baseline A must never call NLI")

    records = run_matching(repo, BASELINE_A_SIMILARITY_ONLY, ExplodingNLI(), [(ca, cb)])
    assert len(records) == 1
    assert records[0].predicted_label in ("supports", "neutral")
    assert records[0].nli_label is None


def test_baseline_b_skips_entity_gate_but_uses_nli(tmp_path):
    repo, a, b = make_repo(tmp_path)
    ca, cb = get_claim_dicts(repo, a, b)
    records = run_matching(repo, BASELINE_B_SIMILARITY_NLI, FakeNLI("contradiction", 0.95), [(ca, cb)])
    assert records[0].predicted_label == "contradicts"
    assert records[0].entity_overlap_score is None  # gate not applied


def test_baseline_c_applies_entity_gate_and_nli(tmp_path):
    repo, a, b = make_repo(tmp_path)
    ca, cb = get_claim_dicts(repo, a, b)
    records = run_matching(repo, BASELINE_C_SIMILARITY_ENTITY_NLI, FakeNLI("contradiction", 0.95), [(ca, cb)])
    assert records[0].entity_overlap_score is not None


def test_similarity_below_threshold_is_uncertain_regardless_of_config(tmp_path):
    repo, a, b = make_repo(tmp_path)
    ca, cb = get_claim_dicts(repo, a, b)
    strict = FULL_AERF.__class__(**{**vars(FULL_AERF), "similarity_threshold": 1.1})
    records = run_matching(repo, strict, FakeNLI("contradiction", 0.95), [(ca, cb)])
    assert records[0].predicted_label == "uncertain"
    assert records[0].nli_label is None  # NLI never called below the similarity gate


def test_ablation_without_entity_alignment_never_gates_on_entities(tmp_path):
    repo, a, b = make_repo(tmp_path)
    ca, cb = get_claim_dicts(repo, a, b)
    records = run_matching(repo, ABLATION_WITHOUT_ENTITY_ALIGNMENT, FakeNLI("entailment", 0.95), [(ca, cb)])
    assert records[0].predicted_label == "supports"
    assert records[0].entity_overlap_score is None


def test_score_with_config_zero_provenance_weight_excludes_that_factor(tmp_path):
    repo, a, b = make_repo(tmp_path)
    from experiments.lib.experiment_config import ABLATION_WITHOUT_PROVENANCE_WEIGHTING
    scores = score_claims_with_config(repo, ABLATION_WITHOUT_PROVENANCE_WEIGHTING, [a, b])
    assert a in scores and 0 <= scores[a] <= 1


def test_score_with_config_respects_use_aerf_scoring_false(tmp_path):
    repo, a, b = make_repo(tmp_path)
    from experiments.lib.experiment_config import BASELINE_A_SIMILARITY_ONLY
    scores = score_claims_with_config(repo, BASELINE_A_SIMILARITY_ONLY, [a, b])
    assert scores == {}

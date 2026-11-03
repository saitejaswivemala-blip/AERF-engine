import numpy as np

from scripts.evaluate_reviewed_mapping import evaluate_mapping, summary


class FakeRepository:
    def __init__(self, claims, embeddings=None, entities=None, relations=None):
        self.claims = claims
        self.embeddings = embeddings or {}
        self.entities = entities or {}
        self.relations = relations or {}

    def get_claim(self, claim_id):
        return self.claims.get(claim_id)

    def get_claim_embedding(self, claim_id):
        return self.embeddings.get(claim_id)

    def get_claim_entities(self, claim_id):
        return self.entities.get(claim_id, [])

    def get_claim_relations(self, claim_id):
        return self.relations.get(claim_id, [])


class FakeDecision:
    claim_a_id = 1
    claim_b_id = 2
    cosine_similarity = 0.9
    entity_overlap_score = 1.0
    relation_alignment_score = 0.0
    nli_label = "entailment"
    nli_confidence = 0.95
    nli_entailment = 0.95
    nli_contradiction = 0.02
    nli_neutral = 0.03
    validated_as = "supports"


class FakeMatcher:
    similarity_threshold = 0.55
    entity_threshold = 0.2

    def __init__(self, decision=None):
        self.decision = decision
        self.calls = 0

    def match(self, *args):
        self.calls += 1
        return self.decision


def cross_claims():
    return {
        1: {"claim_id": 1, "paper_id": 1, "text": "claim one"},
        2: {"claim_id": 2, "paper_id": 2, "text": "claim two"},
    }


def cross_mapping(pair_id="gt_01"):
    return {pair_id: {"claim_a": {"db_claim_id": 1}, "claim_b": {"db_claim_id": 2}}}


def test_classified_match_is_compared_without_persistence():
    repo = FakeRepository(cross_claims(), {1: np.array([1.0, 0.0]), 2: np.array([1.0, 0.0])})
    matcher = FakeMatcher(FakeDecision())
    annotations = [{"pair_id": "gt_01", "human_label": "supports"}]
    rows = evaluate_mapping(repo, annotations, cross_mapping(), matcher)

    assert rows[0]["stage"] == "nli/classified"
    assert rows[0]["predicted_relationship"] == "supports"
    assert rows[0]["nli_confidence"] == 0.95
    assert rows[0]["correct"] is True
    assert matcher.calls == 1


def test_missing_embedding_is_reported_without_calling_matcher():
    repo = FakeRepository(cross_claims(), {1: np.array([1.0, 0.0])})
    matcher = FakeMatcher(FakeDecision())
    annotations = [{"pair_id": "gt_02", "human_label": "contradicts"}]

    row = evaluate_mapping(repo, annotations, cross_mapping("gt_02"), matcher)[0]

    assert row["stage"] == "missing_embedding"
    assert row["correct"] is None
    assert matcher.calls == 0


def test_cosine_gate_is_reported_without_calling_matcher():
    repo = FakeRepository(cross_claims(), {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])})
    matcher = FakeMatcher(FakeDecision())

    row = evaluate_mapping(repo, [{"pair_id": "gt_03", "human_label": "supports"}],
                            cross_mapping("gt_03"), matcher)[0]

    assert row["stage"] == "cosine_rejected"
    assert matcher.calls == 0


def test_entity_gate_is_reported_without_calling_matcher():
    repo = FakeRepository(
        cross_claims(),
        {1: np.array([1.0, 0.0]), 2: np.array([1.0, 0.0])},
        {1: [{"text": "alpha"}], 2: [{"text": "beta"}]},
    )
    matcher = FakeMatcher(FakeDecision())

    row = evaluate_mapping(repo, [{"pair_id": "gt_04", "human_label": "supports"}],
                            cross_mapping("gt_04"), matcher)[0]

    assert row["stage"] == "entity_alignment_rejected"
    assert matcher.calls == 0


def test_null_reviewed_side_is_unresolved_and_excluded():
    repo = FakeRepository({1: {"claim_id": 1, "paper_id": 1}})
    annotations = [{"pair_id": "gt_03", "human_label": "supports"}]
    mapping = {"gt_03": {"claim_a": {"db_claim_id": 1}, "claim_b": {"db_claim_id": None}}}

    rows = evaluate_mapping(repo, annotations, mapping)

    assert rows[0]["status"] == "UNRESOLVED"
    assert rows[0]["correct"] is None
    assert summary(rows) == {
        "total_pairs": 1,
        "fully_mapped": 0,
        "evaluated_cross_paper": 0,
        "same_paper_out_of_scope": 0,
        "unresolved": 1,
        "classified": 0,
        "correct": 0,
        "incorrect": 0,
        "accuracy_over_evaluated": None,
    }


def test_same_paper_is_out_of_scope_without_calling_matcher():
    claims = {
        1: {"claim_id": 1, "paper_id": 7, "text": "claim one"},
        2: {"claim_id": 2, "paper_id": 7, "text": "claim two"},
    }
    matcher = FakeMatcher(FakeDecision())

    row = evaluate_mapping(
        FakeRepository(claims),
        [{"pair_id": "gt_05", "human_label": "contradicts"}],
        cross_mapping("gt_05"),
        matcher,
    )[0]

    assert row["status"] == "OUT_OF_SCOPE_SAME_PAPER"
    assert row["stage"] == "out_of_scope_same_paper"
    assert matcher.calls == 0


def test_summary_excludes_unresolved_from_accuracy():
    rows = [
        {"status": "EVALUATED", "stage": "nli/classified", "correct": True,
         "claim_a_id": 1, "claim_b_id": 2},
        {"status": "EVALUATED", "stage": "cosine_rejected", "correct": None,
         "claim_a_id": 3, "claim_b_id": 4},
        {"status": "EVALUATED", "stage": "nli/classified", "correct": False,
         "claim_a_id": 5, "claim_b_id": 6},
        {"status": "UNRESOLVED", "correct": None, "claim_a_id": None, "claim_b_id": None},
        {"status": "OUT_OF_SCOPE_SAME_PAPER", "correct": None,
         "claim_a_id": 7, "claim_b_id": 8},
    ]

    result = summary(rows)

    assert result["total_pairs"] == 5
    assert result["fully_mapped"] == 4
    assert result["evaluated_cross_paper"] == 3
    assert result["same_paper_out_of_scope"] == 1
    assert result["unresolved"] == 1
    assert result["classified"] == 2
    assert result["correct"] == 1
    assert result["incorrect"] == 1
    assert result["accuracy_over_evaluated"] == 0.5

"""
Regression guard: ClaimMatcher's DEFAULT behavior (entity_alignment_version
not passed, i.e. "v1") must remain exactly what it was before
entity_alignment.py existed. Every previously-documented result
(README.md, experiment reports, curated_comparison_real_models.json) was
produced under v1 -- if this test ever fails, those documented numbers are
no longer reproducible from the current code.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from matching.claim_matcher import ClaimMatcher


def _fake_nli():
    nli = MagicMock()
    result = MagicMock()
    result.label = "entailment"
    result.confidence = 0.9
    result.entailment = 0.9
    result.contradiction = 0.05
    result.neutral = 0.05
    nli.predict_bidirectional.return_value = result
    return nli


def test_default_entity_alignment_version_is_v1():
    matcher = ClaimMatcher(nli=_fake_nli())
    assert matcher.entity_alignment_version == "v1"


def test_v1_overlap_matches_pre_existing_exact_string_behavior():
    """
    Real entity sets from the reviewed benchmark pair gt_01 (claims 4 vs
    13 in experiments/runs/experiment.db). The v1 exact-string overlap for
    this pair was 0.1667 before entity_alignment.py existed -- this must
    not change.
    """
    matcher = ClaimMatcher(nli=_fake_nli())  # no entity_alignment_version passed -> v1
    entities_a = [{"text": t} for t in
                  ["comparison", "vitamin D", "placebo", "intervention", "ARI", "risk", "OR", "CI", "studies"]]
    entities_b = [{"text": t} for t in
                  ["individual", "patient", "statistically significant", "effect", "vitamin D", "ARI",
                   "risk", "ARIs", "treatment group", "pooled", "odds ratio", "rate ratio"]]
    claim_a = {"claim_id": 4, "text": "placeholder a"}
    claim_b = {"claim_id": 13, "text": "placeholder b"}
    emb = [1.0, 0.0, 0.0]

    decision = matcher.match(claim_a, claim_b, entities_a, entities_b, [], [], emb, emb)

    # v1 must still reject this pair at the entity gate (0.1667 < 0.20),
    # exactly as it did before v2 existed.
    assert decision is None


def test_v1_decision_has_no_alignment_detail():
    matcher = ClaimMatcher(nli=_fake_nli())
    entities = [{"text": "vitamin D"}]
    emb = [1.0, 0.0, 0.0]
    claim = {"claim_id": 1, "text": "x"}
    claim2 = {"claim_id": 2, "text": "y"}
    decision = matcher.match(claim, claim2, entities, entities, [], [], emb, emb)
    assert decision is not None
    assert decision.entity_alignment_detail is None


def test_v2_must_be_explicitly_requested():
    matcher = ClaimMatcher(nli=_fake_nli(), entity_alignment_version="v2")
    entities_a = [{"text": t} for t in
                  ["comparison", "vitamin D", "placebo", "intervention", "ARI", "risk", "OR", "CI", "studies"]]
    entities_b = [{"text": t} for t in
                  ["individual", "patient", "statistically significant", "effect", "vitamin D", "ARI",
                   "risk", "ARIs", "treatment group", "pooled", "odds ratio", "rate ratio"]]
    claim_a = {"claim_id": 4, "text": "placeholder a"}
    claim_b = {"claim_id": 13, "text": "placeholder b"}
    emb = [1.0, 0.0, 0.0]

    decision = matcher.match(claim_a, claim_b, entities_a, entities_b, [], [], emb, emb)

    # with v2 explicitly requested, this real pair now clears the gate
    assert decision is not None
    assert decision.entity_alignment_detail is not None
    assert decision.entity_alignment_detail.passed is True


def test_invalid_entity_alignment_version_rejected():
    import pytest
    with pytest.raises(ValueError):
        ClaimMatcher(nli=_fake_nli(), entity_alignment_version="v3")

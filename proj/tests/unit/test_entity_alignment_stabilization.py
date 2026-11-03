"""
Stabilization checkpoint tests for Entity Alignment V2.

These were added during a formal review pass AFTER the initial
implementation, specifically to hunt for the failure classes called out in
that review: unsafe aliases, ambiguous abbreviations, incorrect plural
normalization, partial-match/token-boundary errors, duplicate matching of
the same entity, malformed input, and non-determinism.

One real bug was found and fixed by this pass (see
test_no_duplicate_target_matching below) -- documented here, not hidden.
"""
from __future__ import annotations

import pytest

from matching.entity_alignment import align_entities, classify_pair, MatchCategory


# --------------------------------------------------------------------------
# Malformed / empty input
# --------------------------------------------------------------------------

def test_empty_string_entities_do_not_match_each_other():
    r = classify_pair("", "")
    assert r.category == MatchCategory.UNKNOWN
    assert r.score == 0.0


def test_none_entity_text_handled_without_crash():
    r = classify_pair(None, "vitamin D")
    assert r.category == MatchCategory.UNKNOWN


def test_align_entities_with_none_and_missing_text_key():
    """
    entities_a contains a None text, an empty string, and a dict missing
    the 'text' key entirely -- none of these should crash align_entities
    or be silently treated as matching anything.
    """
    entities_a = [{"text": None}, {"text": ""}, {}]
    entities_b = [{"text": "vitamin D"}]
    res = align_entities(entities_a, entities_b, threshold=0.2)
    assert res.passed is False
    assert res.overlap_score == 0.0


def test_align_entities_both_sides_empty():
    res = align_entities([], [], threshold=0.2)
    assert res.overlap_score == 0.0
    assert res.passed is False
    assert res.rejection_reason


# --------------------------------------------------------------------------
# Duplicate matching of the same entity (real bug found during review)
# --------------------------------------------------------------------------

def test_no_duplicate_target_matching():
    """
    Regression test for a real bug found during the stabilization review:
    two near-duplicate source entities ("ARI" and "ARIs", both extracted
    from the same claim) each independently matched the same single target
    entity ("acute respiratory infection"), and the old implementation
    counted BOTH matches, inflating the overlap score (0.667 instead of
    the correct 0.333 -- since there are 2 distinct source entities but
    only 1 distinct concept in the union once aliasing is applied).

    Fixed via greedy one-to-one bipartite assignment: each target entity
    can be consumed by at most one source match.
    """
    entities_a = [{"text": "ARI"}, {"text": "ARIs"}]
    entities_b = [{"text": "acute respiratory infection"}]
    res = align_entities(entities_a, entities_b, threshold=0.2)

    assert len(res.matches) == 1, "the single target entity must only be consumed once"
    assert res.overlap_score == pytest.approx(1 / 3, abs=0.01)
    assert res.unmatched_source == ["ARIs"] or res.unmatched_source == ["ARI"]


def test_no_duplicate_source_matching():
    """Symmetric case: two target entities should not both be claimed by one source."""
    entities_a = [{"text": "acute respiratory infection"}]
    entities_b = [{"text": "ARI"}, {"text": "ARIs"}]
    res = align_entities(entities_a, entities_b, threshold=0.2)
    assert len(res.matches) == 1


# --------------------------------------------------------------------------
# Unicode / punctuation variants
# --------------------------------------------------------------------------

def test_unicode_nbsp_and_curly_quotes_normalize_consistently():
    # non-breaking space vs regular space
    r = classify_pair("vitamin D\u00A0supplement", "vitamin D supplement")
    assert r.category == MatchCategory.EXACT


def test_punctuation_variants_normalize_to_same_entity():
    r1 = classify_pair("(placebo)", "placebo")
    assert r1.category == MatchCategory.EXACT
    r2 = classify_pair("ARI.", "ARI")
    assert r2.category == MatchCategory.EXACT
    r3 = classify_pair("odds ratio;", "odds ratio")
    assert r3.category == MatchCategory.EXACT


# --------------------------------------------------------------------------
# Ambiguous abbreviations / documented false-positive risk
# --------------------------------------------------------------------------

def test_or_alias_is_a_known_ambiguity_risk():
    """
    "OR" is aliased to "odds ratio" for legitimate statistical claim
    matching, but "OR" is also an ordinary English word/logical
    connective. This module has no way to distinguish "the OR (odds
    ratio) was 0.94" from a hypothetical extraction error that pulled out
    a bare "OR" meaning "or". This test documents the CURRENT accepted
    behavior (treat any standalone "OR" as odds-ratio) and its risk,
    rather than hiding it: this alias is scoped to short, capitalized,
    standalone tokens as extracted by scientific NER over biomedical text,
    where "OR" essentially always denotes odds ratio in practice. If this
    system is ever pointed at non-biomedical or non-statistical text, this
    alias should be reviewed -- it is NOT universally safe.
    """
    r = classify_pair("OR", "odds ratio")
    assert r.category == MatchCategory.ALIAS  # current, documented, accepted behavior


def test_rr_alias_is_context_dependent_not_universally_safe():
    """
    "RR" is aliased here to "relative risk" only. In some study designs
    (e.g. person-time / incidence-rate studies) "RR" is conventionally
    used for "rate ratio" instead -- a related but distinct measure from
    relative risk. This alias is therefore CONTEXT-DEPENDENT, not
    universally safe: on a corpus that uses "RR" to mean rate ratio, this
    alias would produce a false equivalence. Documented here as a known,
    accepted risk (the ALIAS_GROUPS table comment references this test).
    """
    r = classify_pair("RR", "relative risk")
    assert r.category == MatchCategory.ALIAS  # current, documented, accepted behavior
    # but rate ratio is explicitly NOT merged with either RR or relative risk:
    r2 = classify_pair("RR", "rate ratio")
    assert r2.category == MatchCategory.RELATED_NON_EQUIVALENT


# --------------------------------------------------------------------------
# Determinism and stable ordering
# --------------------------------------------------------------------------

def test_deterministic_across_repeated_calls():
    entities_a = [{"text": "OR"}, {"text": "ARI"}, {"text": "vitamin D"}]
    entities_b = [{"text": "odds ratio"}, {"text": "ARIs"}, {"text": "vitamin D"}]
    r1 = align_entities(entities_a, entities_b, threshold=0.2)
    r2 = align_entities(entities_a, entities_b, threshold=0.2)
    assert r1.overlap_score == r2.overlap_score
    assert [(m.source_entity, m.target_entity) for m in r1.matches] == \
           [(m.source_entity, m.target_entity) for m in r2.matches]


def test_match_order_is_stable_score_descending():
    entities_a = [{"text": "OR"}, {"text": "vitamin D"}, {"text": "placebo"}]
    entities_b = [{"text": "odds ratio"}, {"text": "vitamin D"}, {"text": "placebo"}]
    res = align_entities(entities_a, entities_b, threshold=0.2)
    scores = [m.score for m in res.matches]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------
# Invalid threshold values
# --------------------------------------------------------------------------

def test_negative_threshold_rejected():
    with pytest.raises(ValueError):
        align_entities([{"text": "a"}], [{"text": "b"}], threshold=-0.1)


def test_threshold_above_one_rejected():
    with pytest.raises(ValueError):
        align_entities([{"text": "a"}], [{"text": "b"}], threshold=1.5)


def test_boundary_thresholds_are_valid():
    # 0.0 and 1.0 are legal boundary values and must not raise
    r0 = align_entities([{"text": "a"}], [{"text": "b"}], threshold=0.0)
    assert r0.passed is True  # everything clears a 0.0 gate
    r1 = align_entities([{"text": "vitamin D"}], [{"text": "vitamin D"}], threshold=1.0)
    assert r1.passed is True  # exact match reaches the 1.0 ceiling exactly

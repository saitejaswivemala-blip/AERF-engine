"""
Tests for matching/entity_alignment.py (Entity Alignment V2).

These tests exist to prove two things at once:
1. The improved alignment recovers genuine matches that the v1 exact-string
   comparison misses (normalization, alias, partial containment).
2. It does NOT create false matches between scientifically different
   entities -- this is the more important guarantee, per the explicit
   requirement that improving recall must not come at the cost of
   precision on unrelated entities.
"""
from __future__ import annotations

import pytest

from matching.entity_alignment import (
    align_entities,
    classify_pair,
    normalize_entity_text,
    normalize_plural,
    MatchCategory,
)


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

def test_normalize_strips_case_whitespace_and_punctuation():
    assert normalize_entity_text("  Vitamin D  ") == "vitamin d"
    assert normalize_entity_text("ARI.") == "ari"
    assert normalize_entity_text("(placebo)") == "placebo"


def test_normalize_unicode_nfkc():
    # full-width and combining-character variants should normalize to the
    # same plain ASCII form
    assert normalize_entity_text("vitamin D\u00A0") == normalize_entity_text("vitamin D")


def test_plural_normalization_is_conservative():
    assert normalize_plural("trials") == "trial"
    assert normalize_plural("aris") == "aris"       # explicit exception, len==4
    assert normalize_plural("sars") == "sars"       # explicit exception
    assert normalize_plural("os") == "os"           # too short to touch
    assert normalize_plural("class") == "class"     # ends in "ss", left alone


# --------------------------------------------------------------------------
# Positive cases -- real matches v1 would miss
# --------------------------------------------------------------------------

def test_exact_match_after_case_normalization():
    r = classify_pair("Vitamin D", "vitamin d")
    assert r.category == MatchCategory.EXACT
    assert r.score == 1.0


def test_alias_or_odds_ratio():
    r = classify_pair("OR", "odds ratio")
    assert r.category == MatchCategory.ALIAS
    assert r.score == 1.0


def test_alias_ari_plural_and_full_form():
    r = classify_pair("ARIs", "ARI")
    assert r.category == MatchCategory.ALIAS

    r2 = classify_pair("ARI", "acute respiratory infection")
    assert r2.category == MatchCategory.ALIAS


def test_partial_match_token_level_containment():
    r = classify_pair("vitamin D supplements", "vitamin D")
    assert r.category == MatchCategory.PARTIAL
    assert r.score == 0.5


# --------------------------------------------------------------------------
# Negative cases -- must NOT merge (this is the important guarantee)
# --------------------------------------------------------------------------

def test_does_not_merge_unrelated_entities():
    r = classify_pair("vitamin D", "vitamin K")
    assert r.category == MatchCategory.UNKNOWN
    assert r.score == 0.0

    r2 = classify_pair("placebo", "vitamin D")
    assert r2.category == MatchCategory.UNKNOWN


def test_does_not_merge_related_but_distinct_measures():
    """
    Rate ratio, relative risk, and odds ratio are related epidemiological
    quantities but are NOT interchangeable. The alias table must not
    conflate them.
    """
    r = classify_pair("rate ratio", "relative risk")
    assert r.category == MatchCategory.RELATED_NON_EQUIVALENT
    assert r.score == 0.0

    r2 = classify_pair("rate ratio", "RR")
    assert r2.category == MatchCategory.RELATED_NON_EQUIVALENT

    r3 = classify_pair("rate ratio", "odds ratio")
    assert r3.category == MatchCategory.RELATED_NON_EQUIVALENT


def test_regression_no_coincidental_character_substring_match():
    """
    Regression test for a real bug caught during development: a raw
    character-level substring check found "aris" inside "comparison"
    (positions 4-7: c-o-m-P-A-R-I-S-o-n) and incorrectly classified them
    as a partial match. Partial matching must be token-boundary-aware.
    """
    r = classify_pair("comparison", "ARIs")
    assert r.category == MatchCategory.UNKNOWN, (
        "‘comparison’ and ‘ARIs’ share no actual words and must not be "
        "reported as a partial match just because ‘aris’ is a character "
        "substring of ‘comparison’."
    )


def test_short_strings_do_not_trigger_partial_matching():
    # both below the length-4 floor -- must fall through to UNKNOWN, not
    # a spurious partial/substring hit
    r = classify_pair("or", "risk")
    assert r.category != MatchCategory.PARTIAL


# --------------------------------------------------------------------------
# align_entities() -- the aggregate score used as ClaimMatcher's entity gate
# --------------------------------------------------------------------------

def test_align_entities_empty_sets():
    res = align_entities([], [], threshold=0.2)
    assert res.overlap_score == 0.0
    assert res.passed is False


def test_align_entities_real_gt01_pair_now_passes():
    """
    Real reviewed-benchmark pair (gt_01, claims 4 vs 13 in
    experiments/runs/experiment.db). v1 exact-string overlap = 0.167,
    below the 0.20 gate. v2 recovers this via one alias match (OR <->
    odds ratio) with no other entities altered, and passes at 0.222.
    """
    entities_a = [{"text": t} for t in
                  ["comparison", "vitamin D", "placebo", "intervention", "ARI", "risk", "OR", "CI", "studies"]]
    entities_b = [{"text": t} for t in
                  ["individual", "patient", "statistically significant", "effect", "vitamin D", "ARI",
                   "risk", "ARIs", "treatment group", "pooled", "odds ratio", "rate ratio"]]
    res = align_entities(entities_a, entities_b, threshold=0.2)
    assert res.passed is True
    assert res.overlap_score == pytest.approx(0.222, abs=0.01)
    categories = {m.category for m in res.matches}
    assert MatchCategory.ALIAS in categories


def test_align_entities_real_gt02_pair_still_correctly_rejected():
    """
    Same corpus, gt_02 (claims 4 vs 26): no abbreviation overlap exists
    between these two claims' real entities, so v2 correctly still
    rejects this pair -- proving v2 does not just pass everything.
    """
    entities_a = [{"text": t} for t in
                  ["comparison", "vitamin D", "placebo", "intervention", "ARI", "risk", "OR", "CI", "studies"]]
    entities_b = [{"text": t} for t in
                  ["CONCLUSIONS", "adults", "baseline", "prevalence", "sub-optimal", "vitamin D",
                   "implementation", "population-level", "vitamin D replacement", "risk", "ARI", "COVID-19"]]
    res = align_entities(entities_a, entities_b, threshold=0.2)
    assert res.passed is False
    assert res.overlap_score == pytest.approx(0.167, abs=0.01)


def test_alignment_result_is_explainable():
    entities_a = [{"text": "OR"}]
    entities_b = [{"text": "odds ratio"}]
    res = align_entities(entities_a, entities_b, threshold=0.2)
    assert res.passed is True
    assert len(res.matches) == 1
    m = res.matches[0]
    assert m.strategy == "alias_table"
    assert m.reason  # every decision carries a human-readable reason

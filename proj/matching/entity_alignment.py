"""
matching/entity_alignment.py

Entity Alignment V2 — an explainable, deterministic upgrade to the entity
comparison used by ClaimMatcher.match().

WHY THIS EXISTS
---------------
The original (v1, still the default) entity overlap in
matching/claim_matcher.py is:

    ea = {e['text'].strip().lower() for e in entities_a}
    eb = {e['text'].strip().lower() for e in entities_b}
    overlap = len(ea & eb) / len(ea | eb)

That is exact-string-after-lowercasing set overlap. A real diagnostic run
against the 4 cross-paper reviewed benchmark pairs (see
experiments/results/entity_alignment_v2_diagnostic.json, generated
read-only against experiments/runs/experiment.db) found the following
concrete, traceable causes for entity-gate rejection -- NOT a generic
"the threshold is too strict":

  gt_01 (claim 4 vs 13): overlap 0.167 (below 0.20). Root cause: claim 13's
    NER extracted BOTH "ARI" and "ARIs" as separate entities (a plural
    surface-form duplicate), and claim 4 uses "OR" where claim 13's paper
    spells out "odds ratio" -- the same statistical quantity, different
    surface form.
  gt_02 (claim 4 vs 26): overlap 0.167. Root cause: no abbreviation/alias
    overlap available (claim 26 never mentions OR/odds ratio); the shared
    concepts are "vitamin D", "ARI", "risk" only. This pair is NOT
    recoverable by safe normalization alone -- see NOT_RECOVERABLE below.
  gt_03 (claim 13 vs 26): overlap 0.143. Same root cause as gt_02.
  gt_05 (claim 38 vs 4): overlap 0.056. Root cause: claim 38 extracted both
    "relative risk" and "RR" as separate entities (self-duplicate), and
    otherwise shares almost no surface vocabulary with claim 4. NOT
    recoverable by safe normalization alone.

This module does NOT lower any threshold. It computes a *better* overlap
score using normalization and a small, explicit, human-reviewed alias
table, and reports -- for every entity pair -- which matching strategy
produced the decision, so a reviewer can audit every merge. It is wired
into ClaimMatcher as an OPT-IN alternative (entity_alignment_version="v2"),
not a replacement of the default behavior. See ClaimMatcher docstring.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from enum import Enum


class MatchCategory(str, Enum):
    EXACT = "exact"                    # identical after normalization
    ALIAS = "alias"                    # known abbreviation/full-form equivalence
    PARTIAL = "partial"                # one is a substring/superset of the other
    RELATED_NON_EQUIVALENT = "related_non_equivalent"  # related but not the same measure
    CONTRADICTORY = "contradictory"    # explicitly conflicting (not auto-detected here;
                                        # reserved for future attribute-level checks)
    UNKNOWN = "unknown"                # no relationship found


@dataclass(frozen=True)
class EntityMatchResult:
    source_entity: str
    target_entity: str
    source_normalized: str
    target_normalized: str
    strategy: str            # "exact_string" | "alias_table" | "substring" | "none"
    category: MatchCategory
    score: float              # 0.0-1.0, used as this pair's contribution weight
    supporting_attributes: list = field(default_factory=list)
    conflicting_attributes: list = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class AlignmentResult:
    overlap_score: float
    threshold: float
    passed: bool
    matches: list  # list[EntityMatchResult], EXACT/ALIAS/PARTIAL only (i.e. contributing)
    unmatched_source: list
    unmatched_target: list
    rejection_reason: str = ""


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

def normalize_entity_text(text: str) -> str:
    """
    Deterministic normalization: Unicode NFKC, casefold, strip, collapse
    internal whitespace, strip common leading/trailing punctuation.
    Does NOT do any semantic/fuzzy matching -- this is purely mechanical.
    """
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.casefold().strip()
    t = " ".join(t.split())
    t = t.strip(".,;:()[]{}\"'")
    return t


_PLURAL_EXCEPTIONS = {
    # words where naive trailing-"s" stripping would be wrong or unsafe;
    # left un-stripped deliberately.
    "os", "is", "as", "vs", "ras", "sars", "aids", "mers",
}


def normalize_plural(term: str) -> str:
    """
    Conservative singular/plural normalization. Only strips a trailing "s"
    when the word is long enough (>4 chars) that doing so is unlikely to
    collide two unrelated short tokens or acronyms, and the word is not in
    the explicit exception list. This is intentionally NOT a full
    lemmatizer -- it is a narrow, auditable rule, matching the "safe"
    scope requested (do not merge entities on aggressive heuristics).
    """
    if len(term) > 4 and term.endswith("s") and not term.endswith("ss") and term not in _PLURAL_EXCEPTIONS:
        return term[:-1]
    return term


# --------------------------------------------------------------------------
# Alias table -- small, explicit, human-reviewed. Every entry here is a
# genuine abbreviation <-> full-form equivalence, not a semantic-similarity
# guess. Each entry is deliberately narrow: we do NOT alias "rate ratio" to
# "relative risk"/"RR" even though they are related, because they are
# distinct epidemiological measures -- that pair is classified as
# RELATED_NON_EQUIVALENT, not merged. See tests for the negative case.
# --------------------------------------------------------------------------

ALIAS_GROUPS = [
    {"or", "odds ratio"},
    {"ci", "confidence interval"},
    {"rr", "relative risk"},
    {"ari", "aris", "acute respiratory infection", "acute respiratory infections"},
    {"ci", "ci."},
]

_ALIAS_CANONICAL = {}
for _group in ALIAS_GROUPS:
    _canon = sorted(_group, key=len)[-1]  # longest form as canonical label (for readability only)
    for _member in _group:
        _ALIAS_CANONICAL[_member] = _canon

# Explicit "related but NOT equivalent" pairs -- documented so a reviewer
# can see the merges we deliberately did NOT make, and why.
RELATED_NOT_EQUIVALENT_PAIRS = {
    frozenset({"rate ratio", "relative risk"}),
    frozenset({"rate ratio", "rr"}),
    frozenset({"rate ratio", "odds ratio"}),
    frozenset({"relative risk", "odds ratio"}),
}


def _related_non_equivalent(norm_a: str, norm_b: str) -> bool:
    return frozenset({norm_a, norm_b}) in RELATED_NOT_EQUIVALENT_PAIRS


def classify_pair(text_a: str, text_b: str) -> EntityMatchResult:
    """Classify a single (source, target) entity text pair. Pure function, no side effects."""
    na_raw = normalize_entity_text(text_a)
    nb_raw = normalize_entity_text(text_b)
    na = normalize_plural(na_raw)
    nb = normalize_plural(nb_raw)

    if na == nb and na != "":
        return EntityMatchResult(
            text_a, text_b, na, nb, "exact_string", MatchCategory.EXACT, 1.0,
            supporting_attributes=["identical after normalization"],
            reason=f"'{na_raw}' == '{nb_raw}' after unicode/case/whitespace/plural normalization",
        )

    canon_a = _ALIAS_CANONICAL.get(na)
    canon_b = _ALIAS_CANONICAL.get(nb)
    if canon_a is not None and canon_a == canon_b:
        return EntityMatchResult(
            text_a, text_b, na, nb, "alias_table", MatchCategory.ALIAS, 1.0,
            supporting_attributes=[f"both map to alias group '{canon_a}'"],
            reason=f"'{na}' and '{nb}' are a known abbreviation/full-form pair (alias table)",
        )

    if _related_non_equivalent(na, nb):
        return EntityMatchResult(
            text_a, text_b, na, nb, "alias_table", MatchCategory.RELATED_NON_EQUIVALENT, 0.0,
            conflicting_attributes=["distinct statistical measure"],
            reason=f"'{na}' and '{nb}' are related epidemiological terms but are NOT the same "
                   f"measure -- deliberately not merged",
        )

    if na and nb and len(na) >= 4 and len(nb) >= 4:
        tokens_a = set(na.split())
        tokens_b = set(nb.split())
        shorter_tokens, longer_tokens = (tokens_a, tokens_b) if len(na) <= len(nb) else (tokens_b, tokens_a)
        shorter_str, longer_str = (na, nb) if len(na) <= len(nb) else (nb, na)
        # Token-boundary-aware containment only -- deliberately NOT a raw
        # character substring check. Raw substring matching produces false
        # positives on coincidental character runs (e.g. the token "aris"
        # is a raw substring of "comparison" at character level, but the
        # two share no actual words). Requiring every token of the shorter
        # phrase to appear as a whole token of the longer phrase avoids
        # that class of false match entirely.
        if shorter_tokens and shorter_tokens.issubset(longer_tokens):
            return EntityMatchResult(
                text_a, text_b, na, nb, "substring", MatchCategory.PARTIAL, 0.5,
                supporting_attributes=[f"'{shorter_str}' is contained in '{longer_str}' (token-level)"],
                reason=f"'{shorter_str}' is a substring of '{longer_str}' -- treated as a partial, "
                       f"lower-confidence match, not an equivalence",
            )

    return EntityMatchResult(
        text_a, text_b, na, nb, "none", MatchCategory.UNKNOWN, 0.0,
        reason="no exact, alias, or partial relationship found",
    )


def align_entities(entities_a, entities_b, threshold: float = 0.2) -> AlignmentResult:
    """
    entities_a / entities_b: iterables of dicts with a 'text' key (same shape
    ClaimMatcher.match() already receives from Repository.get_claim_entities).

    Returns an AlignmentResult whose overlap_score is a normalized,
    weighted score in [0, 1]: matches are assigned with GREEDY ONE-TO-ONE
    bipartite matching (each source entity may be consumed by at most one
    match, and each target entity may be consumed by at most one match),
    highest-scoring candidate pairs assigned first. This prevents a single
    real-world entity from being counted multiple times just because
    several near-duplicate surface forms on one side (e.g. "ARI" and
    "ARIs" both extracted from the same claim) each independently match
    the same entity on the other side -- see
    tests/unit/test_entity_alignment.py::test_no_duplicate_target_matching
    for the regression this fixes.

    The final score is weighted_sum_of_assigned_matches / union_size,
    where union_size is the count of distinct normalized entities across
    both sides (the same Jaccard-style denominator as v1, so scores remain
    directly comparable).
    """
    texts_a = [e.get("text") for e in entities_a]
    texts_b = [e.get("text") for e in entities_b]
    # malformed entries (missing/None text) normalize to "" and are excluded
    # from consideration entirely rather than being coerced into a fake match.
    texts_a = [t for t in texts_a if t]
    texts_b = [t for t in texts_b if t]

    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be within [0.0, 1.0], got {threshold!r}")

    if not texts_a and not texts_b:
        return AlignmentResult(0.0, threshold, False, [], [], [], "no entities on either side")

    norm_a = {t: normalize_plural(normalize_entity_text(t)) for t in texts_a}
    norm_b = {t: normalize_plural(normalize_entity_text(t)) for t in texts_b}
    union_size = len({*norm_a.values(), *norm_b.values()})

    # Build every candidate (source, target) pair that classifies as a
    # contributing match, then assign greedily, highest score first, each
    # side consumable only once -- this is what prevents double-counting.
    candidates = []
    for ta in texts_a:
        for tb in texts_b:
            r = classify_pair(ta, tb)
            if r.category in (MatchCategory.EXACT, MatchCategory.ALIAS, MatchCategory.PARTIAL):
                candidates.append(r)
    # Stable sort: score descending, then original discovery order, so the
    # assignment (and therefore the final overlap_score) is deterministic
    # across repeated runs given the same input.
    candidates.sort(key=lambda r: -r.score)

    matches = []
    matched_a, matched_b = set(), set()
    for r in candidates:
        if r.source_entity in matched_a or r.target_entity in matched_b:
            continue
        matches.append(r)
        matched_a.add(r.source_entity)
        matched_b.add(r.target_entity)

    weighted_sum = sum(m.score for m in matches)
    overlap_score = (weighted_sum / union_size) if union_size else 0.0
    overlap_score = min(overlap_score, 1.0)

    passed = overlap_score >= threshold
    unmatched_source = [t for t in texts_a if t not in matched_a]
    unmatched_target = [t for t in texts_b if t not in matched_b]
    reason = "" if passed else (
        f"weighted overlap {overlap_score:.3f} < threshold {threshold:.2f} "
        f"({len(matches)} contributing match(es) out of {union_size} distinct entities)"
    )
    return AlignmentResult(overlap_score, threshold, passed, matches,
                            unmatched_source, unmatched_target, reason)

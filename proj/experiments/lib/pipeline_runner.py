"""
Runs claim-pair matching and (optionally) AERF scoring under a given
RunConfig, for baseline comparison and ablation studies.

This module does NOT modify matching/claim_matcher.py or aerf/scorer.py.
It reuses their underlying primitives (ClaimMatcher.cosine,
AERFScorer.factors/score, AdaptiveAERFUpdater) directly, and re-implements
only the small amount of gating logic that needs to be configurable
(entity-alignment on/off, NLI on/off, similarity-only labeling) — the
baseline's own match() method does not expose these as toggles, by design,
since architecturally none of them should be toggle-able in the shipped
product.

`nli` must be an object with `.predict_bidirectional(a, b) -> NLIResult`
(matching nlp.nli.nli_classifier.NLIClassifier's interface). Pass a real
NLIClassifier() where model downloads are available; pass a test double
(see tests/experiments/) for infrastructure testing without network
access — the two are never interchangeable for reporting purposes, and
callers must record which one was used (see ExperimentRun.nli_source).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from matching.claim_matcher import ClaimMatcher, MatchDecision
from matching.entity_alignment import align_entities
from aerf.scorer import AERFScorer
from aerf.adaptive_updater import AdaptiveAERFUpdater
from experiments.lib.experiment_config import RunConfig


@dataclass
class MatchRecord:
    pair_id: str
    claim_a_id: int
    claim_b_id: int
    cosine_similarity: float
    entity_overlap_score: float | None
    nli_label: str | None
    nli_confidence: float | None
    predicted_label: str  # one of supports/contradicts/neutral/uncertain -- what metrics.evaluate() consumes


def _entity_overlap(entities_a, entities_b) -> float:
    ea = {e["text"].strip().lower() for e in entities_a}
    eb = {e["text"].strip().lower() for e in entities_b}
    u = ea | eb
    return len(ea & eb) / len(u) if u else 0.0


def run_matching(
    repo,
    config: RunConfig,
    nli,
    claim_pairs: list[tuple[dict, dict]],
) -> list[MatchRecord]:
    """
    claim_pairs: explicit list of (claim_a, claim_b) dicts to evaluate
    (each with at least 'claim_id' and 'text') plus their embeddings
    resolved via repo.get_claim_embedding. Explicit pairs (rather than
    all-pairs candidate generation) so this can be driven directly by a
    ground-truth annotation set for baseline/ablation comparison.
    """

    records = []

    for a, b in claim_pairs:
        va = repo.get_claim_embedding(a["claim_id"])
        vb = repo.get_claim_embedding(b["claim_id"])

        if va is None or vb is None:
            continue

        sim = ClaimMatcher.cosine(va, vb)
        pair_id = f"{a['claim_id']}-{b['claim_id']}"

        # ------------------------------------------------------------
        # Stage 1: Similarity candidate gate
        # ------------------------------------------------------------
        if sim < config.similarity_threshold:
            records.append(
                MatchRecord(
                    pair_id,
                    a["claim_id"],
                    b["claim_id"],
                    sim,
                    None,
                    None,
                    None,
                    "uncertain",
                )
            )
            continue

        # ------------------------------------------------------------
        # Baseline A: similarity-only classification
        # ------------------------------------------------------------
        if config.label_from_similarity_only:
            # Baseline A: similarity alone decides the label.
            # A higher similarity band is treated as 'supports' and a
            # mid band as 'neutral' purely to give the baseline some
            # three-way behaviour to compare against.
            label = "supports" if sim >= 0.75 else "neutral"

            records.append(
                MatchRecord(
                    pair_id,
                    a["claim_id"],
                    b["claim_id"],
                    sim,
                    None,
                    None,
                    None,
                    label,
                )
            )
            continue

        # ------------------------------------------------------------
        # Stage 2: Entity alignment evidence
        # ------------------------------------------------------------
        ents_a = repo.get_claim_entities(a["claim_id"])
        ents_b = repo.get_claim_entities(b["claim_id"])

        alignment = (
            align_entities(ents_a, ents_b, threshold=config.entity_threshold)
            if config.use_entity_alignment
            else None
        )
        overlap = alignment.overlap_score if alignment is not None else None

        if config.use_entity_alignment:
            u = (
                {e["text"].strip().lower() for e in ents_a}
                | {e["text"].strip().lower() for e in ents_b}
            )
            aligned = (
                alignment.passed
                if u
                else sim >= max(0.75, config.similarity_threshold)
            )
        else:
            aligned = True

        # ------------------------------------------------------------
        # Stage 3: NLI
        # ------------------------------------------------------------
        #
        # IMPORTANT:
        # Entity alignment is retained as evidence, but a low entity
        # overlap no longer prevents NLI from running. This allows
        # semantically related scientific claims with different surface
        # forms to reach the real NLI model.
        # ------------------------------------------------------------
        if not config.use_nli:
            # Baseline without NLI but past the similarity gate:
            # nothing left to decide the relationship type, so label
            # 'uncertain' rather than inventing a signal.
            records.append(
                MatchRecord(
                    pair_id,
                    a["claim_id"],
                    b["claim_id"],
                    sim,
                    overlap,
                    None,
                    None,
                    "uncertain",
                )
            )
            continue

        n = nli.predict_bidirectional(a["text"], b["text"])

        label = "uncertain"

        if aligned and n.confidence >= config.nli_threshold:
            label = {
                "entailment": "supports",
                "contradiction": "contradicts",
                "neutral": "neutral",
            }.get(n.label, "uncertain")

        # ------------------------------------------------------------
        # If entity alignment is weak, retain the NLI result as evidence
        # but do not automatically convert it into a final relationship.
        #
        # This preserves the entity-alignment signal without blocking
        # semantic evaluation.
        # ------------------------------------------------------------
        if not aligned:
            label = "uncertain"

        records.append(
            MatchRecord(
                pair_id,
                a["claim_id"],
                b["claim_id"],
                sim,
                overlap,
                n.label,
                n.confidence,
                label,
            )
        )

    return records


def score_claims_with_config(
    repo,
    config: RunConfig,
    claim_ids: list[int],
) -> dict[int, float]:
    """
    Runs AERF scoring (baseline scorer) honoring zero_provenance_weight
    and use_adaptive_updates. Returns {claim_id: score}.
    """

    if not config.use_aerf_scoring:
        return {}

    weights = dict(AERFScorer(repo).weights)

    if config.zero_provenance_weight:
        weights["provenance"] = 0.0

    scorer = AERFScorer(repo, weights=weights)

    scores = {}

    # ------------------------------------------------------------
    # Initial AERF scoring
    # ------------------------------------------------------------
    for cid in claim_ids:
        result = scorer.score(cid)

        repo.store_aerf_score(
            cid,
            result.score,
            result.factors,
            scorer.version,
        )

        scores[cid] = result.score

    # ------------------------------------------------------------
    # Adaptive AERF update
    # ------------------------------------------------------------
    if config.use_adaptive_updates:
        updater = AdaptiveAERFUpdater(repo, scorer)

        updater.recompute(set(claim_ids))

        for cid in claim_ids:
            current = repo.get_current_aerf(cid)

            if current is not None:
                scores[cid] = current["score"]

    return scores
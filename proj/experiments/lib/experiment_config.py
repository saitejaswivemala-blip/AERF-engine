"""
Configuration objects for baseline comparisons and ablation studies.

These do NOT modify matching/claim_matcher.py, aerf/scorer.py, or any other
baseline module — the brief requires preserving the Phase 3 baseline
untouched. Instead, experiments/lib/pipeline_runner.py re-implements the
matching/scoring decision logic at the granularity these configs need,
reusing the same underlying primitives (NLIClassifier, cosine similarity,
entity overlap, AERFScorer factors) that the baseline uses internally.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    name: str
    description: str
    # matching stage toggles
    use_entity_alignment: bool = True
    use_nli: bool = True
    similarity_threshold: float = 0.55
    entity_threshold: float = 0.15
    nli_threshold: float = 0.6
    label_from_similarity_only: bool = False  # Baseline A special case
    # scoring stage toggles
    use_aerf_scoring: bool = True
    use_adaptive_updates: bool = True
    zero_provenance_weight: bool = False


# The four comparisons requested in the experiment brief (Section 6).
BASELINE_A_SIMILARITY_ONLY = RunConfig(
    name="baseline_a_similarity_only",
    description="Label pairs purely by embedding cosine similarity (>=threshold -> 'supports', "
                "no NLI, no entity alignment, no AERF). Tests whether similarity alone is enough.",
    use_entity_alignment=False, use_nli=False, label_from_similarity_only=True,
    use_aerf_scoring=False, use_adaptive_updates=False,
)

BASELINE_B_SIMILARITY_NLI = RunConfig(
    name="baseline_b_similarity_plus_nli",
    description="Similarity gate + bidirectional NLI label, no entity-alignment gate, no AERF.",
    use_entity_alignment=False, use_nli=True, use_aerf_scoring=False, use_adaptive_updates=False,
)

BASELINE_C_SIMILARITY_ENTITY_NLI = RunConfig(
    name="baseline_c_similarity_entity_nli",
    description="Similarity gate + entity-alignment gate + bidirectional NLI label (this is the "
                "baseline Phase 3 ClaimMatcher's matching logic), no AERF scoring layer.",
    use_entity_alignment=True, use_nli=True, use_aerf_scoring=False, use_adaptive_updates=False,
)

FULL_AERF = RunConfig(
    name="full_aerf",
    description="Similarity -> entity alignment -> bidirectional NLI -> AERF scoring -> adaptive "
                "evidence update -> provenance-aware reasoning. The complete Phase 3 baseline pipeline.",
    use_entity_alignment=True, use_nli=True, use_aerf_scoring=True, use_adaptive_updates=True,
)

# The six ablations requested in the experiment brief (Section 7).
ABLATION_WITHOUT_ENTITY_ALIGNMENT = RunConfig(
    name="ablation_without_entity_alignment",
    description="Full AERF but the entity-overlap gate is disabled (every similarity-gated pair "
                "proceeds to NLI regardless of entity overlap).",
    use_entity_alignment=False, use_nli=True, use_aerf_scoring=True, use_adaptive_updates=True,
)

ABLATION_WITHOUT_ADAPTIVE_SCORING = RunConfig(
    name="ablation_without_adaptive_scoring",
    description="Full AERF but scores are computed once and never recomputed as new evidence "
                "arrives (equivalent to 'static scoring').",
    use_entity_alignment=True, use_nli=True, use_aerf_scoring=True, use_adaptive_updates=False,
)

ABLATION_STATIC_SCORING = ABLATION_WITHOUT_ADAPTIVE_SCORING  # brief lists these as distinct items
# (5 and 4); they are the same mechanism in this codebase's architecture,
# since AERF has no other notion of "static" scoring beyond "computed once,
# never adaptively recomputed" -- documented explicitly rather than
# building a second, functionally-identical config to pad the count.

ABLATION_WITHOUT_PROVENANCE_WEIGHTING = RunConfig(
    name="ablation_without_provenance_weighting",
    description="Full AERF but the provenance-completeness factor's weight is zeroed (redistributed "
                "nowhere -- the AERF score simply excludes that term, per the linear scoring formula).",
    use_entity_alignment=True, use_nli=True, use_aerf_scoring=True, use_adaptive_updates=True,
    zero_provenance_weight=True,
)

ABLATION_SIMILARITY_NLI_ONLY = BASELINE_B_SIMILARITY_NLI  # brief's ablation #6 is baseline B

ALL_BASELINES = [BASELINE_A_SIMILARITY_ONLY, BASELINE_B_SIMILARITY_NLI,
                  BASELINE_C_SIMILARITY_ENTITY_NLI, FULL_AERF]

ALL_ABLATIONS = [FULL_AERF, ABLATION_WITHOUT_ENTITY_ALIGNMENT,
                  ABLATION_WITHOUT_ADAPTIVE_SCORING, ABLATION_WITHOUT_PROVENANCE_WEIGHTING,
                  ABLATION_SIMILARITY_NLI_ONLY]

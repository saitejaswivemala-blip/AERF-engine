"""
Adversarial claim-pair test cases.

IMPORTANT: these are hand-authored, illustrative sentence pairs designed to
probe specific failure modes (per experiment brief Section 9) -- they are
NOT extracted from the paper corpus, and their `expected_label` is an
analyst judgment, not a second annotator's independent label. They belong
in unit/integration testing of the matching logic, not in the ground-truth
claim_pairs evaluation set in experiments/annotations/, and must never be
merged into that set or reported as if drawn from real papers.

A few cases are deliberately grounded in wording that echoes the real
corpus (P1 vs P2's Blue Sky trial dispute; P1's overall-null vs
children-subgroup-positive result) to make the "high similarity /
different population" and "same topic, real methodological disagreement"
cases realistic, but the exact sentences are written for this test set,
not quoted from the papers.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AdversarialCase:
    case_id: str
    category: str
    claim_a: str
    claim_b: str
    expected_label: str  # supports | contradicts | neutral | uncertain
    rationale: str


CASES: list[AdversarialCase] = [
    AdversarialCase(
        "adv_01", "high_similarity_but_contradiction",
        "Vitamin D supplementation reduced the risk of acute respiratory infection.",
        "Vitamin D supplementation did not reduce the risk of acute respiratory infection.",
        "contradicts",
        "Near-identical wording except for negation; a similarity-only baseline is expected to "
        "mislabel this as 'supports' or fail to distinguish it from case adv_09 (paraphrase).",
    ),
    AdversarialCase(
        "adv_02", "high_similarity_but_neutral",
        "Vitamin D supplementation reduced the risk of upper respiratory infection in older adults.",
        "Vitamin D supplementation reduced the risk of lower respiratory infection in older adults.",
        "neutral",
        "High lexical/semantic overlap but distinct outcomes (upper vs lower respiratory tract) that "
        "are not equivalent claims; correct behaviour is 'neutral' or 'uncertain', not 'supports'.",
    ),
    AdversarialCase(
        "adv_03", "negation",
        "Daily vitamin D dosing produced a statistically significant protective effect against ARI.",
        "Daily vitamin D dosing did not produce a statistically significant protective effect against ARI.",
        "contradicts",
        "Direct negation of a specific, dosing-qualified claim.",
    ),
    AdversarialCase(
        "adv_04", "different_populations",
        "Vitamin D supplementation significantly reduced ARI risk in children aged 1 to 16 years.",
        "Vitamin D supplementation did not significantly reduce ARI risk in the pediatric population.",
        "uncertain",
        "Real corpus tension (P1's child subgroup vs P5's pediatric review): overlapping but not "
        "identical population definitions (age-restricted RCT subgroup vs a pooled pediatric "
        "systematic review including treatment trials) make this a genuinely hard case where "
        "'contradicts' is defensible but 'uncertain' pending closer population-definition checking "
        "is the more honest system behaviour; expected_label is set to 'uncertain' deliberately.",
    ),
    AdversarialCase(
        "adv_05", "different_experimental_conditions",
        "A test-and-treat approach to vitamin D replacement did not reduce COVID-19 risk in UK adults.",
        "Daily vitamin D supplementation reduced influenza risk in Japanese schoolchildren.",
        "neutral",
        "Different intervention design, different outcome, different population and country; despite "
        "topical similarity (vitamin D, respiratory illness) these are not comparable claims.",
    ),
    AdversarialCase(
        "adv_06", "different_time_periods",
        "A 2021 meta-analysis found vitamin D supplementation had a significant protective effect "
        "against ARI (OR 0.92).",
        "A 2024 updated meta-analysis of the same trials plus new data found no significant "
        "protective effect against ARI (OR 0.94).",
        "contradicts",
        "Real corpus case (P1 describing its own prior 2021 result vs its 2024 update): same "
        "research question, materially different conclusion after more data accrued -- a genuine "
        "case for the knowledge-evolution/temporal-decay mechanism, not just static contradiction.",
    ),
    AdversarialCase(
        "adv_07", "conflicting_numerical_findings",
        "Vitamin D supplementation reduced ARI incidence by approximately 6% (OR 0.94).",
        "Vitamin D supplementation reduced ARI incidence by approximately 26% (OR 0.74).",
        "neutral",
        "Same qualitative direction (both report a reduction), different magnitude/subgroup; should "
        "not be labelled 'contradicts' merely because the numbers differ.",
    ),
    AdversarialCase(
        "adv_08", "weak_vs_strong_evidence",
        "A single unpowered pilot RCT (n=34) suggested vitamin D may reduce ARI symptoms.",
        "A meta-analysis of 46 RCTs (64,086 participants) found no significant overall effect of "
        "vitamin D on ARI risk.",
        "uncertain",
        "Both claims could be technically true simultaneously (a small trial's point estimate can "
        "differ from a precise pooled null); AERF's evidence-mass weighting should down-weight the "
        "pilot RCT's contribution rather than the matcher labelling this a clean contradiction.",
    ),
    AdversarialCase(
        "adv_09", "duplicate_claims",
        "Vitamin D supplementation did not significantly reduce the risk of acute respiratory "
        "infection (OR 0.94, 95% CI 0.88-1.00).",
        "Vitamin D supplementation was not associated with a statistically significant reduction in "
        "ARI risk (odds ratio 0.94).",
        "supports",
        "Paraphrase of the same underlying finding (P1's headline result restated); a correct system "
        "should recognise these as corroborating, not treat the paraphrase as independent new "
        "evidence with its own full corroboration weight -- flags AERF's lack of a "
        "same-underlying-evidence dedup mechanism, noted as a limitation in the technical audit.",
    ),
    AdversarialCase(
        "adv_10", "incomplete_claims",
        "reduced risk when given daily",
        "Vitamin D supplementation did not reduce ARI risk when given monthly.",
        "uncertain",
        "Claim A is a sentence fragment lacking a clear subject; a robust claim extractor should "
        "not have produced it, and a robust matcher should not confidently classify it.",
    ),
]


def as_records() -> list[dict]:
    return [
        {
            "case_id": c.case_id, "category": c.category,
            "claim_a": c.claim_a, "claim_b": c.claim_b,
            "expected_label": c.expected_label, "rationale": c.rationale,
        }
        for c in CASES
    ]

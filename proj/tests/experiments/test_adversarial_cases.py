from experiments.lib.adversarial_cases import CASES, as_records
from experiments.lib.metrics import LABELS

REQUIRED_CATEGORIES = {
    "high_similarity_but_contradiction", "high_similarity_but_neutral", "negation",
    "different_populations", "different_experimental_conditions", "different_time_periods",
    "conflicting_numerical_findings", "weak_vs_strong_evidence", "duplicate_claims",
    "incomplete_claims",
}


def test_all_ten_brief_categories_are_covered():
    covered = {c.category for c in CASES}
    assert REQUIRED_CATEGORIES <= covered


def test_case_ids_are_unique():
    ids = [c.case_id for c in CASES]
    assert len(ids) == len(set(ids))


def test_expected_labels_are_valid():
    for c in CASES:
        assert c.expected_label in LABELS


def test_every_case_has_two_nonempty_claims_and_a_rationale():
    for c in CASES:
        assert c.claim_a.strip() and c.claim_b.strip() and c.rationale.strip()


def test_as_records_matches_case_count():
    assert len(as_records()) == len(CASES)

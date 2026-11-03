from scripts.inspect_claim_mapping import (
    REVIEWED_MAPPING_PATH,
    load_reviewed_mapping,
    resolve_reviewed_claim,
)


def test_reviewed_mapping_resolves_only_verified_id_and_text():
    mappings = {
        "gt_01": {
            "claim_a": {
                "db_claim_id": 4,
                "db_claim_text": "stored claim",
                "rationale": "same reported result",
            }
        }
    }
    result = resolve_reviewed_claim(
        "gt_01", "claim_a", mappings, [{"claim_id": 4, "text": "stored claim"}]
    )

    assert result["status"] == "resolved"
    assert result["method"] == "reviewed_mapping"
    assert result["claim_id"] == 4
    assert result["rationale"] == "same reported result"


def test_reviewed_mapping_keeps_explicit_null_unresolved():
    mappings = {"gt_04": {"claim_b": {"db_claim_id": None, "rationale": "not stored"}}}

    result = resolve_reviewed_claim("gt_04", "claim_b", mappings, [{"claim_id": 99, "text": "other"}])

    assert result == {"status": "unresolved", "reason": "not stored"}


def test_reviewed_mapping_rejects_stale_stored_text():
    mappings = {"gt_01": {"claim_a": {"db_claim_id": 4, "db_claim_text": "old text"}}}

    result = resolve_reviewed_claim(
        "gt_01", "claim_a", mappings, [{"claim_id": 4, "text": "current text"}]
    )

    assert result["status"] == "unresolved"
    assert "does not match" in result["reason"]


def test_checked_in_mapping_artifact_contains_all_benchmark_pairs():
    mappings = load_reviewed_mapping(REVIEWED_MAPPING_PATH)

    assert mappings is not None
    assert set(mappings) == {f"gt_{index:02d}" for index in range(1, 9)}
    assert all("claim_a" in mappings[pair_id] and "claim_b" in mappings[pair_id]
               for pair_id in mappings)

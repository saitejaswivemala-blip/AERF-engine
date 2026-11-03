"""
tests/unit/test_independent_evaluation_protocol.py

Automated unit tests for the independent evaluation protocol:
1. Manifest integrity and exclusion of development papers.
2. Dataset schema validation and lack of development paper contamination.
3. Dual-annotator agreement computation (Cohen's Kappa) and edge case handling.
4. Adjudication workflow for resolving discrepancies.
5. Automated validation checks:
   - Paper overlap detection
   - Duplicate / leaked sentence detection
   - Missing source reference detection
   - Invalid label detection
6. Frozen extractor configuration integrity and SHA-256 validation.
"""

import csv
import json
from pathlib import Path
import pytest

from scripts.adjudicate_independent_annotations import (
    compute_cohens_kappa,
    process_annotations,
)
from scripts.validate_independent_dataset import (
    check_paper_overlap,
    token_jaccard,
    validate_manifest,
    validate_dataset,
    validate_frozen_config,
    run_full_validation,
)

ROOT = Path(__file__).resolve().parent.parent.parent
INDEP_DIR = ROOT / "tests" / "data" / "independent_eval"
MANIFEST_PATH = INDEP_DIR / "independent_paper_manifest.json"
DATASET_PATH = INDEP_DIR / "claim_extractor_independent_eval.csv"
FROZEN_CONFIG_PATH = ROOT / "reports" / "frozen_extractor_config.json"


# =========================================================================
# 1. Manifest & Protocol Structure Tests
# =========================================================================

def test_independent_eval_dir_and_scaffold_exist():
    assert INDEP_DIR.exists(), f"Missing independent eval dir: {INDEP_DIR}"
    assert MANIFEST_PATH.exists(), f"Missing manifest: {MANIFEST_PATH}"
    assert DATASET_PATH.exists(), f"Missing dataset: {DATASET_PATH}"
    assert FROZEN_CONFIG_PATH.exists(), f"Missing frozen config: {FROZEN_CONFIG_PATH}"


def test_manifest_excludes_all_development_papers():
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert data["status"] == "PENDING_UNSEEN_DATA_COLLECTION"
    assert "strict_exclusion_list" in data
    excluded_ids = [p["paper_id"] for p in data["strict_exclusion_list"]]
    for expected in ["Vaswani2017", "Devlin2018", "Martineau2017", "Bolland2022", "Jolliffe2024"]:
        assert expected in excluded_ids

    # Ensure no registered independent papers overlap with exclusion list
    registered = data.get("registered_independent_papers", [])
    for reg in registered:
        overlap, reason = check_paper_overlap(reg.get("paper_id", ""), reg.get("doi_or_url", ""), reg.get("title", ""))
        assert not overlap, f"Contamination detected in registered independent paper: {reason}"


# =========================================================================
# 2. Dual-Annotator Agreement (Cohen's Kappa) Tests
# =========================================================================

def test_cohens_kappa_perfect_agreement():
    l1 = [1, 1, 0, 0, 1]
    l2 = [1, 1, 0, 0, 1]
    res = compute_cohens_kappa(l1, l2)
    assert res["observed_agreement"] == 1.0
    assert res["cohens_kappa"] == 1.0
    assert res["disagreed_samples"] == 0


def test_cohens_kappa_partial_agreement():
    l1 = [1, 0, 1, 1, 0, 0, 1, 0]
    l2 = [1, 1, 1, 1, 0, 0, 0, 0]
    res = compute_cohens_kappa(l1, l2)
    assert res["total_samples"] == 8
    assert res["agreed_samples"] == 6
    assert res["disagreed_samples"] == 2
    assert 0.0 < res["cohens_kappa"] < 1.0


def test_cohens_kappa_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="mismatch"):
        compute_cohens_kappa([1, 0], [1])


def test_cohens_kappa_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_cohens_kappa([], [])


# =========================================================================
# 3. Adjudication Workflow Tests
# =========================================================================

def test_process_annotations_empty_csv(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("sample_id,sentence,annotator_1_label,annotator_2_label,adjudicated_label\n", encoding="utf-8")
    res = process_annotations(empty_csv, tmp_path / "dis.csv", tmp_path / "final.csv")
    assert res["status"] == "PENDING_DATA_COLLECTION"
    assert res["total_rows"] == 0


def test_process_annotations_with_disagreements_and_resolution(tmp_path):
    raw_csv = tmp_path / "annotated.csv"
    raw_csv.write_text(
        "sample_id,paper_id,sentence,annotator_1_label,annotator_2_label,adjudicated_label,category,doi_or_url,annotator_1_rationale,adjudication_notes\n"
        "1,P1,The alloy exhibits 450 MPa tensile strength.,1,1,,finding,10.1016/j.matlet,agreed,\n"
        "2,P1,Specimens were heated to 600 C for 2 hours.,0,1,0,methodology,10.1016/j.matlet,method,adjudicated non-claim\n",
        encoding="utf-8",
    )
    dis_csv = tmp_path / "dis.csv"
    final_csv = tmp_path / "final.csv"

    res = process_annotations(raw_csv, dis_csv, final_csv)
    assert res["status"] == "RESOLVED"
    assert res["disagreement_count"] == 1
    assert dis_csv.exists()
    assert final_csv.exists()

    with open(final_csv, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["label"] == "1"
    assert rows[1]["label"] == "0"


# =========================================================================
# 4. Validation Checks: Leakage, Overlap, Source, Label Validity
# =========================================================================

def test_paper_overlap_detection():
    # Should catch development papers
    assert check_paper_overlap("Vaswani2017", "arXiv:1706.03762")[0] is True
    assert check_paper_overlap("Devlin2018", "arXiv:1810.04805")[0] is True
    assert check_paper_overlap("PaperX", "10.1136/bmj.i6583")[0] is True
    assert check_paper_overlap("PaperY", "10.1101/2024.09.18.24313866")[0] is True

    # Should allow genuine unseen paper
    assert check_paper_overlap("Astronomy2025", "10.1038/s41586-025-0001-x", "Planetary atmospheric spectra")[0] is False


def test_dataset_validator_catches_invalid_labels(tmp_path):
    bad_csv = tmp_path / "bad_labels.csv"
    bad_csv.write_text(
        "sample_id,paper_id,sentence,label,category,doi_or_url\n"
        "1,MatSci1,Tensile strain was measured.,2,finding,10.1016/j.actamat.2025\n",
        encoding="utf-8",
    )
    errors = validate_dataset(bad_csv, ROOT)
    assert any("invalid label" in e for e in errors)


def test_dataset_validator_catches_missing_doi(tmp_path):
    bad_csv = tmp_path / "missing_doi.csv"
    bad_csv.write_text(
        "sample_id,paper_id,sentence,label,category,doi_or_url\n"
        "1,MatSci1,Tensile strain was measured.,1,finding,unspecified\n",
        encoding="utf-8",
    )
    errors = validate_dataset(bad_csv, ROOT)
    assert any("missing valid DOI" in e for e in errors)


def test_dataset_validator_catches_leaked_sentence(tmp_path):
    bad_csv = tmp_path / "leaked.csv"
    # Sentence from Vaswani et al. in claim_extractor_eval.csv
    bad_csv.write_text(
        "sample_id,paper_id,sentence,label,category,doi_or_url\n"
        '1,NewPaper,"We used the Adam optimizer with beta 1 = 0.9, beta 2 = 0.98 and epsilon = 10−9.",0,methodology,10.1038/s41586\n',
        encoding="utf-8",
    )
    errors = validate_dataset(bad_csv, ROOT)
    assert any("exact duplicate" in e or "token overlap" in e for e in errors)


def test_frozen_configuration_integrity():
    errors = validate_frozen_config(FROZEN_CONFIG_PATH, ROOT)
    assert errors == [], f"Frozen configuration integrity errors: {errors}"


def test_full_validation_passes_on_project():
    res = run_full_validation(ROOT)
    assert res["valid"] is True, f"Full independent validation failed: {res['all_errors']}"


# =========================================================================
# 5. Dual Sheet Merging & Registration Tests
# =========================================================================

def test_merge_separate_annotator_sheets(tmp_path):
    from scripts.adjudicate_independent_annotations import merge_separate_annotator_sheets
    a1_csv = tmp_path / "a1.csv"
    a2_csv = tmp_path / "a2.csv"
    out_csv = tmp_path / "merged.csv"

    a1_csv.write_text(
        "sample_id,paper_id,sentence,source_section,page_number,doi_or_url,annotator_label,annotator_rationale\n"
        "S1,P1,Alloy hardness was 350 HV.,Results,4,10.1016/j.actamat,1,hardness metric\n"
        "S2,P1,Samples were ground with SiC paper.,Methods,2,10.1016/j.actamat,0,preparation step\n",
        encoding="utf-8",
    )
    a2_csv.write_text(
        "sample_id,paper_id,sentence,source_section,page_number,doi_or_url,annotator_label,annotator_rationale\n"
        "S1,P1,Alloy hardness was 350 HV.,Results,4,10.1016/j.actamat,1,finding\n"
        "S2,P1,Samples were ground with SiC paper.,Methods,2,10.1016/j.actamat,1,disagree on claim\n",
        encoding="utf-8",
    )

    merged = merge_separate_annotator_sheets(a1_csv, a2_csv, out_csv)
    assert len(merged) == 2
    # S1 agreed: adjudicated_label should be 1
    assert merged[0]["adjudicated_label"] == "1"
    # S2 disagreed: adjudicated_label should be empty for review
    assert merged[1]["adjudicated_label"] == ""


def test_register_paper_helper(tmp_path):
    from scripts.register_independent_paper import register_paper
    mock_pdf = tmp_path / "test_paper.pdf"
    mock_pdf.write_bytes(b"%PDF-1.4 mock content for registration")
    mock_manifest = tmp_path / "manifest.json"
    mock_manifest.write_text(json.dumps({"registered_independent_papers": []}), encoding="utf-8")

    entry = register_paper(
        pdf_path=mock_pdf,
        paper_id="TEST_INDEP_01",
        title="Novel superalloy creep resistance",
        authors="Taylor M, Davis K",
        year=2024,
        venue="Acta Materialia",
        doi_or_url="10.1016/j.actamat.2024.9999",
        discipline="Materials Science & Metallurgy",
        manifest_path=mock_manifest,
    )

    assert entry["paper_id"] == "TEST_INDEP_01"
    assert "pdf_sha256" in entry
    assert entry["development_history_check"] == "VERIFIED_UNSEEN_IN_AERF"

    # Verify that trying to register a development paper raises ValueError
    with pytest.raises(ValueError, match="REJECTED"):
        register_paper(
            pdf_path=mock_pdf,
            paper_id="TEST_VASWANI_LEAK",
            title="Attention Is All You Need",
            authors="Vaswani et al.",
            year=2017,
            venue="NeurIPS",
            doi_or_url="arXiv:1706.03762",
            discipline="Computer Science",
            manifest_path=mock_manifest,
        )


def test_sentence_filtering_in_sampler():
    from scripts.sample_sentences_from_pdf import is_valid_sentence
    # Valid scientific sentence
    assert is_valid_sentence("The measured lattice parameter of the cubic phase was 0.352 nm at room temperature.") is True
    # Short fragments rejected
    assert is_valid_sentence("Figure 1 shows data.") is False
    # Headers / figure captions rejected
    assert is_valid_sentence("Table 2. Summary of measured mechanical tensile properties.") is False
    # URLs rejected
    assert is_valid_sentence("Additional information is available at https://www.nature.com/materials.") is False

"""
tests/unit/test_evaluate_claim_extractor.py

Automated tests verifying:
- Dataset loading and validation (missing columns, bad labels, empty dataset).
- Real evaluation dataset properties (label distribution, category diversity, paper diversity, no duplicates).
- Metric calculation and zero-division handling.
- Stratified metric breakdown (by category and by paper).
- Report generation (metrics JSON, predictions CSV, confusion matrix PNG).
- Separation of synthetic testing datasets.
"""
import csv
import json
from pathlib import Path
import pytest

from evaluate_claim_extractor import (
    load_dataset,
    compute_metrics,
    compute_stratified_metrics,
    save_reports,
    evaluate,
    DEFAULT_DATASET,
)
from nlp.claim_extractor import ClaimResult


ROOT = Path(__file__).resolve().parent.parent.parent
REAL_DATASET_PATH = ROOT / "tests" / "data" / "claim_extractor_eval.csv"
SYNTHETIC_DATASET_PATH = ROOT / "tests" / "data" / "claim_extractor_synthetic_eval.csv"


class MockExtractor:
    """Fast, deterministic extractor mock for evaluation pipeline testing."""

    def __init__(self, claim_rule=lambda text: "significant" in text.lower() or "achieves" in text.lower()):
        self.claim_rule = claim_rule

    def extract(self, text: str):
        if self.claim_rule(text):
            return [ClaimResult(text=text, start=0, end=len(text), score=0.92, reason="mock_cue")]
        return []


# =========================================================================
# 1. Dataset Loading and Schema Tests
# =========================================================================

def test_real_dataset_exists_and_loads():
    assert REAL_DATASET_PATH.exists(), f"Missing dataset: {REAL_DATASET_PATH}"
    rows = load_dataset(REAL_DATASET_PATH)
    assert len(rows) >= 50, f"Expected at least 50 samples, found {len(rows)}"


def test_real_dataset_has_required_schema():
    rows = load_dataset(REAL_DATASET_PATH)
    for r in rows:
        assert "sentence" in r and r["sentence"].strip()
        assert "label" in r and r["label"] in (0, 1)
        assert "paper_id" in r and r["paper_id"] != "unspecified"
        assert "category" in r and r["category"] != "unspecified"
        assert "annotation_rationale" in r and r["annotation_rationale"].strip()


def test_real_dataset_labels_are_balanced_and_valid():
    rows = load_dataset(REAL_DATASET_PATH)
    labels = [r["label"] for r in rows]
    pos = sum(1 for l in labels if l == 1)
    neg = sum(1 for l in labels if l == 0)

    assert set(labels) == {0, 1}
    assert pos >= 20, f"Expected at least 20 positive claims, got {pos}"
    assert neg >= 20, f"Expected at least 20 negative non-claims, got {neg}"


def test_real_dataset_category_diversity():
    rows = load_dataset(REAL_DATASET_PATH)
    categories = {r["category"] for r in rows}
    required_categories = {
        "finding",
        "negative_finding",
        "statistical_result",
        "hypothesis",
        "background",
        "methodology",
        "dataset_description",
        "limitation",
        "ambiguous",
    }
    missing = required_categories - categories
    assert not missing, f"Dataset is missing required categories: {missing}"


def test_real_dataset_paper_diversity():
    rows = load_dataset(REAL_DATASET_PATH)
    papers = {r["paper_id"] for r in rows}
    assert len(papers) >= 5, f"Expected at least 5 distinct papers, found {len(papers)}: {papers}"


def test_real_dataset_has_no_duplicate_sentences():
    rows = load_dataset(REAL_DATASET_PATH)
    seen = set()
    duplicates = []
    for r in rows:
        norm = r["sentence"].strip().lower()
        if norm in seen:
            duplicates.append(r["sentence"])
        seen.add(norm)
    assert not duplicates, f"Found duplicate sentences: {duplicates}"


def test_load_dataset_missing_file_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        load_dataset(ROOT / "tests" / "data" / "non_existent_file.csv")


def test_load_dataset_invalid_csv_raises_valueerror(tmp_path):
    # Missing 'label' column
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("sentence,other_col\nSome sentence,foo\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV must contain 'sentence' and 'label'"):
        load_dataset(bad_csv)

    # Empty file
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(empty_csv)

    # Invalid label (e.g. 2)
    bad_label_csv = tmp_path / "bad_label.csv"
    bad_label_csv.write_text("sentence,label\nSome text,2\nOther text,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid label '2'"):
        load_dataset(bad_label_csv)

    # Single class only
    single_class_csv = tmp_path / "single.csv"
    single_class_csv.write_text("sentence,label\nText 1,1\nText 2,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="both positive"):
        load_dataset(single_class_csv)


# =========================================================================
# 2. Metric Computation & Zero-Division Safety Tests
# =========================================================================

def test_compute_metrics_standard():
    actual = [1, 1, 0, 0]
    predicted = [1, 0, 0, 1]
    m = compute_metrics(actual, predicted)
    assert m["total_samples"] == 4
    assert m["accuracy"] == 0.5
    assert m["true_positive"] == 1
    assert m["false_negative"] == 1
    assert m["false_positive"] == 1
    assert m["true_negative"] == 1
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]


def test_compute_metrics_zero_division_safety():
    # All negatives predicted
    actual = [1, 1, 0, 0]
    predicted = [0, 0, 0, 0]
    m = compute_metrics(actual, predicted)
    assert m["precision"] == 0.0  # Safe zero-division handling
    assert m["recall"] == 0.0
    assert m["f1_score"] == 0.0

    # All positives predicted
    actual = [0, 0, 0, 0]
    predicted = [1, 1, 1, 1]
    m2 = compute_metrics(actual, predicted)
    assert m2["recall"] == 0.0
    assert m2["f1_score"] == 0.0


def test_compute_metrics_length_mismatch():
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_metrics([1, 0], [1])


# =========================================================================
# 3. Stratified Metrics Tests
# =========================================================================

def test_compute_stratified_metrics():
    rows = [
        {"sentence": "s1", "label": 1, "paper_id": "P1", "category": "finding"},
        {"sentence": "s2", "label": 0, "paper_id": "P1", "category": "methodology"},
        {"sentence": "s3", "label": 1, "paper_id": "P2", "category": "finding"},
    ]
    actual = [1, 0, 1]
    predicted = [1, 0, 0]

    by_cat, by_paper = compute_stratified_metrics(rows, actual, predicted)

    assert "finding" in by_cat
    assert by_cat["finding"]["total"] == 2
    assert by_cat["finding"]["correct"] == 1
    assert by_cat["finding"]["accuracy"] == 0.5

    assert "methodology" in by_cat
    assert by_cat["methodology"]["accuracy"] == 1.0

    assert "P1" in by_paper
    assert by_paper["P1"]["accuracy"] == 1.0
    assert "P2" in by_paper
    assert by_paper["P2"]["accuracy"] == 0.0


# =========================================================================
# 4. Report Generation Tests
# =========================================================================

def test_save_reports_generates_all_files(tmp_path):
    rows = [
        {"sentence": "Sentence 1", "label": 1, "paper_id": "P1", "category": "finding", "doi": "doi1", "annotation_rationale": "r1"},
        {"sentence": "Sentence 2", "label": 0, "paper_id": "P2", "category": "methodology", "doi": "doi2", "annotation_rationale": "r2"},
    ]
    actual = [1, 0]
    predicted = [1, 0]
    metrics = compute_metrics(actual, predicted)
    metrics["dataset_type"] = "mock"

    save_reports(
        report_dir=tmp_path,
        metrics=metrics,
        rows=rows,
        actual=actual,
        predicted=predicted,
        reasons=["mock_reason", "mock_reason"],
    )

    json_file = tmp_path / "claim_extractor_metrics.json"
    csv_file = tmp_path / "claim_extractor_predictions.csv"
    png_file = tmp_path / "claim_extractor_confusion_matrix.png"

    assert json_file.exists()
    assert csv_file.exists()
    assert png_file.exists()

    # Check JSON structure
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert data["total_samples"] == 2
    assert data["accuracy"] == 1.0
    assert "confusion_matrix" in data

    # Check CSV structure
    with csv_file.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["sentence"] == "Sentence 1"
        assert reader[0]["correct"] == "True"


def test_evaluate_function_with_mock_extractor(tmp_path):
    # Create mini test dataset
    test_csv = tmp_path / "test_eval.csv"
    test_csv.write_text(
        "sentence,label,paper_id,category,doi,annotation_rationale\n"
        "This model achieves significant improvement.,1,PaperA,finding,doi:1,rationale1\n"
        "We used stochastic gradient descent.,0,PaperA,methodology,doi:1,rationale2\n",
        encoding="utf-8",
    )

    metrics = evaluate(
        dataset_path=test_csv,
        report_dir=tmp_path / "reports",
        extractor=MockExtractor(),
        verbose=False,
    )

    assert metrics["total_samples"] == 2
    assert metrics["accuracy"] == 1.0
    assert (tmp_path / "reports" / "claim_extractor_metrics.json").exists()
    assert (tmp_path / "reports" / "claim_extractor_predictions.csv").exists()
    assert (tmp_path / "reports" / "claim_extractor_confusion_matrix.png").exists()


# =========================================================================
# 5. Synthetic Dataset Separation Tests
# =========================================================================

def test_synthetic_dataset_is_separated_and_flagged():
    assert SYNTHETIC_DATASET_PATH.exists(), f"Missing synthetic test file: {SYNTHETIC_DATASET_PATH}"
    rows = load_dataset(SYNTHETIC_DATASET_PATH)
    assert len(rows) > 0
    for r in rows:
        assert r.get("is_synthetic") is True, f"Row should be flagged as synthetic: {r}"

    # Verify that the main evaluation dataset is NOT synthetic
    real_rows = load_dataset(REAL_DATASET_PATH)
    for r in real_rows:
        assert r.get("is_synthetic") is False, f"Real evaluation row flagged as synthetic: {r}"


# =========================================================================
# 6. Baseline & Confidence Interval Tests
# =========================================================================

def test_wilson_score_interval_bounds():
    from evaluate_claim_extractor import wilson_score_interval
    ci = wilson_score_interval(50, 100, confidence=0.95)
    assert 0.40 <= ci[0] < 0.50
    assert 0.50 < ci[1] <= 0.60
    assert wilson_score_interval(0, 0) == (0.0, 0.0)


def test_compute_baselines_structure():
    from evaluate_claim_extractor import compute_baselines
    sentences = [
        "Experiments show significant improvement.",
        "We trained the network using Adam optimizer.",
        "The model achieves superior accuracy.",
    ]
    actual = [1, 0, 1]
    baselines = compute_baselines(sentences, actual)

    assert "majority_class_baseline" in baselines
    assert "rule_based_keyword_baseline" in baselines
    assert "accuracy" in baselines["majority_class_baseline"]["metrics"]
    assert "accuracy" in baselines["rule_based_keyword_baseline"]["metrics"]


# =========================================================================
# 7. Paper-Level Split Integrity & Audit Artifact Tests
# =========================================================================

def test_paper_split_disjointness():
    split_file = ROOT / "reports" / "claim_extractor_paper_split.json"
    assert split_file.exists(), f"Missing split file: {split_file}"
    data = json.loads(split_file.read_text(encoding="utf-8"))
    dev_papers = set(data["development_tuning_partition"]["papers"])
    held_out_papers = set(data["diagnostic_held_out_partition"]["papers"])

    # Disjointness check
    overlap = dev_papers & held_out_papers
    assert not overlap, f"Paper split violation: papers present in both splits: {overlap}"
    assert len(dev_papers) >= 3
    assert len(held_out_papers) >= 2


def test_prediction_csv_schema_and_error_types():
    pred_file = ROOT / "reports" / "claim_extractor_predictions.csv"
    assert pred_file.exists(), f"Missing predictions file: {pred_file}"
    with pred_file.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 80
    valid_error_types = {"TP", "FP", "FN", "TN"}
    for row in reader:
        assert row["error_type"] in valid_error_types
        assert row["source_verification_status"]
        assert row["actual_label"] in ("0", "1")
        assert row["predicted_label"] in ("0", "1")


def test_all_audit_artifacts_exist_and_non_empty():
    required_artifacts = [
        ROOT / "reports" / "claim_extractor_metrics.json",
        ROOT / "reports" / "claim_extractor_predictions.csv",
        ROOT / "reports" / "claim_extractor_confusion_matrix.png",
        ROOT / "reports" / "claim_extractor_label_verification.csv",
        ROOT / "reports" / "claim_extractor_leakage_audit.csv",
        ROOT / "reports" / "claim_extractor_paper_split.json",
        ROOT / "reports" / "claim_extractor_change_log.md",
        ROOT / "reports" / "claim_extractor_audit_report.md",
    ]
    for p in required_artifacts:
        assert p.exists(), f"Missing required artifact: {p}"
        assert p.stat().st_size > 0, f"Artifact is empty: {p}"

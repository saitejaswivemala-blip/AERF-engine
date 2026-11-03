"""
scripts/validate_independent_dataset.py

Validates the independent evaluation dataset and manifest for AERF Engine:
1. Detects paper overlap with development / corpus papers.
2. Detects exact and near-duplicate sentences across development splits,
   synthetic sets, and test fixtures.
3. Validates required schemas, source citations (DOI/URL), and label ranges.
4. Verifies frozen extractor configuration integrity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Dict, List, Set, Tuple


DEVELOPMENT_PAPERS = {
    "vaswani": ["vaswani", "1706.03762", "attention is all you need"],
    "devlin": ["devlin", "1810.04805", "bert"],
    "martineau": ["martineau", "bmj.i6583", "10.1136/bmj.i6583"],
    "bolland": ["bolland", "22270409", "10.1101/2022.02.03.22270409"],
    "jolliffe_2024": ["jolliffe", "24313866", "10.1101/2024.09.18.24313866"],
    "coronavit": ["coronavit", "22271707", "10.1101/2022.03.22.22271707"],
    "zhang": ["zhang", "pmc8955452", "35334804"],
}


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join(cleaned.split())


def token_jaccard(t1: str, t2: str) -> float:
    s1 = set(normalize_text(t1).split())
    s2 = set(normalize_text(t2).split())
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def check_paper_overlap(paper_id: str, doi_or_url: str, title: str = "") -> Tuple[bool, str]:
    combined = f"{paper_id} {doi_or_url} {title}".lower()
    for dev_key, patterns in DEVELOPMENT_PAPERS.items():
        for pat in patterns:
            if pat in combined:
                return True, f"Matches development paper '{dev_key}' pattern '{pat}'"
    return False, ""


def load_development_sentences(root: Path) -> List[Tuple[str, str]]:
    dev_sentences = []
    eval_csv = root / "tests" / "data" / "claim_extractor_eval.csv"
    if eval_csv.exists():
        with open(eval_csv, mode="r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                dev_sentences.append((r["sentence"], "claim_extractor_eval.csv"))

    synthetic_csv = root / "tests" / "data" / "claim_extractor_synthetic_eval.csv"
    if synthetic_csv.exists():
        with open(synthetic_csv, mode="r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                dev_sentences.append((r["sentence"], "claim_extractor_synthetic_eval.csv"))

    diag_test = root / "tests" / "unit" / "test_claim_extractor_diagnostic.py"
    if diag_test.exists():
        content = diag_test.read_text(encoding="utf-8")
        matches = re.findall(r'"([^"\n]{25,})"', content)
        for m in matches:
            if any(w in m.lower() for w in ("model", "experiment", "training", "transformer", "attention", "ratio", "intervention")):
                dev_sentences.append((m, "test_claim_extractor_diagnostic.py"))

    return dev_sentences


def validate_manifest(manifest_path: Path) -> List[str]:
    errors = []
    if not manifest_path.exists():
        errors.append(f"Manifest missing: {manifest_path}")
        return errors

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        errors.append(f"Invalid JSON in manifest {manifest_path}: {e}")
        return errors

    if "registered_independent_papers" not in data:
        errors.append("Manifest missing 'registered_independent_papers' key")
        return errors

    papers = data["registered_independent_papers"]
    for p in papers:
        pid = p.get("paper_id", "")
        doi = p.get("doi_or_url", "")
        title = p.get("title", "")
        is_overlap, reason = check_paper_overlap(pid, doi, title)
        if is_overlap:
            errors.append(f"Manifest paper '{pid}' overlaps with development corpus: {reason}")
        if not doi:
            errors.append(f"Manifest paper '{pid}' missing DOI or source URL")
        if not p.get("scientific_discipline"):
            errors.append(f"Manifest paper '{pid}' missing scientific_discipline")

    return errors


def validate_dataset(dataset_path: Path, root: Path) -> List[str]:
    errors = []
    if not dataset_path.exists():
        errors.append(f"Dataset missing: {dataset_path}")
        return errors

    with open(dataset_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    if not reader:
        # Empty dataset is allowed when pending data collection
        return errors

    dev_sentences = load_development_sentences(root)
    seen_internal: Set[str] = set()

    for idx, row in enumerate(reader, start=1):
        sent = row.get("sentence", "").strip()
        if not sent:
            errors.append(f"Row {idx} missing sentence text")
            continue

        norm_sent = normalize_text(sent)
        if norm_sent in seen_internal:
            errors.append(f"Row {idx} is a duplicate sentence within independent dataset: '{sent[:50]}...'")
        seen_internal.add(norm_sent)

        # Check label validity
        label_raw = row.get("label", row.get("adjudicated_label", "")).strip()
        if label_raw and label_raw not in ("0", "1"):
            errors.append(f"Row {idx} has invalid label '{label_raw}'; must be '0' or '1'")

        # Check source reference
        doi = row.get("doi_or_url", row.get("doi", "")).strip()
        if not doi or doi.lower() == "unspecified":
            errors.append(f"Row {idx} missing valid DOI/URL reference")

        # Check paper overlap
        pid = row.get("paper_id", "")
        is_overlap, reason = check_paper_overlap(pid, doi)
        if is_overlap:
            errors.append(f"Row {idx} paper '{pid}' overlaps with development corpus: {reason}")

        # Check sentence leakage against development corpus
        for dev_sent, source_file in dev_sentences:
            if norm_sent == normalize_text(dev_sent):
                errors.append(f"Row {idx} is an exact duplicate of sentence in {source_file}: '{sent[:50]}...'")
                break
            jacc = token_jaccard(sent, dev_sent)
            if jacc >= 0.80:
                errors.append(f"Row {idx} has high token overlap ({jacc:.2f}) with {source_file}: '{sent[:50]}...'")
                break

    return errors


def validate_frozen_config(config_path: Path, root: Path) -> List[str]:
    errors = []
    if not config_path.exists():
        errors.append(f"Frozen configuration missing: {config_path}")
        return errors

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        errors.append(f"Invalid JSON in {config_path}: {e}")
        return errors

    hashes = data.get("source_code_integrity_hashes", {})
    for rel_path, expected_hash in hashes.items():
        target = root / rel_path
        if not target.exists():
            errors.append(f"Frozen target file missing: {target}")
            continue
        h = hashlib.sha256()
        h.update(target.read_bytes())
        actual_hash = h.hexdigest()
        if actual_hash != expected_hash:
            errors.append(f"Hash mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}")

    return errors


def run_full_validation(root: Path) -> Dict:
    manifest_path = root / "tests" / "data" / "independent_eval" / "independent_paper_manifest.json"
    dataset_path = root / "tests" / "data" / "independent_eval" / "claim_extractor_independent_eval.csv"
    config_path = root / "reports" / "frozen_extractor_config.json"

    manifest_errors = validate_manifest(manifest_path)
    dataset_errors = validate_dataset(dataset_path, root)
    config_errors = validate_frozen_config(config_path, root)

    all_errors = manifest_errors + dataset_errors + config_errors

    return {
        "valid": len(all_errors) == 0,
        "error_count": len(all_errors),
        "manifest_errors": manifest_errors,
        "dataset_errors": dataset_errors,
        "config_errors": config_errors,
        "all_errors": all_errors,
    }


def main():
    root = Path(__file__).resolve().parent.parent
    results = run_full_validation(root)
    print(json.dumps(results, indent=2))
    if not results["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

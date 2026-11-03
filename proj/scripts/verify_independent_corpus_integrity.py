"""
scripts/verify_independent_corpus_integrity.py

Automated and exhaustive verification of the 9 collected independent research papers:
1. ArXiv ID / DOI metadata resolution via official arXiv API (title, authors, publication date, categories).
2. Local PDF readability, page count, and title/author verification within PDF text.
3. SHA-256 checksum verification against manifest.
4. Occurrence verification of all 143 sampled sentences within source PDFs.
5. Duplicate sentence detection within and across papers.
6. Development-history and corpus overlap checks (ensuring 0 overlap with development papers).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

import pdfplumber
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "independent_test_corpus"
MANIFEST_PATH = ROOT / "tests" / "data" / "independent_eval" / "independent_paper_manifest.json"
SENTENCES_CSV = ROOT / "tests" / "data" / "independent_eval" / "claim_extractor_independent_eval.csv"
DEV_CSV = ROOT / "tests" / "data" / "claim_extractor_eval.csv"
DEV_MANIFEST = ROOT / "experiments" / "datasets" / "paper_manifest.json"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text_for_search(text: str) -> str:
    # Remove hyphens at line breaks, unify dashes, normalize quotes, lowercase
    text = text.replace("-\n", "").replace("­", "")
    text = re.sub(r"[–—−]", "-", text)
    text = re.sub(r"[\u2018\u2019`]", "'", text)
    text = re.sub(r'[\u201c\u201d"]', '"', text)
    text = re.sub(r"\s+", " ", text).lower()
    # Strip common punctuation differences
    return text.strip()


def query_arxiv_api(arxiv_id: str) -> dict:
    clean_id = arxiv_id.replace("arXiv:", "").strip()
    url = f"https://export.arxiv.org/api/query?id_list={clean_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "AERF-Verification-Bot/1.0"})
    
    # Retry logic
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                xml_data = resp.read()
            break
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(2)
            
    root = ET.fromstring(xml_data)
    atom_ns = "{http://www.w3.org/2005/Atom}"
    arxiv_ns = "{http://arxiv.org/schemas/atom}"
    
    entry = root.find(f"{atom_ns}entry")
    if entry is None:
        return {"found": False, "error": "No entry in atom feed"}
    
    title_elem = entry.find(f"{atom_ns}title")
    title = normalize_whitespace(title_elem.text) if title_elem is not None and title_elem.text else ""
    
    summary_elem = entry.find(f"{atom_ns}summary")
    summary = normalize_whitespace(summary_elem.text) if summary_elem is not None and summary_elem.text else ""
    
    published_elem = entry.find(f"{atom_ns}published")
    published = published_elem.text if published_elem is not None else ""
    
    authors = []
    for author_elem in entry.findall(f"{atom_ns}author"):
        name_elem = author_elem.find(f"{atom_ns}name")
        if name_elem is not None and name_elem.text:
            authors.append(name_elem.text.strip())
            
    categories = []
    for cat_elem in entry.findall(f"{atom_ns}category"):
        term = cat_elem.attrib.get("term")
        if term:
            categories.append(term)
            
    primary_cat_elem = entry.find(f"{arxiv_ns}primary_category")
    primary_cat = primary_cat_elem.attrib.get("term") if primary_cat_elem is not None else (categories[0] if categories else "")

    return {
        "found": bool(title),
        "arxiv_id": clean_id,
        "api_title": title,
        "api_authors": authors,
        "published": published,
        "primary_category": primary_cat,
        "categories": categories,
        "summary": summary[:200]
    }


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_all():
    print("=" * 80)
    print("AERF Engine Independent Evaluation Corpus Verification")
    print("=" * 80)

    # 1. Load Manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    papers = manifest_data.get("registered_independent_papers", [])
    print(f"Loaded {len(papers)} papers from manifest: {MANIFEST_PATH.name}")

    # Load 143 sampled sentences
    df_sentences = pd.read_csv(SENTENCES_CSV)
    print(f"Loaded {len(df_sentences)} sampled sentences from: {SENTENCES_CSV.name}")

    # Load Dev Papers & Sentences for Overlap Check
    dev_sentences = []
    if DEV_CSV.exists():
        df_dev = pd.read_csv(DEV_CSV)
        dev_sentences = df_dev["sentence"].tolist()
        print(f"Loaded {len(dev_sentences)} development sentences for overlap checking.")

    dev_paper_titles = []
    if DEV_MANIFEST.exists():
        with open(DEV_MANIFEST, "r", encoding="utf-8") as f:
            dev_m = json.load(f)
        for dp in dev_m.get("papers", []):
            dev_paper_titles.append(dp.get("title", "").lower())
        print(f"Loaded {len(dev_paper_titles)} development paper titles for overlap checking.")

    # Results dictionary
    paper_reports = {}
    all_passed = True

    for p in papers:
        pid = p["paper_id"]
        title = p["title"]
        authors = p["authors"]
        doi_or_url = p["doi_or_url"]
        pdf_filename = p["pdf_filename"]
        expected_sha256 = p["pdf_sha256"]
        pdf_path = CORPUS_DIR / pdf_filename

        print("\n" + "-" * 80)
        print(f"Verifying [{pid}]: {title[:60]}...")
        print(f"  Identifier: {doi_or_url} | File: {pdf_filename}")

        checks = {
            "arxiv_api_resolution": False,
            "pdf_file_exists": False,
            "pdf_readable": False,
            "pdf_matches_metadata": False,
            "sha256_checksum_match": False,
            "sentences_present_in_pdf": False,
            "no_duplicate_sentences": False,
            "no_dev_corpus_overlap": False,
        }
        evidence = {}

        # Check 1: ArXiv API Resolution
        try:
            print("  [1/7] Querying official arXiv API...")
            arxiv_res = query_arxiv_api(doi_or_url)
            if arxiv_res["found"]:
                # Check title similarity
                norm_manifest_title = normalize_text_for_search(title)
                norm_api_title = normalize_text_for_search(arxiv_res["api_title"])
                title_match = (
                    norm_manifest_title in norm_api_title
                    or norm_api_title in norm_manifest_title
                    or len(set(norm_manifest_title.split()) & set(norm_api_title.split())) / max(len(norm_manifest_title.split()), 1) > 0.8
                )
                evidence["arxiv_api"] = {
                    "api_title": arxiv_res["api_title"],
                    "api_authors": arxiv_res["api_authors"],
                    "published": arxiv_res["published"],
                    "primary_category": arxiv_res["primary_category"],
                    "title_match": title_match
                }
                if title_match:
                    checks["arxiv_api_resolution"] = True
                    print(f"    -> RESOLVED: '{arxiv_res['api_title']}' by {', '.join(arxiv_res['api_authors'][:3])}")
                else:
                    print(f"    -> TITLE MISMATCH: Manifest='{title}', API='{arxiv_res['api_title']}'")
            else:
                evidence["arxiv_api"] = {"error": arxiv_res.get("error")}
                print(f"    -> FAILED TO RESOLVE: {arxiv_res.get('error')}")
        except Exception as e:
            evidence["arxiv_api"] = {"exception": str(e)}
            print(f"    -> EXCEPTION during API query: {e}")

        # Check 2: PDF File Existence & Readability
        print("  [2/7] Checking local PDF readability and page extraction...")
        if pdf_path.exists():
            checks["pdf_file_exists"] = True
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    num_pages = len(pdf.pages)
                    # Extract full text page by page with proper spacing tolerance
                    page_texts = [page.extract_text(x_tolerance=1.5) or "" for page in pdf.pages]
                    full_pdf_text = "\n".join(page_texts)
                    first_page_text = page_texts[0] if page_texts else ""
                    
                checks["pdf_readable"] = (num_pages > 0 and len(full_pdf_text) > 1000)
                evidence["pdf_pages"] = num_pages
                evidence["pdf_total_chars"] = len(full_pdf_text)
                print(f"    -> PDF readable: {num_pages} pages, {len(full_pdf_text)} characters extracted.")
            except Exception as e:
                checks["pdf_readable"] = False
                evidence["pdf_read_error"] = str(e)
                print(f"    -> FAILED to read PDF: {e}")
                full_pdf_text = ""
                page_texts = []
        else:
            print(f"    -> PDF file does not exist: {pdf_path}")
            full_pdf_text = ""
            page_texts = []

        # Check 3: PDF matches metadata (title & authors on page 1/2)
        print("  [3/7] Verifying metadata in PDF text...")
        if full_pdf_text:
            first_pages_norm = normalize_text_for_search("\n".join(page_texts[:2]))
            norm_title_words = [w for w in normalize_text_for_search(title).split() if len(w) > 3]
            matched_words = [w for w in norm_title_words if w in first_pages_norm]
            title_word_overlap = len(matched_words) / max(len(norm_title_words), 1)
            
            # Check author presence
            author_names = [a.strip() for a in authors.replace("et al.", "").split(",") if a.strip()]
            matched_authors = []
            for a in author_names:
                parts = [p.lower() for p in re.findall(r"[a-zA-Z]{3,}", a)]
                if any(p in first_pages_norm for p in parts):
                    matched_authors.append(a)
            author_found = len(matched_authors) > 0
            
            if title_word_overlap >= 0.70 and author_found:
                checks["pdf_matches_metadata"] = True
                print(f"    -> METADATA CONFIRMED in PDF text ({title_word_overlap*100:.1f}% title word overlap, authors found: {matched_authors}).")
            else:
                print(f"    -> METADATA MISMATCH in PDF: {title_word_overlap*100:.1f}% title overlap, authors found: {matched_authors}")
            evidence["metadata_match"] = {
                "title_word_overlap": title_word_overlap,
                "matched_authors": matched_authors,
                "author_found": author_found
            }

        # Check 4: SHA-256 Checksum Verification
        print("  [4/7] Verifying SHA-256 checksum...")
        if pdf_path.exists():
            computed_sha = compute_sha256(pdf_path)
            if computed_sha.lower() == expected_sha256.lower():
                checks["sha256_checksum_match"] = True
                print(f"    -> CHECKSUM MATCH: {computed_sha[:16]}... (exact match)")
            else:
                print(f"    -> CHECKSUM MISMATCH: computed={computed_sha}, manifest={expected_sha256}")
            evidence["computed_sha256"] = computed_sha
            evidence["expected_sha256"] = expected_sha256

        # Check 5: 143 Sampled Sentences Occurrence in Source PDF
        print("  [5/7] Verifying sampled sentences occurrence in source PDF...")
        paper_sentences = df_sentences[df_sentences["paper_id"] == pid]
        evidence["sample_sentence_count"] = len(paper_sentences)
        
        norm_full_pdf = normalize_text_for_search(full_pdf_text)
        missing_sentences = []
        sentence_page_hits = []

        for _, row in paper_sentences.iterrows():
            s_id = row["sample_id"]
            s_text = row["sentence"]
            s_norm = normalize_text_for_search(s_text)
            
            # Test occurrence in full PDF
            found = False
            page_found = None
            if s_norm in norm_full_pdf:
                found = True
            else:
                # Try finding across individual pages with whitespace relaxation
                s_words = s_norm.split()
                if len(s_words) > 5:
                    prefix = " ".join(s_words[:6])
                    suffix = " ".join(s_words[-6:])
                    if prefix in norm_full_pdf and suffix in norm_full_pdf:
                        found = True

            # Determine page number
            for p_idx, p_text in enumerate(page_texts, 1):
                norm_p = normalize_text_for_search(p_text)
                if s_norm in norm_p:
                    page_found = p_idx
                    break
                s_words = s_norm.split()
                if len(s_words) > 5:
                    prefix = " ".join(s_words[:5])
                    suffix = " ".join(s_words[-5:])
                    if prefix in norm_p and suffix in norm_p:
                        page_found = p_idx
                        break

            if found:
                sentence_page_hits.append({"sentence_id": s_id, "found": True, "page": page_found})
            else:
                missing_sentences.append({"sentence_id": s_id, "text": s_text})

        if len(missing_sentences) == 0 and len(paper_sentences) > 0:
            checks["sentences_present_in_pdf"] = True
            print(f"    -> ALL {len(paper_sentences)} SENTENCES VERIFIED to appear in source PDF text.")
        else:
            print(f"    -> MISSING SENTENCES: {len(missing_sentences)} of {len(paper_sentences)} not found!")
            for ms in missing_sentences[:3]:
                print(f"       [{ms['sentence_id']}] {ms['text'][:70]}...")
        evidence["missing_sentences"] = missing_sentences
        evidence["sentence_page_hits"] = len(sentence_page_hits)

        # Check 6: Duplicate sentences within paper
        paper_s_texts = [normalize_text_for_search(t) for t in paper_sentences["sentence"].tolist()]
        duplicates = len(paper_s_texts) - len(set(paper_s_texts))
        if duplicates == 0:
            checks["no_duplicate_sentences"] = True
            print(f"    -> ZERO duplicates within {len(paper_sentences)} sampled sentences.")
        else:
            print(f"    -> FOUND {duplicates} duplicate sentences in {pid}!")
        evidence["internal_duplicates"] = duplicates

        # Check 7: Overlap with Development Papers / Sentences
        overlap_found = False
        norm_title = normalize_text_for_search(title)
        for dt in dev_paper_titles:
            if dt and (dt in norm_title or norm_title in dt):
                overlap_found = True
                print(f"    -> DEV CORPUS OVERLAP DETECTED with title: {dt}")
        for st in paper_sentences["sentence"].tolist():
            norm_st = normalize_text_for_search(st)
            for dst in dev_sentences:
                if norm_st == normalize_text_for_search(dst):
                    overlap_found = True
                    print(f"    -> DEV CORPUS SENTENCE OVERLAP DETECTED: {st[:60]}")

        if not overlap_found:
            checks["no_dev_corpus_overlap"] = True
            print(f"    -> ZERO overlap with development papers or sentences.")
        evidence["dev_overlap"] = overlap_found

        paper_pass = all(checks.values())
        if not paper_pass:
            all_passed = False
        status_str = "PASS" if paper_pass else "FAIL"
        print(f"  >>> RESULT FOR [{pid}]: {status_str} <<<")

        paper_reports[pid] = {
            "title": title,
            "doi_or_url": doi_or_url,
            "status": status_str,
            "checks": checks,
            "evidence": evidence
        }

    # Cross-paper duplicate check
    all_s_texts = [normalize_text_for_search(t) for t in df_sentences["sentence"].tolist()]
    total_dupes = len(all_s_texts) - len(set(all_s_texts))
    print("\n" + "=" * 80)
    print(f"Cross-Paper Duplicate Check: {total_dupes} duplicates across all {len(all_s_texts)} sentences.")
    print(f"Overall Independent Corpus Status: {'ALL 9 PAPERS PASSED' if all_passed else 'FAILURES DETECTED'}")
    print("=" * 80)

    # Save detailed report
    out_path = ROOT / "reports" / "corpus_integrity_verification_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "all_passed": all_passed,
                "total_papers": len(papers),
                "total_sentences": len(df_sentences),
                "cross_paper_duplicates": total_dupes,
                "papers": paper_reports
            },
            f,
            indent=2
        )
    print(f"Saved full JSON verification report to: {out_path}")
    return all_passed


if __name__ == "__main__":
    success = verify_all()
    sys.exit(0 if success else 1)

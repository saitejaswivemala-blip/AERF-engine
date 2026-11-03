"""
scripts/collect_and_register_corpus.py

Automated collection, verification, SHA-256 calculation, and blind sentence
extraction for 8-9 real research papers across diverse scientific domains.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.register_independent_paper import register_paper
from scripts.sample_sentences_from_pdf import extract_and_sample_sentences

CORPUS_DIR = Path("data/independent_test_corpus")
MANIFEST_PATH = Path("tests/data/independent_eval/independent_paper_manifest.json")
SHEET_1 = Path("tests/data/independent_eval/annotator_1_sheet.csv")
SHEET_2 = Path("tests/data/independent_eval/annotator_2_sheet.csv")
MASTER_CSV = Path("tests/data/independent_eval/claim_extractor_independent_eval.csv")

PAPERS_TO_COLLECT = [
    {
        "id": "INDEP_01_MATSCI",
        "url": "https://arxiv.org/pdf/2401.17672.pdf",
        "filename": "INDEP_01_MATSCI.pdf",
        "title": "Effect of severe plastic deformation realized by rotary swaging on the mechanical properties and corrosion resistance of near-alpha-titanium alloy Ti-2.5Al-2.6Zr",
        "authors": "V.N. Chuvil'deev, V.I. Kopylov, A.V. Nokhrin, et al.",
        "year": 2024,
        "venue": "arXiv:cond-mat.mtrl-sci",
        "doi": "arXiv:2401.17672",
        "discipline": "Materials Science & Metallurgy"
    },
    {
        "id": "INDEP_02_ASTRO",
        "url": "https://arxiv.org/pdf/2401.15548.pdf",
        "filename": "INDEP_02_ASTRO.pdf",
        "title": "Muted Features in the JWST NIRISS Transmission Spectrum of Hot-Neptune LTT 9779 b",
        "authors": "Michael Radica, David Lafreniere, et al.",
        "year": 2024,
        "venue": "arXiv:astro-ph.EP",
        "doi": "arXiv:2401.15548",
        "discipline": "Astrophysics & Planetary Science"
    },
    {
        "id": "INDEP_03_ENERGY",
        "url": "https://arxiv.org/pdf/2610.28152.pdf",
        "filename": "INDEP_03_ENERGY.pdf",
        "title": "Enhancement of Li and Na Storage in Janus HfSSe Monolayers by Co-Induced Electron-Deficient Chalcogen Sites: A First-Principles Study",
        "authors": "Md. Muktadir Hasan, Md. Mafizul Islam, Ahmed Zubair",
        "year": 2026,
        "venue": "arXiv:physics.app-ph",
        "doi": "arXiv:2610.28152",
        "discipline": "Renewable Energy & Battery Systems"
    },
    {
        "id": "INDEP_04_GEOL",
        "url": "https://arxiv.org/pdf/2610.27561.pdf",
        "filename": "INDEP_04_GEOL.pdf",
        "title": "HypoNet Nankai PS: rapid hypocenter determination for the Nankai Trough using physics-informed neural networks with three-dimensional P- and S-wave velocity models",
        "authors": "Ryoichiro Agata, Ayako Nakanishi, Dan Bassett, et al.",
        "year": 2026,
        "venue": "arXiv:physics.geo-ph",
        "doi": "arXiv:2610.27561",
        "discipline": "Structural Geology & Seismology"
    },
    {
        "id": "INDEP_05_CLIM",
        "url": "https://arxiv.org/pdf/2610.28219.pdf",
        "filename": "INDEP_05_CLIM.pdf",
        "title": "Disentangling forced and internal components in observed climate evolution using spatiotemporal deep learning",
        "authors": "Parvathi Kooloth, Wenyu Zhou, L. Ruby Leung, et al.",
        "year": 2026,
        "venue": "arXiv:physics.ao-ph",
        "doi": "arXiv:2610.28219",
        "discipline": "Atmospheric Physics & Climatology"
    },
    {
        "id": "INDEP_06_ECOL",
        "url": "https://arxiv.org/pdf/2610.26235.pdf",
        "filename": "INDEP_06_ECOL.pdf",
        "title": "The resource cost of cooperative invasion",
        "authors": "Abhishek Chowdhury",
        "year": 2026,
        "venue": "arXiv:q-bio.PE",
        "doi": "arXiv:2610.26235",
        "discipline": "Ecology & Marine Biodiversity"
    },
    {
        "id": "INDEP_07_QUANTUM",
        "url": "https://arxiv.org/pdf/2610.28462.pdf",
        "filename": "INDEP_07_QUANTUM.pdf",
        "title": "Tuning between collective dynamics and scrambling through initial-state symmetry alignment in long-range spin models",
        "authors": "Sridevi Kuriyattil, Johannes Kombe, Andrew J. Daley, et al.",
        "year": 2026,
        "venue": "arXiv:quant-ph",
        "doi": "arXiv:2610.28462",
        "discipline": "Quantum Optics & Condensed Matter"
    },
    {
        "id": "INDEP_08_PLANTBIO",
        "url": "https://arxiv.org/pdf/2405.06644.pdf",
        "filename": "INDEP_08_PLANTBIO.pdf",
        "title": "The Mathematics of Chlorophyll Photoexcitation",
        "authors": "Archit Chaturvedi",
        "year": 2024,
        "venue": "arXiv:q-bio.BM",
        "doi": "arXiv:2405.06644",
        "discipline": "Plant Biology & Photosynthesis"
    },
    {
        "id": "INDEP_09_ENVGEO",
        "url": "https://arxiv.org/pdf/2409.11848.pdf",
        "filename": "INDEP_09_ENVGEO.pdf",
        "title": "Groundwater dynamics beneath a marine ice sheet",
        "authors": "Gabriel J. Cairns, Graham P. Benham, Ian J. Hewitt",
        "year": 2024,
        "venue": "arXiv:physics.geo-ph",
        "doi": "arXiv:2409.11848",
        "discipline": "Environmental Geochemistry & Hydrology"
    }
]


def download_file(url: str, dest: Path) -> bool:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            if not data.startswith(b"%PDF"):
                print(f"  [WARN] Downloaded data from {url} is not a valid PDF header.")
                return False
            with open(dest, "wb") as f:
                f.write(data)
            return True
    except Exception as e:
        print(f"  [ERROR] Failed to download {url}: {e}")
        return False


def main():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    all_sampled_sentences = []

    print(f"=== Collecting {len(PAPERS_TO_COLLECT)} Real Research Papers ===")

    for paper_info in PAPERS_TO_COLLECT:
        pid = paper_info["id"]
        pdf_dest = CORPUS_DIR / paper_info["filename"]
        print(f"\nProcessing {pid} ({paper_info['discipline']}):")

        if not pdf_dest.exists() or pdf_dest.stat().st_size == 0:
            print(f"  Downloading from {paper_info['url']}...")
            ok = download_file(paper_info["url"], pdf_dest)
            if not ok:
                print(f"  Skipping {pid} due to download failure.")
                continue
            time.sleep(1)  # polite pacing
        else:
            print(f"  PDF already exists at {pdf_dest} ({pdf_dest.stat().st_size:,} bytes).")

        # Verify PDF validity
        with open(pdf_dest, "rb") as f:
            header = f.read(10)
            if not header.startswith(b"%PDF"):
                print(f"  [ERROR] Invalid PDF header for {pdf_dest}")
                continue

        # Register paper in manifest
        try:
            reg_entry = register_paper(
                pdf_path=pdf_dest,
                paper_id=pid,
                title=paper_info["title"],
                authors=paper_info["authors"],
                year=paper_info["year"],
                venue=paper_info["venue"],
                doi_or_url=paper_info["doi"],
                discipline=paper_info["discipline"],
                manifest_path=MANIFEST_PATH,
            )
            print(f"  Registered in manifest: SHA-256={reg_entry['pdf_sha256'][:16]}...")
        except ValueError as ve:
            if "already registered" in str(ve):
                print(f"  Already registered in manifest.")
            else:
                print(f"  [ERROR] Registration rejected: {ve}")
                continue

        # Blind sentence extraction (zero model interaction)
        try:
            sampled = extract_and_sample_sentences(
                pdf_path=pdf_dest,
                paper_id=pid,
                doi_or_url=paper_info["doi"],
                target_count=16,
                seed=42,
            )
            print(f"  Extracted {len(sampled)} blind sentences across sections.")
            all_sampled_sentences.extend(sampled)
        except Exception as e:
            print(f"  [ERROR] Failed to extract sentences from {pdf_dest}: {e}")

    print(f"\nTotal blind sentences collected across all papers: {len(all_sampled_sentences)}")

    # Write annotator sheets (leaving annotator_label empty!)
    import csv
    for sheet_path in [SHEET_1, SHEET_2]:
        with open(sheet_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sample_id", "paper_id", "sentence", "source_section", "page_number", "doi_or_url", "annotator_label", "annotator_rationale"],
            )
            writer.writeheader()
            writer.writerows(all_sampled_sentences)
        print(f"Updated {sheet_path} with {len(all_sampled_sentences)} candidate rows (labels blank).")

    # Update master claim_extractor_independent_eval.csv (labels blank!)
    with open(MASTER_CSV, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sample_id", "paper_id", "sentence", "source_section", "page_number",
            "annotator_1_label", "annotator_1_rationale", "annotator_2_label", "annotator_2_rationale",
            "adjudicated_label", "label", "category", "doi_or_url", "verification_status", "adjudication_notes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_sampled_sentences:
            writer.writerow({
                "sample_id": s["sample_id"],
                "paper_id": s["paper_id"],
                "sentence": s["sentence"],
                "source_section": s["source_section"],
                "page_number": s["page_number"],
                "annotator_1_label": "",
                "annotator_1_rationale": "",
                "annotator_2_label": "",
                "annotator_2_rationale": "",
                "adjudicated_label": "",
                "label": "",
                "category": s["source_section"].lower(),
                "doi_or_url": s["doi_or_url"],
                "verification_status": "VERIFIED_OPEN_ACCESS_PDF",
                "adjudication_notes": "Pending independent human annotation",
            })
    print(f"Updated {MASTER_CSV} with {len(all_sampled_sentences)} candidate rows (labels blank).")


if __name__ == "__main__":
    main()

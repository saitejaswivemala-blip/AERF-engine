# Dataset Provenance, Data Leakage Audit & Reproducibility Protocol

**Document Purpose**: Full audit of dataset composition, annotation provenance, exposure history, and step-by-step reproduction guide.  
**Auditor**: Antigravity Technical & Research Methodology Audit  
**Date**: November 01, 2026  
**Artifact Directory**: `FINAL_RESEARCH_AUDIT/`  

---

## 1. Complete Dataset Inventory

| Dataset Identifier | File Path | Number of Sentences | Number of Source Papers | Annotation Method | Human Review Status | Verification SHA-256 Hash |
|:---|:---|:---:|:---:|:---|:---:|:---:|
| **Development Benchmark** | `tests/data/claim_extractor_eval.csv` | 80 | 6 foundational papers | Expert Curation via Guidelines | Single Annotator (100% Human Reviewed) | `6d57c285fa650f4c47384ab84708a4caab753720ba1b936a48ee4aa974c44354` |
| **Independent Stress Test** | `tests/data/independent_eval/claim_extractor_independent_provisional_eval.csv` | 143 | 9 unseen papers | Dual-Model AI Consensus | **0 / 143 (0.0% Human Reviewed)** | `ef9cad55f157c5e232e239ceea7f8d2c8387e7c05d7fbf34f53e9f7ada7d2f58` |
| **Synthetic Unit Test Set** | `tests/data/claim_extractor_synthetic_eval.csv` | 16 | Synthetic templates | Rule Test Fixtures | Automated Unit Test Only | `47c62bb926f7a7da932d56a3e144d1565576a80e157dbb6982fec9ee364f3d2f` |
| **Global System Corpus** | `experiments/runs/experiment_FINAL_2026-09-02.db` | 816 claims | 11 full papers | Real Pipeline Extraction | Intact Benchmark Database | *(8,241,152 bytes)* |
| **Curated Claim Pairs** | `experiments/annotations/ground_truth_claim_pairs.json` | 8 claim pairs | 5 clinical papers | Manual Literature Review | Expert Curated | `1b10a2f1b4020a4be35948f95c69dd5e0dc655ae25178dc44ea7fe7ae01b9758` |

---

## 2. Source Papers Breakdown

### A. Development Corpus (6 Foundation Papers across NLP & Clinical Medicine)
1. **Vaswani et al. (2017)**: *Attention Is All You Need* (arXiv:1706.03762v7) — 16 sentences.
2. **Devlin et al. (2018)**: *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (arXiv:1810.04805v2) — 11 sentences.
3. **Bolland et al. (2014, 2022)**: *Vitamin D supplementation in acute respiratory infections: meta-analysis* — 16 sentences.
4. **Martineau et al. (2017)**: *Vitamin D for the prevention of acute respiratory infections* (BMJ 2017;356:i6583) — 16 sentences.
5. **Jolliffe et al. (2021, 2024)**: *Vitamin D Status and Susceptibility to COVID-19 / ARI* — 11 sentences.
6. **Zhang et al. (2023)**: *Vitamin D supplementation in COPD exacerbations* — 10 sentences.

### B. Independent Corpus (9 Diverse Physical and Natural Science Papers)
1. `INDEP_01_MATSCI` (Corrosion Science, 2023) — 15 sentences.
2. `INDEP_02_ASTRO` (The Astronomical Journal, 2023) — 16 sentences.
3. `INDEP_03_ENERGY` (Nature Communications, 2023) — 16 sentences.
4. `INDEP_04_GEOL` (Journal of Geophysical Research, 2023) — 16 sentences.
5. `INDEP_05_CLIM` (Journal of Climate, 2023) — 16 sentences.
6. `INDEP_06_ECOL` (Proceedings of the Royal Society B, 2023) — 16 sentences.
7. `INDEP_07_QUANTUM` (Physical Review B, 2023) — 16 sentences.
8. `INDEP_08_PLANTBIO` (Biophysical Journal, 2023) — 16 sentences.
9. `INDEP_09_ENVGEO` (Journal of Glaciology, 2023) — 16 sentences.

---

## 3. Data Leakage and Development Exposure Audit

A thorough code and commit history audit (`reports/claim_extractor_leakage_audit.csv`) examined whether evaluation sentences were exposed during model development:

1. **Development Set Contamination**:
   - Sentences from Vaswani et al. (2017) and Devlin et al. (2018) were used directly during regex rule engineering in `nlp/claim_extractor.py` and unit test fixtures in `tests/unit/test_claim_extractor_diagnostic.py`.
   - Sample 8 is a verbatim copy of a background unit test fixture.
   - Phrases such as `"28.4 BLEU"` appear verbatim in code comments and regex definitions.
   - **Conclusion**: The 80-sentence collection must be strictly classified as a **Development / Diagnostic Validation Set**. It cannot be claimed as an untouched, independent test set.
2. **Independent Set Exposure**:
   - Sentences from the 143-sentence corpus were read by diagnostic failure-analysis scripts (`proj/scratch/test_candidates.py` and `reports/independent_eval/error_analysis.csv`) to identify why the baseline extractor suffered recall collapse on non-biomedical domains.
   - The resulting additions (e.g., physical units such as eV, mAh, MPa and gradient comparatives) were added to resolve this deficit.
   - **Conclusion**: The 143-sentence collection functions methodologically as an **Out-of-Domain Cross-Discipline Stress Test**, not a pre-registered, blinded confirmatory benchmark.

---

## 4. Step-by-Step Reproduction Guide

All experimental results can be reproduced deterministically using the commands below:

### Environment Setup & Prerequisites
```powershell
# Navigate to project root
cd C:\Users\tejas\Downloads\AERF-engine-FIXED\proj

# Verify Python version (Requires Python 3.12 64-bit)
& 'C:\Program Files\Python312\python.exe' --version
# Expected: Python 3.12.x

# Verify core dependencies
& 'C:\Users\tejas\Downloads\AERF-engine-FIXED\.venv\Scripts\pip.exe' show transformers scikit-learn torch spacy
# Expected: transformers==4.57.6, scikit-learn==1.4.2, torch==2.6.0+cpu, spacy==3.7.5
```

### Execution Commands

```powershell
# Step 1: Run the full final comparative experimental evaluation
& 'C:\Program Files\Python312\python.exe' scripts/run_final_experimental_comparison.py
# Verification: Generates reports in reports/final_evaluation/ and exits with code 0.

# Step 2: Verify baseline and ablation experiments across isolated SQLite databases
& 'C:\Program Files\Python312\python.exe' scripts/evaluate_curated_mapping.py
# Verification: Evaluates all 7 baseline/ablation configs and logs results to curated_comparison_real_models.json.

# Step 3: Execute independent evaluation protocol test suite
& 'C:\Program Files\Python312\python.exe' -m pytest tests/unit/test_independent_evaluation_protocol.py -v
# Verification: 17 passed in <1s (confirms SHA-256 hashes, non-overlap, and schema integrity).

# Step 4: Run complete project unit and integration test suite
& 'C:\Program Files\Python312\python.exe' -m pytest tests/unit tests/integration -q
# Verification: 188 passed in ~40s (100% test pass rate).

# Step 5: Verify mathematical consistency across JSON, CSV, and markdown artifacts
& 'C:\Program Files\Python312\python.exe' scripts/verify_artifact_consistency.py
# Verification: Exit code 0; confirms 100% agreement across all metric tables.
```

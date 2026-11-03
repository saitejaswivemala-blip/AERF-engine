# AERF Scientific Claim Extractor — Reproducibility & Replication Guide

## 1. Environment & Dependencies
- **OS**: Windows 11 (64-bit)
- **Python**: 3.12.3
- **Core Libraries**:
  - `spacy == 3.7.5` (`en_core_sci_sm`)
  - `transformers == 5.16.1`
  - `torch == 2.6.0+cpu`
  - `scikit-learn == 1.7.1`
  - `pdfplumber == 0.11.9`
- **Cached Models**:
  - `valhalla/distilbart-mnli-12-3` (Primary zero-shot classifier in `claim_extractor.py`)
  - `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (Independent validation model)

---

## 2. Dataset Provenance & Separation

1. **Development Dataset** (`tests/data/claim_extractor_eval.csv`):
   - **Sample Count**: 80 sentences (34 claims, 46 non-claims).
   - **Provenance**: Verbatim extractions from 7 published papers (Vaswani et al. 2017, Devlin et al. 2018, Martineau et al. 2017, Bolland et al. 2022, Jolliffe et al. 2024, Jolliffe et al. 2022, Zhang et al. 2022).
   - **Use**: Model development, rule formulation, ablation benchmarking.
2. **Independent Test Corpus** (`data/independent_test_corpus/`):
   - **Sample Count**: 143 sentences from 9 open-access research PDFs across 9 physical and natural science disciplines.
   - **Provenance**: Downloaded from official arXiv open access, SHA-256 verified, 0 author/paper overlap with development corpus.
   - **Status**: Labeled via dual AI consensus (`PROVISIONAL_AI_PREDICTION_UNVERIFIED`).

---

## 3. Exact Commands to Reproduce All Results

### A. Run Development Dataset Benchmark (AERF 2.0 Extractor):
```powershell
python evaluate_claim_extractor.py --dataset tests/data/claim_extractor_eval.csv --report-dir reports
```

### B. Run Independent Provisional Evaluation:
```powershell
python evaluate_claim_extractor.py --dataset tests/data/independent_eval/claim_extractor_independent_provisional_eval.csv --report-dir reports/independent_eval
```

### C. Run Full Diagnostic & Regression Unit Test Suite:
```powershell
python -m pytest tests/unit/test_claim_extractor_diagnostic.py tests/unit/test_evaluate_claim_extractor.py tests/unit/test_independent_evaluation_protocol.py -v
```

### D. Run Final Comparative & Ablation Suite:
```powershell
python scripts/run_final_experimental_comparison.py
```

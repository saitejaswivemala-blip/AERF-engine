# AERF Engine — Independent Evaluation Protocol

**Document Version**: 1.0.0  
**Effective Date**: October 21, 2026  
**Status**: Active Protocol  
**Scope**: Scientific Claim Extractor (`ScientificClaimExtractor` in `nlp/claim_extractor.py`)

---

## 1. Executive Rationale & Academic Defense

The completed audit of the AERF claim extractor established that the 80-sample dataset (`tests/data/claim_extractor_eval.csv`) cannot serve as an independent test set due to confirmed development-coupled leakage:
- **Vaswani et al. (2017)** sentences were directly used to hand-craft heuristic rescue regexes (`_STRONG`, `_METRIC_NUMBER`, `_CONTRIBUTION`) and unit test fixtures in `tests/unit/test_claim_extractor_diagnostic.py`.
- **Jolliffe et al. (2024)**, **Bolland et al. (2022)**, **CORONAVIT (2022)**, and **Zhang et al. (2022)** formed the project's primary experiment corpus (`experiments/datasets/paper_manifest.json`) and ground-truth mapping targets.
- **Devlin et al. (2018)** and **Martineau et al. (2017)** share the exact vocabulary and clinical lineage targeted by the heuristic rescue rules.

Consequently, the existing 83.75% accuracy and 83.54% F1 metrics represent **development and diagnostic performance**, not proof of generalized performance across scientific literature.

This document establishes the binding protocol for collecting, vetting, annotating, adjudicating, and evaluating a **genuinely independent, out-of-domain evaluation corpus**.

---

## 2. Ingestion Pipeline Segregation & Anti-Contamination Rules

### Current Ingestion Architecture
In the AERF Engine:
1. `ingestion/pdf_loader.py` loads PDFs page-by-page.
2. `ingestion/text_cleaner.py` and `ingestion/chunker.py` create text chunks.
3. `pipeline/orchestrator.py` ingests chunks into SQLite database `experiments/runs/experiment.db`.
4. `pipeline/phase2.py` runs `ScientificClaimExtractor` over all chunks and persists extracted claims into `claims_p2`.
5. `pipeline/phase3.py` matches claims across papers to detect support/contradiction.

### Anti-Contamination Rules
To prevent contamination of the independent test set:

1. **Storage Isolation**: Independent evaluation PDFs must NEVER be placed in `data/raw_pdfs/` or `experiments/datasets/pdfs/`. They must be staged in an isolated directory:
   `data/independent_test_corpus/`
2. **Database Isolation**: Independent evaluation PDFs must NEVER be ingested into `experiments/runs/experiment.db` or processed by `pipeline/orchestrator.py` during model development.
3. **Model & Rule Freeze**: The extractor configuration (`reports/frozen_extractor_config.json`) is locked prior to evaluation. Once sentences from the independent set are extracted or viewed, **no developer is permitted to modify regexes, thresholds, prompts, or model weights** in response to errors.
4. **Single-Pass Evaluation**: Independent evaluation is conducted strictly as a one-shot forward pass. Errors are analyzed post-hoc as reported limitations, never as opportunities for rule tuning.

---

## 3. Paper Selection Criteria (Unseen Scientific Disciplines)

Independent test papers must satisfy all of the following requirements:

1. **Domain Diversity**: Selected papers must come from scientific disciplines completely outside the two development domains (Transformer NLP and Vitamin D clinical trials). Required target domains:
   - Materials Science & Metallurgy
   - Astrophysics & Planetary Science
   - Plant Biology & Agronomy
   - Environmental Chemistry & Oceanography
   - Renewable Energy & Battery Systems
   - Structural Geology & Geophysics
2. **Exclusion Check**: Papers must not be written by authors of the existing corpus (e.g., Vaswani, Devlin, Martineau, Jolliffe, Bolland) and must have zero appearance in git logs, test fixtures, or project manifests.
3. **Publication Standard**: Peer-reviewed journal publications or verified open-access preprints with valid, permanent DOIs.
4. **Full-Text Accessibility**: Verifiable, open-access PDFs where every sentence can be confirmed verbatim against the published text.
5. **SHA-256 Checksum**: The source PDF must have its SHA-256 hash recorded in `tests/data/independent_eval/independent_paper_manifest.json`.

---

## 4. Sentence Extraction Protocol

For each selected paper, extract 15–20 representative sentences using stratified sampling:

| Section | Target Count | Categories to Sample |
|---|---|---|
| **Abstract** | 3–4 sentences | Primary findings, overall conclusion, background problem |
| **Introduction** | 2–3 sentences | Domain background, accepted facts, prospective research aims |
| **Methods / Experimental** | 3–4 sentences | Methodology, model parameters, equipment, dataset descriptions |
| **Results** | 4–6 sentences | Quantitative statistical results, null/negative findings, comparative findings |
| **Discussion / Conclusion** | 2–3 sentences | Limitations, study caveats, future work directions, boundary claims |

### Extraction Rules:
- **Verbatim Fidelity**: Extract the exact character sequence from the primary PDF text.
- **Provenance Recording**: Record the exact section header and PDF page number.
- **Sentence Integrity**: Do not split or join sentences artificially. If PDF extraction creates hyphenation or line breaks, record normalization explicitly.

---

## 5. Dual-Annotator Review & Adjudication Workflow

To achieve academic publication standard (e.g. Scopus / Web of Science indexed journals), sentence annotations must be established via independent dual annotation with inter-rater reliability reporting:

```
[Candidate Sentence from Unseen Paper]
               │
       ┌───────┴───────┐
       ▼               ▼
[Annotator 1]    [Annotator 2]
(Blind Label: 0/1) (Blind Label: 0/1)
       │               │
       └───────┬───────┘
               ▼
   [Inter-Rater Agreement Check]
   - Raw Agreement %
   - Cohen's Kappa (κ)
               │
       ┌───────┴───────┐
  Agreement?           Disagreement?
       │                       │
       ▼                       ▼
[Accepted Label]       [Adjudication Panel]
                       (Senior Reviewer / Consensus)
                               │
                               ▼
                       [Adjudicated Label]
                               │
                               ▼
            [Frozen Independent Dataset CSV]
```

### Protocol Steps:
1. **Blind Annotation**:
   - Annotator 1 and Annotator 2 independently annotate each sentence following `docs/CLAIM_ANNOTATION_GUIDELINES.md`.
   - Annotators record binary labels (`1` = finding/result, `0` = non-finding) and written rationales without communicating.
2. **Inter-Rater Reliability Calculation**:
   - Calculate raw percent agreement:
     $$p_o = \frac{\text{agreed samples}}{N}$$
   - Calculate expected agreement by chance ($p_e$) and Cohen's Kappa ($\kappa$):
     $$\kappa = \frac{p_o - p_e}{1 - p_e}$$
   - Report $\kappa$ transparently. High agreement ($\kappa \ge 0.80$) establishes reliable operational guidelines.
3. **Adjudication of Discrepancies**:
   - Automated script `scripts/adjudicate_independent_annotations.py` extracts all rows where `annotator_1_label != annotator_2_label`.
   - An independent third adjudicator reviews the sentence, source context, and both rationales to assign the final `adjudicated_label` with justification.
4. **Dataset Assembly**:
   - Adjudicated sentences are compiled into `tests/data/independent_eval/claim_extractor_independent_eval.csv`.

---

## 6. Execution Commands & Reporting

When the independent dataset is compiled and validated:

```powershell
# 1. Validate the independent dataset and manifest integrity
python scripts/validate_independent_dataset.py

# 2. Evaluate the frozen extractor and baselines on the independent dataset
python evaluate_claim_extractor.py `
    --dataset tests/data/independent_eval/claim_extractor_independent_eval.csv `
    --report-dir reports/independent_eval

# 3. Run validation tests
python -m pytest tests/unit/test_independent_evaluation_protocol.py -v
```

---

## 7. Current Status

**Independent Evaluation Status**: `PENDING_UNSEEN_DATA_COLLECTION`

No fake, synthetic, or reused data is populated in the independent test location. The architecture, validation tooling, manifest, and annotation protocol are established and frozen, awaiting real multi-domain paper ingestion by human annotators.

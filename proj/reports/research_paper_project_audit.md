# AERF Engine: Comprehensive Research-Paper Project Audit Report

**Project Path**: `C:\Users\tejas\Downloads\AERF-engine-FIXED\proj`  
**System Name**: Adaptive Evidence Reasoning Framework (AERF) Engine  
**Audit Date**: October 26, 2026  
**Document Purpose**: Factual reference report for scientific paper preparation  

---

## 1. Problem the Project Addresses

Scientific literature grows at millions of publications annually, making manual synthesis across papers infeasible. Existing automated evidence extraction and reasoning systems exhibit two major research gaps:

1. **The Epistemic Extraction Gap**: Standard zero-shot Natural Language Processing (NLP) models (e.g., zero-shot NLI) and keyword extractors fail to distinguish *epistemic status*. They misclassify hypotheses (*"We hypothesized that..."*), background literature reviews (*"Prior work by Smith et al. showed..."*), experimental procedures (*"We trained for 50 epochs..."*), and study limitations as empirical findings because the grammatical structure resembles an assertion. Conversely, supervised biomedical classifiers (e.g., PubMed-RCT) are computationally heavy and fail to generalize outside clinical medicine.
2. **The Dynamic Evidence Synthesis Gap**: Traditional scientific knowledge graphs link papers via coarse document-level citations. Automated claim-verification frameworks (e.g., SciFact, FEVER) evaluate isolated claim-evidence pairs statically. They lack continuous, multi-dimensional reliability scoring, dynamic evidence-mass propagation across interconnected claim networks, and automated detection of unresolved scientific contradictions.

**AERF addresses these problems by providing an end-to-end framework that**:
- Extracts fine-grained scientific claims with exact byte-level provenance using a lightweight, discourse-gated hybrid zero-shot model.
- Aligns and validates semantic relationships across papers using two-stage cosine-gated Natural Language Inference (NLI).
- Computes a deterministic, bounded 7-factor reliability score for each claim.
- Adaptively propagates evidence mass across the claim graph when new findings are ingested.
- Automatically discovers research gaps, separating uncorroborated assertions from contested scientific contradictions.
- Enforces session-scoped isolation, allowing multi-paper uploads without corrupting the global scientific database.

---

## 2. Implemented Features and System Architecture

The implemented architecture is organized into three pipeline phases, an adaptive scoring engine, a graph layer, and an interactive session UI:

```
Scientific PDF 
   │
   ▼ [Phase 1: Ingestion & Provenance]
Text Cleaning -> Character-Span Chunking -> SQLite Storage
   │
   ▼ [Phase 2: Entity, Claim & Relation Extraction]
SciNER (en_core_sci_sm) + Discourse-Gated Zero-Shot Claim Extraction + Relation Extraction + MiniLM-L6 Embeddings
   │
   ▼ [Phase 3: Cross-Paper Reasoning & Evidence Synthesis]
Cosine Similarity Gating (0.55) -> Entity Overlap Filter -> DeBERTa-v3 NLI Validation
   │
   ├──> 7-Factor AERF Reliability Scorer
   ├──> Adaptive Neighborhood Updater (Iterative Score Propagation)
   ├──> Topological Evidence Graph (Claim-Entity-Relation Edges)
   ├──> Temporal Evolution Tracking
   └──> Automated Research Gap Detection
   │
   ▼ [Session & Presentation Layer]
Session Manager (Isolated SQLite DBs) + Streamlit Interactive Dashboard
```

### Distinction of Implemented vs. Planned / Incomplete Features

| Pipeline Component | Module Path | Implemented Features | Incomplete / Planned Features | Implementation Status |
|:---|:---|:---|:---|:---:|
| **Ingestion & Provenance** | `ingestion/pdf_loader.py`<br>`ingestion/chunker.py`<br>`ingestion/text_cleaner.py` | PDF text extraction, column de-hyphenation, non-destructive cleaning, chunking with start/end character offsets into SQLite. | Table extraction and figure image parsing are text-only; does not parse vector graphics. | **IMPLEMENTED** |
| **Named Entity Recognition** | `nlp/ner.py` | Biomedical/scientific entity extraction using `en_core_sci_sm`. Entities linked to chunk offsets. | Entity linking to external ontologies (e.g., UMLS, Wikidata) is not implemented. | **IMPLEMENTED** |
| **Claim Extraction** | `nlp/claim_extractor.py` | Lightweight discourse-gated hybrid extractor: DistilBART-MNLI multi-class zero-shot backbone + epistemic discourse router + result cue rescue. Exact byte spans. | Cross-sentence claim coreference resolution is not implemented. | **IMPLEMENTED** |
| **Relation Extraction** | `nlp/relation_extractor.py` | Dependency parse pattern-matching over claim spans linking subject and object entities via predicate verbs. | Neural relation extraction (e.g., fine-tuned RE transformer) is not implemented; relies on spaCy dependency rules. | **IMPLEMENTED (Rule-Based)** |
| **Claim Embeddings** | `nlp/embeddings.py` | 384-dimensional dense semantic representations using `sentence-transformers/all-MiniLM-L6-v2`. | Multi-task sentence embeddings or fine-tuned scientific encoders (Specter) not integrated. | **IMPLEMENTED** |
| **Cross-Paper Alignment** | `matching/claim_matcher.py` | Two-stage matching: Vector dot-product cosine pre-filter ($\ge 0.55$) + Jaccard entity overlap ($\ge 0.15$). | Cross-lingual claim alignment is not supported. | **IMPLEMENTED** |
| **NLI Validation** | `nlp/nli/nli_classifier.py` | Cross-encoder inference using `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. Validates `supports`, `contradicts`, `neutral`. | Stance calibration for hedging/modal verbs is handled via thresholding ($\ge 0.60$). | **IMPLEMENTED** |
| **Reliability Scoring** | `aerf/scorer.py` | Deterministic closed-form 7-factor calculation bounded in $[0, 1]$. | Weights are currently configured heuristics, not learned via meta-learning. | **IMPLEMENTED** |
| **Adaptive Graph Update** | `aerf/adaptive_updater.py` | Dynamic neighborhood re-scoring upon arrival of new claims/evidence pairs. | Distributed asynchronous graph propagation (e.g., Celery/Kafka) not implemented. | **IMPLEMENTED** |
| **Evidence Graph** | `graph/evidence_graph.py` | Typed graph construction: claim nodes, entity nodes, typed edges (`mentions`, `predicate`, `supports`, `contradicts`, `neutral`). | Graph neural network (GNN) link prediction is not implemented. | **IMPLEMENTED** |
| **Research Gap Detection** | `gaps/gap_detector.py` | Automated discovery of isolated uncorroborated claims ($F_{\text{corrob}} < 0.35$) and contested findings ($|M_{\text{supp}} - M_{\text{contra}}| \le 0.15$). | Automated hypothesis generation from discovered gaps is planned/future work. | **IMPLEMENTED** |
| **Temporal Evolution** | `evolution/timeline_builder.py` | Tracks longitudinal claim consensus, decay, and stance shifts across publication years. | Citation velocity modeling is not implemented. | **IMPLEMENTED** |
| **Session Isolation** | `pipeline/session_manager.py` | Dynamic routing of multi-paper analysis sessions to isolated SQLite databases; global corpus DB preserved. | Multi-user cloud tenancy / authentication not implemented. | **IMPLEMENTED** |
| **User Interface** | `ui/app.py` | Streamlit dashboard: Multi-PDF upload, graph exploration, claim inspection, gap review, and session switching. | Export to LaTeX or automated systematic review PRISMA flowchart not implemented. | **IMPLEMENTED** |

---

## 3. The Seven Reliability-Scoring Factors and Their Calculation

The AERF score for a claim $c$ is deterministically computed in [`aerf/scorer.py`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/aerf/scorer.py):
$$R(c) = \operatorname{clip}\left(\sum_{k \in \mathcal{K}_{\text{pos}}} w_k F_k(c) - w_{\text{contra}} F_{\text{contra}}(c), \, 0.0, \, 1.0\right)$$
Where weights $w = \{ \text{source}: 0.10, \text{methodology}: 0.15, \text{corroboration}: 0.20, \text{contradiction}: 0.20, \text{temporal}: 0.10, \text{semantic}: 0.15, \text{provenance}: 0.10 \}$.

```
                 ┌─────────────────────────────────────────────────────────────┐
                 │                  AERF Reliability Score                     │
                 │                 R(c) in [0.0, 1.0]                          │
                 └──────────────────────────────┬──────────────────────────────┘
                                                │
         ┌──────────────┬──────────────┬────────┴─────┬──────────────┬──────────────┬──────────────┐
         ▼              ▼              ▼              ▼              ▼              ▼              ▼
   +0.10 Source   +0.15 Method   +0.20 Corrob   -0.20 Contra   +0.10 Temporal +0.15 Semantic +0.10 Provenance
   Metadata       Study Rigor    Support Mass   Conflict Mass  Decay Window   NLI Confidence SQLite Offset
```

### Exact Mathematical Definitions:

1. **Source Factor ($F_{\text{source}} \in [0, 1]$)**:
   Measures metadata completeness of the publishing paper:
   $$F_{\text{source}} = \frac{\mathbb{I}(\text{title}) + \mathbb{I}(\text{authors}) + \mathbb{I}(\text{year}) + \mathbb{I}(\text{venue})}{4}$$
2. **Methodology Factor ($F_{\text{methodology}} \in [0, 1]$)**:
   Measures experimental rigor based on design, sample size, and statistical reporting:
   $$F_{\text{methodology}} = \operatorname{clip}(0.4 \cdot \text{design} + 0.3 \cdot \text{sample} + 0.3 \cdot \text{stats}, 0.0, 1.0)$$
   - $\text{design} = 1$ if text matches: `randomized`, `randomised`, `controlled`, `cohort`, `case-control`, `meta-analysis`, `systematic review`.
   - $\text{sample} = 1$ if text contains numbers and matches: `n=`, `n =`, `sample`, `patients`, `participants`.
   - $\text{stats} = 1$ if text contains: `p <`, `p=`, `p =`, `confidence interval`, `95% ci`, `odds ratio`, `hazard ratio`.
3. **Corroboration Factor ($F_{\text{corrob}} \in [0, 1]$)**:
   Non-linear saturating evidence mass from validated cross-document supporting claims:
   $$F_{\text{corrob}} = \operatorname{clip}\left(\frac{M_{\text{support}}}{k + M_{\text{support}}}, 0.0, 1.0\right), \quad M_{\text{support}} = \sum_{n \in \mathcal{N}_{\text{supp}}} R(n) \cdot P_{\text{NLI}}(n)$$
   Where smoothing constant $k = 2.0$.
4. **Contradiction Factor ($F_{\text{contra}} \in [0, 1]$)**:
   Non-linear saturating evidence mass from validated cross-document contradicting claims:
   $$F_{\text{contra}} = \operatorname{clip}\left(\frac{M_{\text{contra}}}{k + M_{\text{contra}}}, 0.0, 1.0\right), \quad M_{\text{contra}} = \sum_{n \in \mathcal{N}_{\text{contra}}} R(n) \cdot P_{\text{NLI}}(n)$$
   Where $k = 2.0$. This term enters with a negative coefficient ($-0.20$).
5. **Temporal Factor ($F_{\text{temporal}} \in [0, 1]$)**:
   Linear recency decay over a 10-year sliding window ($W = 10.0$):
   $$F_{\text{temporal}} = \begin{cases} 0.5, & \text{if year is None or no year history} \\ \max\left(0.0, \, 1.0 - \frac{Y_{\max} - Y_c}{W}\right), & \text{otherwise} \end{cases}$$
6. **Semantic Alignment Factor ($F_{\text{semantic}} \in [0, 1]$)**:
   Highest validated NLI confidence score among linked pairs:
   $$F_{\text{semantic}} = \operatorname{clip}\left(\max_{p \in \mathcal{P}(c)} P_{\text{NLI}}(p), 0.0, 1.0\right) \quad (\text{defaults to } 0.5 \text{ if isolated})$$
7. **Provenance Factor ($F_{\text{provenance}} \in [0, 1]$)**:
   Verifies physical traceability: checks whether `chunk_id`, `page_number`, `start_char`, and `end_char` offsets are validly indexed in SQLite.

---

## 4. Algorithms and Models Actually Used

| Task | Model / Algorithm | Provenance / Implementation | Parameters / Thresholds |
|:---|:---|:---|:---|
| **Text Segmentation** | Custom multi-pass regex + spaCy sentencizer | [`nlp/claim_extractor.py`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/nlp/claim_extractor.py) | Abbreviation-boundary merging (`et al.`, `Fig.`), glued-heading prefix detachment, minimum length $\ge 25$ chars, alpha ratio $\ge 0.55$. |
| **Named Entity Extraction** | `en_core_sci_sm` | spaCy scientific model | Token POS tagger + transition-based NER parser. |
| **Zero-Shot Claim Classification** | `valhalla/distilbart-mnli-12-3` | Hugging Face Transformers | Hypothesis template: `"This sentence {}."`<br>Labels: 4 multi-class labels.<br>Threshold: $\tau = 0.35$. |
| **Epistemic Discourse Routing** | Deterministic Regex Rules | [`nlp/claim_extractor.py`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/nlp/claim_extractor.py) | Suppression guards: `_HYPOTHESIS_OBJECTIVE`, `_EXTERNAL_LITERATURE`, `_DECLARATIONS_AND_REFS`, `_CONFIG_SPEC`, `_PROCEDURAL_METHOD`, `_LIMITATION_FUTURE`. |
| **Result Cue Rescue** | Multi-domain scientific pattern rescue | [`nlp/claim_extractor.py`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/nlp/claim_extractor.py) | Cues: `_STRONG` (comparatives, statistical indicators), `_EMPIRICAL_OBSERVATION` (units: eV, mAh, MPa), `_CONTRIBUTION` (system-subject novelty assertions). Rescue threshold: $0.10 \le f \le 0.15$. |
| **Sentence Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | PyTorch / Transformers | 384 dimensions, cosine distance metric. |
| **Cross-Paper NLI** | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | PyTorch / Transformers | Max sequence length 512, validation threshold $\ge 0.60$. |
| **Graph Edge Routing** | Graph Builder Algorithm | [`graph/evidence_graph.py`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/graph/evidence_graph.py) | Claim-Entity, Entity-Relation, and Claim-Claim typed directed edges. |
| **Gap Detection** | Contradiction Margin Algorithm | [`gaps/gap_detector.py`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/gaps/gap_detector.py) | Low corroboration cutoff: $F_{\text{corrob}} < 0.35$.<br>Contradiction margin: $|M_{\text{supp}} - M_{\text{contra}}| \le 0.15$. |

---

## 5. Datasets and Number of Papers/Samples Evaluated

### 1. Development/Validation Benchmark ($N=80$)
- **File**: [`tests/data/claim_extractor_eval.csv`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/tests/data/claim_extractor_eval.csv)
- **Provenance**: Verified Human Expert Ground Truth.
- **Corpus Composition**: 80 sentences sampled from 6 foundation papers across Natural Language Processing and Clinical Medicine:
  1. *Attention Is All You Need* (Vaswani et al., 2017)
  2. *BERT: Pre-training of Deep Bidirectional Transformers* (Devlin et al., 2018)
  3. *Vitamin D Supplementation in ARI* (Bolland et al., 2014)
  4. *Vitamin D for Prevention of Acute Respiratory Infections* (Martineau et al., 2017)
  5. *Vitamin D Status and Susceptibility to COVID-19* (Jolliffe et al., 2021)
  6. *Vitamin D in COPD Exacerbations* (Zhang et al., 2023)
- **Class Balance**: 34 Claims (42.5%), 46 Non-claims (57.5%).
- **Role**: Used exclusively for development validation, threshold calibration, and rule debugging.

### 2. Independent Provisional Benchmark ($N=143$)
- **File**: [`tests/data/independent_eval/claim_extractor_independent_provisional_eval.csv`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/tests/data/independent_eval/claim_extractor_independent_provisional_eval.csv)
- **Provenance**: Provisional AI Consensus (`dual_guidelines_deberta_consensus`), unverified by human annotators.
- **Corpus Composition**: 143 sentences sampled from 9 unseen peer-reviewed papers outside NLP and Clinical Medicine:
  1. `INDEP_01_MATSCI` (Materials Science / Corrosion Engineering) — 15 sentences
  2. `INDEP_02_ASTRO` (Exoplanet Astrophysics) — 16 sentences
  3. `INDEP_03_ENERGY` (Solid-State Battery Energy / 2D Materials) — 16 sentences
  4. `INDEP_04_GEOL` (Earthquake Geophysics / PINN Travel-Time Inference) — 16 sentences
  5. `INDEP_05_CLIM` (Climate Modeling / Monsoon Precipitation Projections) — 16 sentences
  6. `INDEP_06_ECOL` (Theoretical Evolutionary Ecology / Cooperator Dynamics) — 16 sentences
  7. `INDEP_07_QUANTUM` (Quantum Many-Body Physics / Entanglement Crossover) — 16 sentences
  8. `INDEP_08_PLANTBIO` (Biophysics / Chlorophyll Excitonic Transitions) — 16 sentences
  9. `INDEP_09_ENVGEO` (Subglacial Hydrology / Aquifer Slip Approximation) — 16 sentences
- **Section Distribution**: Results (36), Introduction (27), Methods (27), Discussion (27), Abstract (26).
- **Class Balance**: 109 Provisional Claims (76.2%), 34 Provisional Non-claims (23.8%).
- **Verification Status**: 0 / 143 (0.0%) reviewed by a human expert.

### 3. Full System Database Corpora
- **Global Evaluation Corpus Database** ([`experiments/runs/experiment_FINAL_2026-09-02.db`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/experiments/runs/experiment_FINAL_2026-09-02.db)):
  - **11 papers** completely ingested and indexed.
  - **816 extracted claims**, **671 claim pairs**, **12,286 evidence graph edges**.
- **Isolated Multi-Paper Session Database** ([`experiments/runs/sessions/session_20261026_182922_b04beb.db`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/experiments/runs/sessions/session_20261026_182922_b04beb.db)):
  - **2 papers** ingested in an isolated multi-paper session.
  - **220 claims**, **11 claim pairs**, **3,538 evidence graph edges**.

---

## 6. All Existing Experimental Results & Source Report Index

### Summary Results Table

| Benchmark | Model Configuration | Samples | Accuracy (95% CI) | Claim Precision | Claim Recall | Claim F1 | Macro F1 | Weighted F1 | Confusion Matrix | Source File Path |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Development** | Majority Class Baseline | 80 | 57.50% [46.6%, 67.7%] | 0.00% | 0.00% | 0.00% | 36.51% | 41.98% | $\text{TP}=0, \text{FP}=0, \text{FN}=34, \text{TN}=46$ | [`reports/final_evaluation/final_metrics.json`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/final_evaluation/final_metrics.json) |
| **Development** | Rule-Based Keyword Baseline | 80 | 66.25% [55.4%, 75.6%] | 88.89% | 23.53% | 37.21% | 57.07% | 60.04% | $\text{TP}=8, \text{FP}=1, \text{FN}=26, \text{TN}=45$ | [`reports/final_evaluation/final_metrics.json`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/final_evaluation/final_metrics.json) |
| **Development** | Baseline Extractor (Pre-Improvement) | 80 | 83.75% [74.2%, 90.2%] | 73.33% | 97.06% | 83.54% | 83.75% | 83.78% | $\text{TP}=33, \text{FP}=12, \text{FN}=1, \text{TN}=34$ | [`reports/final_evaluation/baseline_vs_improved.csv`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/final_evaluation/baseline_vs_improved.csv) |
| **Development** | **Current Improved AERF Extractor** | **80** | **91.25% [83.0%, 95.7%]** | **82.93%** | **100.00%** | **90.67%** | **91.22%** | **91.30%** | $\mathbf{\text{TP}=34, \text{FP}=7, \text{FN}=0, \text{TN}=39}$ | [`reports/final_evaluation/final_report.md`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/final_evaluation/final_report.md) |
| **Independent** | Baseline Extractor (Pre-Improvement) | 143 | 46.85% [38.9%, 55.0%] | 85.11% | 36.70% | 51.28% | 46.41% | 48.97% | $\text{TP}=40, \text{FP}=7, \text{FN}=69, \text{TN}=27$ | [`reports/final_evaluation/baseline_vs_improved.csv`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/final_evaluation/baseline_vs_improved.csv) |
| **Independent** | **Current Improved AERF Extractor** | **143** | **81.82% [74.7%, 87.3%]** | **90.29%** | **85.32%** | **87.74%** | **76.30%** | **82.30%** | $\mathbf{\text{TP}=93, \text{FP}=10, \text{FN}=16, \text{TN}=24}$ | [`reports/final_evaluation/final_metrics.json`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/final_evaluation/final_metrics.json) |

### Source Report Directory Index

1. **Executive Evaluation Report**: [`reports/final_evaluation/final_report.md`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/final_evaluation/final_report.md)
2. **Machine-Readable Metrics**: [`reports/final_evaluation/final_metrics.json`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/final_evaluation/final_metrics.json)
3. **Confusion Matrix Records**: [`reports/final_evaluation/final_confusion_matrix.csv`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/final_evaluation/final_confusion_matrix.csv)
4. **Baseline vs. Improved Records**: [`reports/final_evaluation/baseline_vs_improved.csv`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/final_evaluation/baseline_vs_improved.csv)
5. **False Negative Audit**: [`reports/independent_eval/false_negative_audit.csv`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/independent_eval/false_negative_audit.csv)
6. **Prioritized Review Decisions**: [`reports/independent_eval/priority_review_decisions.csv`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/independent_eval/priority_review_decisions.csv)
7. **Scientific Novelty & Contributions Report**: [`reports/final_evaluation/novelty_and_contributions.md`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/final_evaluation/novelty_and_contributions.md)
8. **Reproducibility Protocol**: [`reports/final_evaluation/reproducibility.md`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/final_evaluation/reproducibility.md)
9. **Frozen Configuration Registry**: [`reports/frozen_extractor_config.json`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/reports/frozen_extractor_config.json)

---

## 7. Baselines and Comparisons Already Performed

The project benchmark evaluated four distinct extractor configurations:

1. **Majority Class Baseline (`predict_constant_0`)**:
   - Assumes every sentence is a non-finding (Class 0).
   - Acc: 57.50%, Macro F1: 36.51%, Claim F1: 0.00%. Demonstrates the baseline floor and class skew.
2. **Transparent Rule-Based Keyword Baseline (`keyword_and_suppression_regex`)**:
   - Uses empirical keyword matching (`outperform`, `showed that`, `accuracy`, `bleu`) with method-lead suppression.
   - Acc: 66.25%, Prec: 88.89%, Rec: 23.53%, Claim F1: 37.21%. Demonstrates high precision but extreme false negative rate (misses 76.5% of claims lacking explicit keywords).
3. **Pre-Improvement Baseline Zero-Shot Extractor (`nlp/claim_extractor.py.bak`)**:
   - Original implementation using threshold 0.55 and uncalibrated discourse rules.
   - Dev ($N=80$): Acc 83.75%, Prec 73.33%, Rec 97.06%, F1 83.54% ($\text{FP}=12, \text{FN}=1$).
   - Independent ($N=143$): Acc 46.85%, Prec 85.11%, Rec 36.70%, F1 51.28% ($\text{FP}=7, \text{FN}=69$). Suffered massive out-of-domain recall collapse on physical sciences.
4. **Current Improved AERF Extractor (`nlp/claim_extractor.py`)**:
   - Calibrated multi-class threshold (0.35), asymmetric epistemic discourse gating, expanded physical gradient comparatives, and third-person system-subject contribution patterns.
   - Dev ($N=80$): Acc 91.25%, Prec 82.93%, Rec 100.00%, F1 90.67% ($\text{FP}=7, \text{FN}=0$).
   - Independent ($N=143$): Acc 81.82%, Prec 90.29%, Rec 85.32%, F1 87.74% ($\text{FP}=10, \text{FN}=16$).

---

## 8. Limitations, Incomplete Features, and Evaluation Weaknesses

To uphold academic defensibility, the following limitations must be stated explicitly in any research paper submission:

1. **Unverified Human Ground Truth on Independent Set**:
   - The 143-sentence independent benchmark ground-truth labels currently originate from dual-AI model consensus (`dual_guidelines_deberta_consensus`).
   - Human review status is currently **0 out of 143 (0.0%) reviewed**. Independent benchmark metrics reflect inter-model agreement rates, not certified human gold truth.
2. **Small Size of Development Benchmark ($N=80$)**:
   - The verified development benchmark contains only 80 sentences across 6 papers. While drawn from foundational NLP and Clinical Medicine literature, a larger multi-annotator sample is required for statistical power.
3. **Rule-Based Relation Extraction**:
   - Relation extraction in [`nlp/relation_extractor.py`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/nlp/relation_extractor.py) uses dependency parse patterns rather than a modern fine-tuned transformer. Complex relations with long-distance dependencies are prone to extraction errors.
4. **PDF Ingestion Parsing Artifacts**:
   - Extracting text from multi-column PDFs containing complex mathematical formatting (e.g., Quantum Physics and Environmental Geology papers) produces text-splicing fragments and LaTeX artifact strings (e.g., `(cid:88)`, `⋄ ⋄`).
5. **Fixed Heuristic Reliability Weights**:
   - The weights assigned to the 7 reliability factors ($0.10, 0.15, 0.20$, etc.) in `aerf/scorer.py` are deterministic expert-configured values. They were not learned via regression against empirical replication studies (e.g., DARPA SCORE or Cochrane meta-analyses).
6. **Cross-Document Coreference Absence**:
   - Claims are represented as standalone sentences. If an empirical finding relies on an anaphoric pronoun (*"This effect was statistically significant..."*), the extractor does not resolve the antecedent from the preceding sentence.

---

## 9. Potential Research Contributions Supported by Implementation Evidence

The following technical contributions are supported by verifiable implementation code and reproducible empirical data:

1. **Asymmetric Epistemic Discourse Gating**:
   A lightweight hybrid extraction paradigm coupling small zero-shot language models with deterministic epistemic routers that filter out non-claims (untested hypotheses, external citations, parameter tables, and study limitations), reducing development false positives by 41.7% and eliminating false negatives.
2. **Formal Continuous Multi-Factor Reliability Formulation**:
   A deterministic closed-form formulation ($R(c) \in [0, 1]$) combining physical provenance completeness, experimental study design rigor, temporal recency decay, and saturating non-linear evidence-mass aggregation from cross-document supporting and contradicting claims.
3. **Adaptive Graph Propagation of Scientific Evidence**:
   A graph reasoning engine that dynamically updates affected claim neighborhoods upon the arrival of new scientific publications, modeling the shifting consensus of scientific knowledge over time.
4. **Contradiction-Margin Research Gap Identification**:
   An automated gap-detection algorithm that separates isolated assertions lacking corroboration ($F_{\text{corrob}} < 0.35$) from active scientific controversies characterized by near-equal competing evidence masses ($|M_{\text{supp}} - M_{\text{contra}}| \le 0.15$).
5. **Session-Scoped Multi-Paper Evidence Isolation**:
   An architectural pattern enabling researchers to execute ad-hoc comparative evidence reviews on newly uploaded PDFs within temporary SQLite databases without corrupting or contaminating the persistent global scientific database.

---

## 10. Dependencies and Step-by-Step Reproduction Guide

### Environment & Dependencies
- **Runtime**: Python 3.12 (Windows 64-bit)
- **Key Libraries**:
  - `spacy>=3.7.0` (with `en_core_sci_sm` model installed)
  - `transformers>=4.38.0`, `torch>=2.2.0`
  - `sentence-transformers>=2.5.0`
  - `scikit-learn>=1.4.0`, `numpy>=1.26.0`, `pandas>=2.2.0`
  - `pypdf>=4.0.0`, `pdfplumber>=0.10.0`
  - `streamlit>=1.32.0`, `networkx>=3.2.0`, `sqlite3`

### Step-by-Step Reproduction Commands

Execute the following commands from `C:\Users\tejas\Downloads\AERF-engine-FIXED\proj`:

```powershell
# 1. Reproduce all historical and candidate baselines deterministically
& 'C:\Program Files\Python312\python.exe' scripts/reproduce_all_baselines.py
# Expected Output: Exit Code 0; prints Dev (N=80) and Indep (N=143) tables.

# 2. Run the full final comparative experimental suite
& 'C:\Program Files\Python312\python.exe' scripts/run_final_experimental_comparison.py
# Expected Output: Exit Code 0; regenerates all 7 reports in reports/final_evaluation/.

# 3. Execute the independent protocol test suite
& 'C:\Program Files\Python312\python.exe' -m pytest tests/unit/test_independent_evaluation_protocol.py -v
# Expected Output: 17 passed in <1s (validates hash integrity, non-overlap, Cohen's kappa).

# 4. Execute the complete project unit and integration test suite
& 'C:\Program Files\Python312\python.exe' -m pytest tests/unit tests/integration -q
# Expected Output: 188 passed in ~40s.

# 5. Verify mathematical consistency across JSON, CSV, and Markdown artifacts
& 'C:\Program Files\Python312\python.exe' scripts/verify_artifact_consistency.py
# Expected Output: Exit Code 0; prints 100% agreement across all metric files.
```

---

## 11. Certified Configuration Hashes & Integrity Inventory

| Component / Artifact | Path | SHA-256 Hash | Status |
|:---|:---|:---:|:---:|
| **Active Frozen Extractor** | [`nlp/claim_extractor.py`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/nlp/claim_extractor.py) | `1b643899b6e62893510b7eca7d95a3ff4e69a4a1c714017c3e9d8738f390ffa3` | **FROZEN** |
| **Pre-Improvement Baseline** | [`nlp/claim_extractor.py.bak`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/nlp/claim_extractor.py.bak) | `f1dddfaa35e8c76f250fb0bfa069d758f1a96b882ed80580d1d59bed55f968f7` | **BACKUP** |
| **Initial Candidate Backup** | [`nlp/claim_extractor.py.best_current`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/nlp/claim_extractor.py.best_current) | `f28b10f9ee759cdf55d998b0002fb511834048a40c82577a08d83ef2e0e88e6b` | **BACKUP** |
| **Calibrated Candidate Backup** | [`nlp/claim_extractor.py.calibrated_threshold35`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/nlp/claim_extractor.py.calibrated_threshold35) | `88fa836d40cbb9bf3654e0c8c8770a081536af1a6280ee7061f0aab5efa1e8db` | **BACKUP** |
| **Development Benchmark** | [`tests/data/claim_extractor_eval.csv`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/tests/data/claim_extractor_eval.csv) | `6d57c285fa650f4c47384ab84708a4caab753720ba1b936a48ee4aa974c44354` | **FROZEN** |
| **Independent Benchmark** | [`tests/data/independent_eval/claim_extractor_independent_provisional_eval.csv`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/tests/data/independent_eval/claim_extractor_independent_provisional_eval.csv) | `ef9cad55f157c5e232e239ceea7f8d2c8387e7c05d7fbf34f53e9f7ada7d2f58` | **FROZEN** |
| **Global 11-Paper Database** | [`experiments/runs/experiment_FINAL_2026-09-02.db`](file:///C:/Users/tejas/Downloads/AERF-engine-FIXED/proj/experiments/runs/experiment_FINAL_2026-09-02.db) | *(11 papers, 816 claims, 671 pairs, 12,286 edges)* | **INTACT** |

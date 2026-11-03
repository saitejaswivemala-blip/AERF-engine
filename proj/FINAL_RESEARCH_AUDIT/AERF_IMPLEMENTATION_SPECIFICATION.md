# AERF Engine: Complete Implementation & Algorithmic Specification

**Document Purpose**: Formally verified technical reference for the Methodology section of the AERF research manuscript.  
**Auditor**: Antigravity Technical & Research Methodology Audit  
**Date**: November 01, 2026  
**Implementation Root**: `proj/`  

---

## 1. System Pipeline Overview

The Adaptive Evidence Reliability and Reasoning Framework (AERF) executes an end-to-end, multi-stage pipeline converting unstructured scientific PDF literature into a queryable, provenance-anchored topological evidence graph:

```
[Scientific Research PDFs]
           │
           ▼ Phase 1: Ingestion & Provenance
┌────────────────────────────────────────────────────────┐
│ - pdfplumber character extraction                      │
│ - Abbreviation-aware sentence splitting & normalization│
│ - Physical byte & character coordinate anchoring       │
│ - SQLite relational persistence                        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼ Phase 2: Scientific NLP Enrichment & Claim Extraction
┌────────────────────────────────────────────────────────┐
│ - SciSpaCy NER (en_core_sci_sm v0.5.4)                 │
│ - Hybrid Zero-Shot Classifier (valhalla/distilbart)    │
│   with Calibrated Decision Threshold (tau = 0.35)      │
│ - Asymmetric Epistemic Discourse Router (6 Guards)     │
│ - Multi-Domain Scientific Result Cue Rescuer           │
│ - Sentence Embeddings (all-MiniLM-L6-v2, 384-d)       │
│ - Dependency-Parse Relation Extraction (Base Lemmas)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼ Phase 3: Alignment, Reliability & Evidence Graph
┌────────────────────────────────────────────────────────┐
│ - Cosine Semantic Candidate Gating (>= 0.55)           │
│ - Entity Jaccard Overlap Alignment Filter (>= 0.15)    │
│ - Cross-Encoder NLI (DeBERTa-v3-base-mnli-fever-anli)  │
│ - Deterministic 7-Factor Reliability Scoring (R in 0-1)│
│ - Adaptive Neighborhood Score Propagation (eps = 0.001)│
│ - Topological Evidence Graph (Claims, Entities, Edges) │
│ - Contradiction-Margin Research Gap Identification     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼ Session & Presentation Layer
┌────────────────────────────────────────────────────────┐
│ - Session Isolation Engine (Isolated SQLite Databases) │
│ - Interactive Streamlit Dashboard                      │
└────────────────────────────────────────────────────────┘
```

---

## 2. Phase 1: Ingestion, Preprocessing & Provenance

### 1. PDF Loading & Text Extraction (`ingestion/pdf_loader.py`)
- Extraction engine: `pdfplumber` (v0.11.9).
- Fallback engine: `pypdf` (v4.0.0).
- Functionality: Reads page-by-page text streams from PDF files, tracking page numbers and raw text offsets.
- *Limitation*: Text layer extraction only; image-scanned PDFs without OCR layers produce empty strings.

### 2. Normalization & Sentence Segmentation (`ingestion/text_cleaner.py`, `nlp/claim_extractor.py`)
- Glued-heading detachment: Separates section headers glued to prose (e.g., `RESULTSAnd` $\to$ `RESULTS And`).
- Abbreviation boundary protection: Handles common academic abbreviations (`et al.`, `Fig.`, `Figs.`, `Eq.`, `Tab.`, `vs.`, `i.e.`, `e.g.`, `approx.`, `Dr.`, `Prof.`) to prevent premature sentence termination.
- Quality filtering: Rejects fragments $< 25$ characters, strings lacking alphabet characters, and lines with alphabetic character ratio $< 0.55$.
- Provenance coordinates: Anchors `start_char` and `end_char` offsets into the cleaned chunk string stored in the SQLite database.

---

## 3. Phase 2: Scientific Enrichment & Claim Extraction

### 1. Named Entity Recognition (`nlp/ner.py`)
- Backbone: SciSpaCy `en_core_sci_sm` (v0.5.4).
- Extracted Classes: Biomedical and physical scientific entities linked directly to sentence character offsets.

### 2. Hybrid Zero-Shot Claim Extractor (`nlp/claim_extractor.py`)
- Model: `valhalla/distilbart-mnli-12-3` (Hugging Face Transformers).
- Hypothesis Template: `"This sentence {}."`
- Candidate Labels: 
  - `"reports a scientific finding or conclusion"` (Class 1)
  - `"describes background information or prior work"` (Class 0)
  - `"describes experimental methodology or setup"` (Class 0)
  - `"discusses study limitations or future work"` (Class 0)
- **Calibrated Decision Threshold**: $\tau = 0.35$ (calibrated from the uncalibrated $0.55$ baseline to account for probability dilution across 4 multi-class labels).

### 3. Asymmetric Epistemic Discourse Router (`nlp/claim_extractor.py`)
Intercepts zero-shot false positives using 6 deterministic precision suppression guards:
1. `_HYPOTHESIS_OBJECTIVE`: Rejects prospective conjectures (*"We hypothesized that..."*, *"We aimed to determine..."*).
2. `_EXTERNAL_LITERATURE`: Rejects citations of external literature (*"A 2021 meta-analysis by Smith et al. revealed..."*, *"Recent studies have reported..."*).
3. `_DECLARATIONS_AND_REFS`: Rejects ethics, competing interests, data availability, and citation strings (*"The authors declare no competing interests"*, *"Data are available at..."*).
4. `_CONFIG_SPEC`: Rejects computational and hyperparameter configurations (*"We set batch size to 32 and learning rate to 0.001..."*).
5. `_PROCEDURAL_METHOD`: Rejects physical actions of data collection and participant enrollment (*"Participants were enrolled between 2018 and 2020..."*).
6. `_LIMITATION_FUTURE`: Rejects caveats and future directions (*"A limitation of our secondary analysis is that..."*).

### 4. Multi-Domain Scientific Result Cue Rescuer (`nlp/claim_extractor.py`)
Rescues true claims where zero-shot confidence falls below threshold ($0.10 \le P \le 0.35$):
- Quantitative statistical indicators: $p$-values ($p < 0.05$), odds ratios, hazard ratios, 95% confidence intervals.
- Non-biomedical physical units: electron-volts ($\text{eV}$), milliampere-hours ($\text{mAh}$), megapascals ($\text{MPa}$), gigapascals ($\text{GPa}$), BLEU scores, decibels ($\text{dB}$).
- Directional/Comparative verbs: `outperform`, `surpass`, `exhibit higher`, `produced a significant decrease`.
- Explicit null/refutational outcomes: `failed to show`, `did not significantly affect`, `dispute earlier claims`.

### 5. Sentence Embeddings (`nlp/embeddings.py`)
- Backbone: `sentence-transformers/all-MiniLM-L6-v2`.
- Vector Dimension: 384 dense float32 dimensions.
- Distance Metric: Dot-product cosine similarity.

### 6. Dependency-Parse Relation Extraction (`nlp/relation_extractor.py`)
- Pattern-matching over dependency parses linking subject and object entities across approved base predicate lemmas:
  `treat`, `cause`, `prevent`, `improve`, `increase`, `reduce`, `inhibit`, `enhance`.

---

## 4. Phase 3: Alignment, NLI, Reliability Scoring & Evidence Graph

### 1. Two-Stage Cross-Paper Alignment Funnel (`matching/claim_matcher.py`)
- **Stage 1 (Semantic Cosine Filter)**: Evaluates vector cosine similarity between claim embeddings. Pairs with $\text{sim} < 0.55$ are discarded.
- **Stage 2 (Entity Overlap Filter)**: Calculates token Jaccard overlap between extracted entity sets. Pairs with Jaccard $< 0.15$ are filtered out.

### 2. Natural Language Inference Cross-Encoder (`nlp/nli/nli_classifier.py`)
- Model: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`.
- Stance Labels: `supports`, `contradicts`, `neutral`.
- Stance Confidence Cutoff: $P(\text{label}) \ge 0.60$.

### 3. Deterministic 7-Factor AERF Reliability Scorer (`aerf/scorer.py`)
The reliability score $R(c) \in [0.0, 1.0]$ for a claim $c$ is deterministically calculated as:

$$R(c) = \operatorname{clip}\left(\sum_{k \in \mathcal{K}_{\text{pos}}} w_k F_k(c) - w_{\text{contra}} F_{\text{contra}}(c), \, 0.0, \, 1.0\right)$$

Where $\mathcal{K}_{\text{pos}} = \{ \text{source}, \text{methodology}, \text{corroboration}, \text{temporal}, \text{semantic}, \text{provenance} \}$, and the configured weights are:

| Factor | Weight ($w_k$) | Definition & Mathematical Implementation |
|:---|:---:|:---|
| **Source Factor ($F_{\text{source}}$)** | **0.10** | Metadata completeness: $\frac{1}{4}[\mathbb{I}(\text{title}) + \mathbb{I}(\text{authors}) + \mathbb{I}(\text{year}) + \mathbb{I}(\text{venue})]$. |
| **Methodology Factor ($F_{\text{methodology}}$)** | **0.15** | Study rigor index: $\operatorname{clip}(0.4D + 0.3N + 0.3T, 0.0, 1.0)$, where $D=1$ for randomized/controlled designs, $N=1$ for explicit sample counts, and $T=1$ for formal statistical reporting ($p$-values, CIs, effect sizes). |
| **Corroboration Factor ($F_{\text{corrob}}$)** | **0.20** | Non-linear saturating evidence mass from validated supporting claims: $\frac{M_{\text{supp}}}{2.0 + M_{\text{supp}}}$, where $M_{\text{supp}} = \sum_{j \in \mathcal{N}_{\text{supp}}} R(c_j) \cdot P_{\text{NLI}}(j)$. |
| **Contradiction Factor ($F_{\text{contra}}$)** | **0.20** | Non-linear saturating contradiction penalty: enters negatively ($-0.20$), where $F_{\text{contra}} = \frac{M_{\text{contra}}}{2.0 + M_{\text{contra}}}$, with $M_{\text{contra}} = \sum_{j \in \mathcal{N}_{\text{contra}}} R(c_j) \cdot P_{\text{NLI}}(j)$. |
| **Temporal Context Factor ($F_{\text{temporal}}$)** | **0.10** | Recency decay over a 10-year sliding window ($W = 10.0$): $\max\left(0.0, \, 1.0 - \frac{Y_{\max} - Y_c}{10.0}\right)$, defaulting to $0.5$ if year is missing. |
| **Semantic Alignment Factor ($F_{\text{semantic}}$)** | **0.15** | Highest validated NLI confidence score among connected evidence pairs: $\max_{p \in \mathcal{P}(c)} P_{\text{NLI}}(p)$, defaulting to $0.5$ if unlinked. |
| **Provenance Completeness ($F_{\text{provenance}}$)** | **0.10** | Verification ratio checking that valid `chunk_id`, `page_number`, `start_char`, and `end_char` offsets are registered in SQLite. |

### 4. Adaptive Neighborhood Updater (`aerf/adaptive_updater.py`)
- Upon ingestion of a new paper or updated claim pair, identifies all claims in the 1-hop topological neighborhood: $\mathcal{N}_1(c)$.
- Iteratively recomputes $R(c)$ using the dynamic evidence mass formulas.
- If $|R_{\text{new}}(c) - R_{\text{old}}(c)| > \epsilon$ (where $\epsilon = 0.001$), commits the new score and records an entry in `aerf_history` with deltas and human-readable explanations.

### 5. Automated Research Gap Detector (`gaps/gap_detector.py`)
Flags two distinct categories of candidate scientific research gaps:
1. **Uncorroborated Claims**: Claims with low corroborating evidence mass: $F_{\text{corrob}}(c) < 0.35$.
2. **Contested Scientific Findings**: Claims characterized by unresolved contradictory evidence where support and contradiction masses are closely balanced: $|M_{\text{supp}}(c) - M_{\text{contra}}(c)| \le 0.15$.

### 6. Session Isolation Architecture (`pipeline/session_manager.py`)
- Directs user-uploaded multi-paper reviews into session-scoped SQLite database files (`experiments/runs/sessions/session_YYYYMMDD_HHMMSS_xxxxxx.db`).
- Guarantees that ad-hoc interactive reviews do not contaminate or modify the persistent benchmark database (`experiments/runs/experiment_FINAL_2026-09-02.db`).

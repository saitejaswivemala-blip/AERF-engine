# AERF Engine — Independent Evaluation Corpus Collection Report

**Audit Subject**: AERF Scientific Claim Extractor (`ScientificClaimExtractor` in `nlp/claim_extractor.py`)  
**Report Date**: October 21, 2026  
**Status**: `PAPERS_COLLECTED_PENDING_INDEPENDENT_ANNOTATION`  
**Location of PDFs**: `data/independent_test_corpus/`  
**Manifest**: `tests/data/independent_eval/independent_paper_manifest.json`  
**Master Annotation Sheet**: `tests/data/independent_eval/claim_extractor_independent_eval.csv`  

---

## 1. Executive Summary & Verification Declaration

In accordance with strict academic integrity standards:
1. **Real Empirical Papers Only**: **9 genuine research papers** across 9 distinct scientific disciplines have been searched, retrieved, verified, and downloaded directly into `data/independent_test_corpus/`. Zero fake, synthetic, or invented papers were used.
2. **True Out-of-Domain Breadth**: All 9 papers represent scientific fields outside Computer Science (Transformer NLP) and Clinical Medicine (Vitamin D trials).
3. **Verified Development Independence**: All 9 papers and their authors were verified to have **zero overlap** with the 7 development papers (*Vaswani 2017, Devlin 2018, Martineau 2017, Bolland 2022, Jolliffe 2024, CORONAVIT 2022, Zhang 2022*), zero appearances in project manifests, and zero presence in diagnostic test fixtures.
4. **Blind Sentence Sampling**: **143 genuine scientific sentences** were sampled directly from the primary PDF text across Abstract, Introduction, Methods, Results, and Discussion sections. Sentence extraction was performed **blind to the claim extractor model** (the model was not executed or consulted during selection).
5. **Labels Unassigned**: All 143 sentences have their labels left unassigned (`label=""`, `annotator_1_label=""`, `annotator_2_label=""`) awaiting independent dual-annotator labeling. Zero evaluation metrics are reported.

---

## 2. Inventory of Collected & Verified Independent Papers

| Paper ID | Scientific Discipline | Publication Title | Authors | Year | Venue / Archive | DOI / URL | PDF Size | SHA-256 Checksum | Sentences |
|---|---|---|---|---|---|---|---|---|---|
| **`INDEP_01_MATSCI`** | Materials Science & Metallurgy | *Effect of severe plastic deformation realized by rotary swaging on the mechanical properties and corrosion resistance of near-alpha-titanium alloy Ti-2.5Al-2.6Zr* | V.N. Chuvil'deev, V.I. Kopylov, A.V. Nokhrin, et al. | 2024 | `arXiv:cond-mat.mtrl-sci` | `arXiv:2401.17672` | 5,099,840 B | `bd22c00c63ca7bb1f2c4f0886663c2a5c0752fa7041394902761cd48e2e62f47` | 15 |
| **`INDEP_02_ASTRO`** | Astrophysics & Planetary Science | *Muted Features in the JWST NIRISS Transmission Spectrum of Hot-Neptune LTT 9779 b* | Michael Radica, David Lafreniere, et al. | 2024 | `arXiv:astro-ph.EP` | `arXiv:2401.15548` | 2,190,232 B | `749df59038e0035036c1437624a03d563c71e986dc57e747052dcf7a0e5486f8` | 16 |
| **`INDEP_03_ENERGY`** | Renewable Energy & Battery Systems | *Enhancement of Li and Na Storage in Janus HfSSe Monolayers by Co-Induced Electron-Deficient Chalcogen Sites: A First-Principles Study* | Md. Muktadir Hasan, Md. Mafizul Islam, Ahmed Zubair | 2026 | `arXiv:physics.app-ph` | `arXiv:2610.28152` | 3,069,244 B | `ec7c9054bc842584b8e61ecbf220dae7fe36e196017e18888fea5fd42303f4a6` | 16 |
| **`INDEP_04_GEOL`** | Structural Geology & Seismology | *HypoNet Nankai PS: rapid hypocenter determination for the Nankai Trough using physics-informed neural networks with three-dimensional P- and S-wave velocity models* | Ryoichiro Agata, Ayako Nakanishi, Dan Bassett, et al. | 2026 | `arXiv:physics.geo-ph` | `arXiv:2610.27561` | 6,834,957 B | `41705438b1f70f15c15e61c3db77c6e61f2249e0b1716ea055376c66d21f83fa` | 16 |
| **`INDEP_05_CLIM`** | Atmospheric Physics & Climatology | *Disentangling forced and internal components in observed climate evolution using spatiotemporal deep learning* | Parvathi Kooloth, Wenyu Zhou, L. Ruby Leung, et al. | 2026 | `arXiv:physics.ao-ph` | `arXiv:2610.28219` | 15,956,096 B | `e1c62e6640a16d3a87ec2d8ea2e0d3cb35d25e0c5f21272fc4a107efb6d19420` | 16 |
| **`INDEP_06_ECOL`** | Ecology & Marine Biodiversity | *The resource cost of cooperative invasion* | Abhishek Chowdhury | 2026 | `arXiv:q-bio.PE` | `arXiv:2610.26235` | 1,274,785 B | `38de50e5f68480cb13c3aa4e601cff3fc8956e5f3aa900ef270e59a60e0a969f` | 16 |
| **`INDEP_07_QUANTUM`** | Quantum Optics & Condensed Matter | *Tuning between collective dynamics and scrambling through initial-state symmetry alignment in long-range spin models* | Sridevi Kuriyattil, Johannes Kombe, Andrew J. Daley, et al. | 2026 | `arXiv:quant-ph` | `arXiv:2610.28462` | 4,291,282 B | `5974f1b170080231cfb263b65597950fefab314e1f7dcf7aa92a544c4fae95bc` | 16 |
| **`INDEP_08_PLANTBIO`** | Plant Biology & Photosynthesis | *The Mathematics of Chlorophyll Photoexcitation* | Archit Chaturvedi | 2024 | `arXiv:q-bio.BM` | `arXiv:2405.06644` | 70,196 B | `8a4217a86606a88e7d704ea11100f91a5611c0fcefa5950ee05cf00958fa1ef1` | 16 |
| **`INDEP_09_ENVGEO`** | Environmental Geochemistry & Hydrology | *Groundwater dynamics beneath a marine ice sheet* | Gabriel J. Cairns, Graham P. Benham, Ian J. Hewitt | 2024 | `arXiv:physics.geo-ph` | `arXiv:2409.11848` | 3,665,623 B | `ba5ea793d90948556dbb4241ccda9b66236b281f6ae2b7efc687e87a2a0ff3bb` | 16 |

---

## 3. Structural Section Distribution of Sampled Sentences

The 143 sampled sentences are distributed across paper sections:
- **Abstract**: 27 sentences (18.9%)
- **Introduction**: 27 sentences (18.9%)
- **Methods**: 27 sentences (18.9%)
- **Results**: 35 sentences (24.5%)
- **Discussion / Conclusion**: 27 sentences (18.9%)
- **Total**: **143 sentences** across 9 independent papers.

---

## 4. Verification Gates Audit

All 9 papers passed every verification gate in the exhaustive verification audit (`scripts/verify_independent_corpus_integrity.py`):
1. **Authoritative API Resolution**: Verified. Every paper's arXiv ID was queried against the official arXiv Atom API (`https://export.arxiv.org/api/query`), successfully resolving full titles, authors, and primary categories.
2. **Open-Access Full-Text PDF & Readability**: Verified. Valid, readable binary `%PDF` documents downloaded in `data/independent_test_corpus/`, with front-matter title and author confirmation in PDF text.
3. **SHA-256 Checksum Integrity**: Verified. Computed SHA-256 hashes match `tests/data/independent_eval/independent_paper_manifest.json` exactly.
4. **Sentence Occurrence in Source PDF**: Verified. All 143 sampled sentences were traced and confirmed to appear verbatim within their respective source PDFs. Text extraction was hardened (`x_tolerance=1.5`) and filtered to eliminate page headers, footers, math expressions, and glued-word artifacts.
5. **Zero Duplicate Sentences**: Verified. 0 duplicates within any individual paper, and 0 duplicates across the entire 143-sentence corpus.
6. **Development Independence Check**: Verified. Zero overlap with development papers (`Vaswani`, `Devlin`, `Martineau`, `Bolland`, `Jolliffe`, `CORONAVIT`, `Zhang`), zero overlap with `experiments/datasets/paper_manifest.json`, and zero overlap with the 80 development sentences in `claim_extractor_eval.csv`.
7. **Frozen Configuration Check**: Verified. Codebase hashes match `reports/frozen_extractor_config.json`. Extractor has not been executed or tuned.

---

## 5. Next Steps for Independent Evaluation

1. **Independent Dual Annotation**:
   - Annotator 1 annotates `tests/data/independent_eval/annotator_1_sheet.csv`.
   - Annotator 2 annotates `tests/data/independent_eval/annotator_2_sheet.csv`.
   - Both follow `docs/CLAIM_ANNOTATION_GUIDELINES.md` assigning `1` (scientific claim/finding) or `0` (non-claim).
2. **Disagreement Adjudication & Kappa Calculation**:
   ```powershell
   python scripts/adjudicate_independent_annotations.py `
       --annotator-1 tests/data/independent_eval/annotator_1_sheet.csv `
       --annotator-2 tests/data/independent_eval/annotator_2_sheet.csv
   ```
3. **Validation Gate**:
   ```powershell
   python scripts/validate_independent_dataset.py
   python -m pytest tests/unit/test_independent_evaluation_protocol.py -v
   ```
4. **Frozen Model Evaluation**:
   ```powershell
   python evaluate_claim_extractor.py `
       --dataset tests/data/independent_eval/claim_extractor_independent_final.csv `
       --report-dir reports/independent_eval
   ```

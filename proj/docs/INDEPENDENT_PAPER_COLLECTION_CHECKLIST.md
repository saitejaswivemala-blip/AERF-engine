# AERF Engine — Independent Paper Collection Checklist & Field Guide

**Document**: `docs/INDEPENDENT_PAPER_COLLECTION_CHECKLIST.md`  
**Purpose**: Practical, step-by-step field manual for collecting, verifying, sampling, and annotating 8–10 genuinely unseen open-access papers for independent evaluation.  
**Strict Integrity Rule**: Do NOT populate synthetic or invented examples. The independent evaluation dataset must remain empty until real, verified scientific literature is ingested and dual-annotated.

---

## 1. Domain Prioritization Matrix

To establish true multi-domain generalization, papers must come from scientific disciplines outside Computer Science/NLP and Vitamin D clinical trials:

| Slot | Scientific Discipline | Recommended Open-Access Venues | Search Queries / Repositories |
|---|---|---|---|
| **1** | **Materials Science & Metallurgy** | *Nature Communications*, *Acta Materialia*, *Advanced Functional Materials*, *arXiv:cond-mat.mtrl-sci* | "tensile strength alloy", "perovskite crystal", "superconductivity grain" |
| **2** | **Astrophysics & Cosmology** | *Astrophysical Journal*, *MNRAS*, *Astronomy & Astrophysics*, *arXiv:astro-ph* | "exoplanet transit spectroscopy", "pulsar timing array", "redshift galaxy cluster" |
| **3** | **Plant Biology & Agronomy** | *Frontiers in Plant Science*, *Plant Physiology*, *bioRxiv (Plant Biology)* | "photosystem II drought stress", "stomatal conductance nitrogen", "Arabidopsis transcription factor" |
| **4** | **Renewable Energy & Battery Systems** | *PRX Energy*, *Journal of Power Sources*, *Energy & Environmental Science* | "lithium dendrite solid electrolyte", "photocatalytic hydrogen evolution", "sodium-ion cathode" |
| **5** | **Environmental Chemistry & Oceanography** | *Environmental Science & Technology*, *Marine Chemistry*, *Geophysical Research Letters* | "microplastic benthic sediment", "ocean acidification coral calcification", "per- and polyfluoroalkyl substances" |
| **6** | **Structural Geology & Seismology** | *JGR: Solid Earth*, *Tectonophysics*, *Earth and Planetary Science Letters* | "fault rupture slip rate", "mantle plume shear wave", "subduction zone seismicity" |
| **7** | **Quantum Optics & Condensed Matter** | *Physical Review Research (APS OA)*, *Quantum*, *Optica* | "quantum state tomography", "squeezed light cavity", "Majorana zero mode" |
| **8** | **Ecology & Marine Biodiversity** | *Ecology Letters*, *Global Change Biology*, *Frontiers in Marine Science* | "trophic cascade keystone species", "coral bleaching sea surface temperature", "canopy biodiversity" |
| **9** | **Biochemistry & Enzymology (Non-Clinical)**| *Journal of Biological Chemistry*, *ACS Chemical Biology*, *FEBS Letters* | "enzyme catalytic turnover kcat", "allosteric binding kinetics", "cytochrome P450 reduction" |
| **10** | **Atmospheric Physics & Climatology** | *Atmospheric Chemistry and Physics (EGU)*, *Journal of Climate* | "stratospheric aerosol radiative forcing", "tropospheric ozone flux", "Hadley cell expansion" |

---

## 2. Paper Verification Gates (Before Extraction)

Every prospective paper must pass all four gates before a single sentence is sampled:

```
[Candidate PDF Downloaded]
         │
         ▼
[Gate 1: DOI & Venue Verification] ──► Valid DOI? Peer-reviewed or verified preprint?
         │
         ▼
[Gate 2: Development Independence] ──► Not in {Vaswani, Devlin, Martineau, Bolland, Jolliffe, Zhang}?
         │                             No shared authors with development corpus?
         ▼
[Gate 3: Open-Access License]      ──► Permissive license (CC-BY 4.0 or Open Access)?
         │                             Full text verbatim readable?
         ▼
[Gate 4: Staging & SHA-256 Hash]   ──► Placed into data/independent_test_corpus/
                                       SHA-256 recorded in manifest
```

### Gate Checklist:
- [ ] **DOI Verified**: DOI resolves to an authentic publication page on `https://doi.org/<DOI>`.
- [ ] **Venue Verified**: Published in a reputable peer-reviewed journal or indexed archive (arXiv/bioRxiv).
- [ ] **Author Independence**: First author, senior author, and co-authors have zero involvement with the AERF Engine development corpus (no Vaswani, Devlin, Martineau, Jolliffe, Bolland, etc.).
- [ ] **Topic Independence**: Substantive scientific content is NOT deep learning NLP benchmarks and NOT Vitamin D clinical trials.
- [ ] **PDF Staging**: PDF saved to `data/independent_test_corpus/<paper_id>.pdf`.
- [ ] **Checksum Recorded**: SHA-256 checksum generated and stored in `tests/data/independent_eval/independent_paper_manifest.json`.

---

## 3. Blind Sentence-Sampling Protocol (15–20 Sentences per Paper)

To avoid selection bias, sentences must be chosen **blind to the model's predictions**. Do not run the claim extractor when choosing sentences.

### Target Distribution per Paper:
For each paper, sample exactly **16 sentences** (or 15–20) distributed across the following standard structural sections:

| Target Section | Sentences | Sampling Strategy & Categories to Cover |
|---|---|---|
| **Abstract** | 3 sentences | - 1 primary positive empirical finding<br>- 1 problem statement / background<br>- 1 overall takeaway / conclusion |
| **Introduction / Background** | 3 sentences | - 1 widely accepted domain background fact (label 0)<br>- 1 citation to prior literature ("Smith et al. showed...") (label 0)<br>- 1 prospective research aim or hypothesis ("We hypothesized that...") (label 0) |
| **Methods / Experimental** | 3 sentences | - 1 sample preparation or apparatus description (label 0)<br>- 1 numeric parameter / recipe / operating conditions (label 0)<br>- 1 statistical software or analytical measurement protocol (label 0) |
| **Results** | 4 sentences | - 2 quantitative empirical findings with measurements / units / p-values (label 1)<br>- 1 comparative empirical finding ("Alloy A showed greater ductility than B") (label 1)<br>- 1 null, negative, or non-significant result ("No significant variation was detected") (label 1) |
| **Discussion / Limitations** | 3 sentences | - 1 explicit study limitation or caveat (label 0)<br>- 1 interpretive discussion takeaway (label 0 or 1 depending on empirical basis)<br>- 1 future work direction ("Future studies will examine...") (label 0) |

### Sentence Selection Rules:
1. **Verbatim Extraction**: Copy the exact character string from the PDF text.
2. **Exclude Non-Sentences**: Do not select section headers, author affiliations, mathematical equations displayed on standalone lines, table column headers, or figure captions.
3. **Record Metadata**: For every sentence, record:
   - `paper_id`
   - `source_section` (e.g., `Abstract`, `Introduction`, `Methods`, `Results`, `Discussion`)
   - `page_number` (printed page in the PDF)
   - `doi_or_url`

---

## 4. Dual-Annotator Review Workflow

To satisfy academic publication criteria, two annotators must annotate independently without conferring:

### Annotation Procedure:
1. **Annotator 1** receives `tests/data/independent_eval/annotator_1_sheet.csv` containing candidate sentences with blank label columns.
2. **Annotator 2** receives `tests/data/independent_eval/annotator_2_sheet.csv` containing the identical candidate sentences with blank label columns.
3. Both annotators read `docs/CLAIM_ANNOTATION_GUIDELINES.md` and assign:
   - `annotator_label`: `1` (scientific finding/claim) or `0` (non-claim).
   - `annotator_rationale`: Brief 5–10 word rationale (e.g., "quantitative empirical finding", "methodology parameter", "prospective hypothesis").
4. **Neither annotator may see the extractor's predictions or the other annotator's labels during this step.**

### Adjudication Procedure:
Once both sheets are filled:
1. Merge the annotations into `tests/data/independent_eval/claim_extractor_independent_eval.csv`.
2. Run the adjudication script:
   ```powershell
   python scripts/adjudicate_independent_annotations.py
   ```
3. The script calculates:
   - Observed Agreement ($p_o$)
   - Cohen's Kappa ($\kappa$)
4. If $\kappa \ge 0.80$, agreement is strong. All discrepant rows are automatically written to `reports/independent_adjudication_needed.csv`.
5. A senior referee reviews each disagreement and enters the `adjudicated_label` and `adjudication_notes`.
6. Rerun the script to produce the frozen evaluation dataset: `tests/data/independent_eval/claim_extractor_independent_final.csv`.

---

## 5. Summary of Automated Verification Commands

Before running the final evaluation, run the validation tool to ensure zero contamination:

```powershell
# Step 1: Verify the manifest and dataset integrity
python scripts/validate_independent_dataset.py

# Step 2: Run automated validation unit tests
python -m pytest tests/unit/test_independent_evaluation_protocol.py -v

# Step 3: Run the final independent evaluation on frozen extractor
python evaluate_claim_extractor.py `
    --dataset tests/data/independent_eval/claim_extractor_independent_final.csv `
    --report-dir reports/independent_eval
```

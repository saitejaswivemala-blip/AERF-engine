# AERF Engine — Phase 2

Phase 2 extends the approved Phase 1 ingestion foundation with scientific NER,
semantic claim extraction, relation extraction, claim embeddings, SQLite
persistence, and pipeline integration. NLI, contradiction detection, AERF,
adaptive AERF, evidence graph, knowledge evolution, research gaps and the
Streamlit dashboard are intentionally NOT implemented here.

## Models
- **NER:** `en_core_sci_sm` from SciSpaCy. Lightweight scientific NER suitable for local CPU use.
- **Claims:** `valhalla/distilbart-mnli-12-3` zero-shot classification. Four semantic labels distinguish findings from background, methodology and irrelevant text. No trigger-word rule decides claimhood.
- **Relations:** SciSpaCy dependency parse with six approved predicates: treats, causes, prevents, improves, increases, reduces. Conservative by design.
- **Embeddings:** `all-MiniLM-L6-v2` Sentence Transformer, normalized vectors stored as SQLite BLOBs.

## Provenance
Phase 1 chunks already carry `paper_id`, `page_number`, `char_start`, `char_end` and exact chunk text. Phase 2 entity and claim offsets are relative to the chunk/claim text and remain traceable through `paper_id -> chunk_id -> span`. Claim text is copied verbatim from the Phase-1 chunk; no generative rewriting is performed.

## Install (Windows PowerShell)
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Install the SciSpaCy model compatible with the installed SciSpaCy release. For example, use the official SciSpaCy model package for `en_core_sci_sm`.

## Run
First ingest a paper using the existing Phase 1 CLI:
```powershell
python main.py --init-db --file tests\fixtures\sample_paper.pdf
```
Then run Phase 2 on its `paper_id`:
```powershell
python main.py --phase2-paper-id 1
```

## Tests
```powershell
pytest -v
```
The included Phase 2 tests use injected fake NLP components for deterministic logic/provenance tests, so they do not pretend to validate model quality. Run a real-model smoke test after installing the SciSpaCy and Sentence-Transformer weights.

## Limitations
- No labelled claim dataset is included, so no claim precision/recall/F1 is reported.
- Zero-shot classification is not a domain-trained scientific claim detector; threshold requires validation on labelled examples.
- Relation extraction is intentionally conservative and can miss long-distance/implicit relations.
- SciSpaCy entity confidence is not persisted because the standard spaCy API does not expose a calibrated per-entity probability.
- PDF tables/figures and scanned PDFs remain limited by Phase 1 text extraction.

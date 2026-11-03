# AERF Engine — Phase 1: Ingestion Foundation

**Adaptive Scientific Knowledge Reasoning Engine using Explainable Evidence Reliability Framework (AERF)**

This is Phase 1 only:

```
PDF -> extraction -> cleaning -> provenance-preserving chunks -> SQLite storage
```

No NER, claim extraction, relation extraction, embeddings, NLI, AERF scoring, or dashboard is implemented yet — those are later phases, built on top of this foundation without changing it.

---

## 1. Project Structure

```
aerf-engine/
├── config/
│   ├── settings.yaml        # all paths/thresholds — nothing is hardcoded in code
│   └── config_loader.py     # single point of access for settings.yaml
├── ingestion/
│   ├── pdf_loader.py        # PDF -> per-page raw text
│   ├── text_cleaner.py      # de-hyphenation + whitespace normalization
│   └── chunker.py           # sentence segmentation with char offsets
├── db/
│   ├── schema.sql           # full schema (Phase 1 tables + future-phase tables, empty)
│   ├── models.py            # Paper, Chunk dataclasses
│   └── repository.py        # the ONLY module that talks to SQLite directly
├── pipeline/
│   └── orchestrator.py      # wires ingestion -> cleaning -> chunking -> DB
├── utils/
│   ├── logging_config.py    # shared logger setup (console + file)
│   └── exceptions.py        # typed exceptions (PDFLoadError, DatabaseError, ...)
├── scripts/
│   ├── init_db.py           # creates the database schema
│   └── make_fixture_pdf.py  # regenerates the test fixture PDF
├── tests/
│   ├── unit/                 # one test file per module
│   ├── integration/          # full pipeline test
│   └── fixtures/
│       └── sample_paper.pdf  # 2-page fixture PDF (already included)
├── data/
│   ├── raw_pdfs/             # put your real PDFs here (empty in the repo)
│   └── aerf.db                # created on first run — not committed
├── main.py                    # CLI entry point
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## 2. Windows + VS Code Setup (exact steps)

**Prerequisites:** Python 3.10+ installed and on PATH (check with `python --version` in a new terminal). VS Code with the Python extension installed.

1. Open the `aerf-engine` folder in VS Code (`File > Open Folder...`).

2. Open a terminal in VS Code (`` Ctrl+` ``) — it defaults to PowerShell on Windows.

3. Create a virtual environment:
   ```powershell
   python -m venv venv
   ```

4. Activate it:
   ```powershell
   venv\Scripts\activate
   ```
   If PowerShell blocks the script with an "execution policy" error, run this once (in an admin PowerShell, or just for your user) and try activating again:
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```
   You should now see `(venv)` at the start of your terminal prompt.

5. In VS Code, select the venv's interpreter: `Ctrl+Shift+P` -> "Python: Select Interpreter" -> choose the one under `.\venv\Scripts\python.exe`.

6. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 3. Initialize the Database

From the `aerf-engine` folder, with the venv active:

```powershell
python scripts\init_db.py
```

This creates `data\aerf.db` with all tables from `db\schema.sql` (Phase 1 tables `papers`/`chunks` will actually be used; the rest exist now as empty tables for later phases, per the approved architecture). Safe to re-run — it won't drop existing data.

---

## 4. Provide a PDF (fixture already included)

A ready-to-use fixture PDF is already at `tests\fixtures\sample_paper.pdf` — a small 2-page synthetic paper built specifically to exercise Phase 1 (multi-page extraction, a word hyphenated across a line break, and an "et al." abbreviation that should not be mistaken for a sentence boundary). You don't need to supply your own PDF just to try Phase 1.

To regenerate it (or see exactly what it contains) run:
```powershell
python scripts\make_fixture_pdf.py
```

**To ingest your own real papers:** drop `.pdf` files into `data\raw_pdfs\` and either point `main.py --file` at one directly, or use `main.py --dir data\raw_pdfs` to ingest all of them. Any real, text-based (non-scanned) PDF works — scanned/image-only PDFs are out of scope for Phase 1 since no OCR is implemented, and will raise a clear `PDFLoadError` explaining why.

---

## 5. Run the Phase 1 Pipeline

Ingest a single file:
```powershell
python main.py --file tests\fixtures\sample_paper.pdf
```

Ingest every PDF in a folder:
```powershell
python main.py --dir data\raw_pdfs
```

You can also initialize the DB and ingest in one command:
```powershell
python main.py --init-db --file tests\fixtures\sample_paper.pdf
```

Re-running on the same filename is safe — it's detected as already ingested and skipped (no duplicate rows).

---

## 6. Run the Tests

```powershell
pytest -v
```

Run only unit tests, or only the integration test:
```powershell
pytest tests\unit -v
pytest tests\integration -v
```

---

## 7. Expected Output

**Running `python main.py --init-db --file tests\fixtures\sample_paper.pdf`:**
```
2026-XX-XX ... | INFO | aerf.db.repository | Database initialized at ...\data\aerf.db
2026-XX-XX ... | INFO | aerf.pipeline.orchestrator | Starting Phase 1 ingestion for: tests\fixtures\sample_paper.pdf
2026-XX-XX ... | INFO | aerf.ingestion.pdf_loader | Extracted 2 page(s) from sample_paper.pdf (2 with text).
2026-XX-XX ... | INFO | aerf.db.repository | Inserted paper id=1 (sample_paper.pdf)
2026-XX-XX ... | INFO | aerf.db.repository | Inserted 7 chunk(s) for paper_id=1
2026-XX-XX ... | INFO | aerf.pipeline.orchestrator | Finished ingesting 'sample_paper.pdf': paper_id=1, 2 page(s), 7 chunk(s).

Ingested 'sample_paper.pdf' (paper_id=1, pages=2, chunks=7)
```
A matching log file is also written to `logs\aerf_engine.log`.

**Running `pytest -v`:**
```
tests/integration/test_phase1_pipeline.py::test_ingest_paper_end_to_end PASSED
...
tests/unit/test_text_cleaner.py::test_clean_text_strips_leading_trailing_whitespace PASSED
======================== 36 passed in 0.6s ========================
```

**Inspecting the database directly** (optional, useful for demoing provenance in your viva):
```powershell
python -c "import sqlite3; c = sqlite3.connect('data/aerf.db'); c.row_factory = sqlite3.Row; [print(dict(r)) for r in c.execute('SELECT chunk_id, page_number, char_start, char_end, text FROM chunks LIMIT 5')]"
```
Each row shows exactly which paper, page, and character span a chunk of text came from — this is the provenance chain that later phases (entities, claims, AERF scores) will all link back to.

---

## 8. Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'pdfplumber'` (or yaml/pytest) | Dependencies not installed, or venv not active | Confirm `(venv)` shows in your prompt, then `pip install -r requirements.txt` |
| `settings.yaml not found at ...` | Running a script from the wrong working directory | Always run commands from inside `aerf-engine\` (the folder containing `main.py`) |
| PowerShell: `cannot be loaded because running scripts is disabled` | Windows execution policy blocks venv activation | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then retry `venv\Scripts\activate` |
| `PDFLoadError: PDF ... produced no extractable text on any page` | The PDF is scanned/image-only (no text layer) | Out of scope for Phase 1 — use a text-based PDF, or wait for an OCR step in a later phase if you need scanned support |
| `DatabaseError: Paper with source_filename='...' already exists` | Re-ingesting the same filename after deleting and re-adding it under a different path | This is expected — `ingest_paper()` normally handles this gracefully by returning the existing paper; this error only surfaces if you call `repository.insert_paper()` directly, bypassing the orchestrator |
| `sqlite3.OperationalError: database is locked` | Another process (e.g. a DB browser tool) has `data\aerf.db` open | Close any other program that has the `.db` file open, then retry |
| Tests can't import `ingestion`, `db`, etc. | Running `pytest` from outside `aerf-engine\`, or `pytest.ini` missing | Run `pytest` from the `aerf-engine\` folder; `pytest.ini` (included) sets `pythonpath = .` automatically |
| VS Code shows import errors (red squiggles) but code runs fine | VS Code is using the wrong Python interpreter | `Ctrl+Shift+P` -> "Python: Select Interpreter" -> pick `.\venv\Scripts\python.exe` |

---

## 9. Known Limitations (by design, for Phase 1)

- **Sentence segmentation is a regex heuristic**, not a trained model (deliberately — `nlp/`-based sentencizers are a Phase 2+ concern). It handles common abbreviations (`et al.`, `Fig.`, `e.g.`, etc.) but can still mis-split unusual scientific notation. This is expected and acceptable for the ingestion layer.
- **Paper metadata** (`title`, `authors`, `publication_year`, `venue`) is left `NULL` in Phase 1 — no metadata-extraction step exists yet. The columns exist in the schema for later phases to populate (e.g. from PDF headers or a manual entry step).
- **No OCR** — scanned/image-only PDFs will fail with a clear error rather than silently producing empty output.
- **Paragraph structure is not preserved** after cleaning (see `text_cleaner.py` docstring) — Phase 1 chunks at sentence granularity, which is what later NLP stages need.

---

## 10. What's Next (not part of Phase 1 — do not implement yet)

Phase 2 will add `nlp/ner`, `nlp/claims`, `nlp/relations`, `nlp/embeddings` on top of these chunks, without modifying `ingestion/`, `db/repository.py`'s existing functions, or this pipeline's Phase 1 behavior.

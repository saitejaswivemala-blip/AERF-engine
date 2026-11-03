"""
Unit tests for main.py's load_metadata_by_filename(), which maps a
paper-manifest JSON's "papers" list to {pdf basename: metadata dict} for
ingest_paper()/ingest_directory()'s `metadata`/`metadata_by_filename`
arguments (see pipeline/orchestrator.py).
"""
import json

from main import load_metadata_by_filename


def _write_manifest(tmp_path, papers):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"papers": papers}))
    return path


def test_maps_pdf_path_basename_to_metadata(tmp_path):
    path = _write_manifest(tmp_path, [
        {"pdf_path": "experiments/datasets/pdfs/P1_foo.pdf", "title": "T1",
         "authors": "A1", "year": 2024, "venue": "V1"},
    ])
    result = load_metadata_by_filename(path)
    assert result == {
        "P1_foo.pdf": {"title": "T1", "authors": "A1", "publication_year": 2024, "venue": "V1"},
    }


def test_maps_manifest_year_field_to_publication_year_key(tmp_path):
    # "year" in the manifest, "publication_year" in the Paper/DB field --
    # this mapping is the one thing this function exists to do correctly.
    path = _write_manifest(tmp_path, [
        {"pdf_path": "x/y.pdf", "title": None, "authors": None, "year": 2019, "venue": None},
    ])
    result = load_metadata_by_filename(path)
    assert result["y.pdf"]["publication_year"] == 2019
    assert "year" not in result["y.pdf"]


def test_handles_multiple_papers(tmp_path):
    path = _write_manifest(tmp_path, [
        {"pdf_path": "a.pdf", "title": "A", "authors": None, "year": None, "venue": None},
        {"pdf_path": "b.pdf", "title": "B", "authors": None, "year": None, "venue": None},
    ])
    result = load_metadata_by_filename(path)
    assert set(result) == {"a.pdf", "b.pdf"}
    assert result["a.pdf"]["title"] == "A"
    assert result["b.pdf"]["title"] == "B"


def test_missing_optional_fields_default_to_none(tmp_path):
    path = _write_manifest(tmp_path, [{"pdf_path": "a.pdf"}])
    result = load_metadata_by_filename(path)
    assert result["a.pdf"] == {"title": None, "authors": None, "publication_year": None, "venue": None}


def test_empty_papers_list_gives_empty_map(tmp_path):
    path = _write_manifest(tmp_path, [])
    assert load_metadata_by_filename(path) == {}

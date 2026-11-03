"""
Regression test for scripts/run_experiment.py's stage_ingest().

Bug this guards against: stage_ingest() deletes and recreates
experiments/runs/experiment.db on every run (by design -- it's the
reproducible-from-scratch ingest stage), but used to call
ingest_paper(pdf_path, repository=repo) with no metadata argument, even
though paper_manifest.json (which this same function already reads) has
real title/authors/year/venue for every paper. That meant re-running this
stage would silently wipe out any previously-applied metadata fix. Fixed
to load metadata via main.load_metadata_by_filename() and pass it through.

Runs in an isolated tmp_path with its own experiments/datasets/,
experiments/runs/ and a copy of the real fixture PDF -- never touches the
real experiments/runs/experiment.db, since stage_ingest() unconditionally
deletes and recreates whatever "experiments/runs/experiment.db" resolves
to relative to the current working directory.
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PDF = PROJECT_ROOT / "tests" / "fixtures" / "sample_paper.pdf"


def _load_run_experiment_module():
    spec = importlib.util.spec_from_file_location(
        "run_experiment", PROJECT_ROOT / "scripts" / "run_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_experiment"] = module
    spec.loader.exec_module(module)
    return module


def _make_isolated_project(tmp_path):
    """Sets up just enough of the project layout under tmp_path for
    stage_ingest() to run against, with a real (fixture) PDF and a real
    manifest containing real-looking metadata."""
    (tmp_path / "experiments" / "datasets" / "pdfs").mkdir(parents=True)
    (tmp_path / "experiments" / "runs").mkdir(parents=True)
    (tmp_path / "experiments" / "results").mkdir(parents=True)

    pdf_dest = tmp_path / "experiments" / "datasets" / "pdfs" / "sample_paper.pdf"
    shutil.copy(FIXTURE_PDF, pdf_dest)

    manifest = {
        "papers": [
            {
                "paper_id": "P1",
                "pdf_path": "experiments/datasets/pdfs/sample_paper.pdf",
                "title": "A Real Test Title",
                "authors": "Real Test Authors",
                "year": 2023,
                "venue": "Real Test Venue",
            }
        ]
    }
    (tmp_path / "experiments" / "datasets" / "paper_manifest.json").write_text(json.dumps(manifest))


def test_stage_ingest_persists_real_manifest_metadata(tmp_path, monkeypatch):
    _make_isolated_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, str(PROJECT_ROOT))

    run_experiment = _load_run_experiment_module()
    run_experiment.stage_ingest()

    from db.repository import Repository
    repo = Repository(tmp_path / "experiments" / "runs" / "experiment.db")
    paper = repo.get_paper_by_filename("sample_paper.pdf")

    assert paper.title == "A Real Test Title"
    assert paper.authors == "Real Test Authors"
    assert paper.publication_year == 2023
    assert paper.venue == "Real Test Venue"


def test_stage_ingest_metadata_survives_a_rerun(tmp_path, monkeypatch):
    """The actual regression: stage_ingest() deletes and recreates the
    database every call -- metadata must be present again after a SECOND
    run, not just the first."""
    _make_isolated_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, str(PROJECT_ROOT))

    run_experiment = _load_run_experiment_module()
    run_experiment.stage_ingest()
    run_experiment.stage_ingest()  # simulates re-running the reproducible stage

    from db.repository import Repository
    repo = Repository(tmp_path / "experiments" / "runs" / "experiment.db")
    paper = repo.get_paper_by_filename("sample_paper.pdf")
    assert paper.title == "A Real Test Title"

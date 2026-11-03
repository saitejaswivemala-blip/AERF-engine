from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

from db.models import Paper
from db.repository import Repository
from pipeline.orchestrator import ingest_paper

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "experiments" / "runs"
SESSIONS_DIR = DB_DIR / "sessions"
REGISTRY_FILE = SESSIONS_DIR / "session_registry.json"


def generate_session_id() -> str:
    """Generate a unique, human-readable session ID with timestamp and entropy."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_suffix = uuid.uuid4().hex[:6]
    return f"session_{timestamp}_{rand_suffix}"


def get_session_db_path(session_id: str) -> Path:
    """Return the canonical database path for a given session."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    # Check if a db file exists in SESSIONS_DIR or DB_DIR
    in_sessions = SESSIONS_DIR / f"{session_id}.db"
    in_db_dir = DB_DIR / f"{session_id}.db"
    if in_sessions.exists():
        return in_sessions
    if in_db_dir.exists():
        return in_db_dir
    # Default location for new session databases
    return SESSIONS_DIR / f"{session_id}.db"


def _read_registry() -> dict:
    if not REGISTRY_FILE.exists():
        return {}
    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not read session registry: %s", exc)
        return {}


def _save_registry(registry: dict) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, default=str)
    except Exception as exc:
        logger.warning("Could not save session registry: %s", exc)


def create_session(
    session_id: str | None = None,
    name: str | None = None,
    custom_db_path: Path | None = None,
) -> tuple[str, Repository]:
    """
    Initialize a new isolated analysis session with its own database.
    Guarantees no modifications to the existing corpus database.
    """
    sid = session_id or generate_session_id()
    db_path = custom_db_path or get_session_db_path(sid)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    repo = Repository(db_path)
    repo.initialize_database()
    repo.ensure_phase2_schema()
    repo.ensure_phase3_schema()
    repo.ensure_session_schema()

    disp_name = name or f"Session {datetime.now().strftime('%b %d, %Y %H:%M')}"
    repo.create_session(sid, name=disp_name, status="created", paper_count=0)

    # Update persistent registry
    registry = _read_registry()
    registry[sid] = {
        "session_id": sid,
        "name": disp_name,
        "created_at": datetime.now().isoformat(),
        "status": "created",
        "db_path": str(db_path),
        "paper_count": 0,
        "papers": [],
    }
    _save_registry(registry)

    logger.info("Created isolated session %s at %s", sid, db_path)
    return sid, repo


def get_session_repository(session_id: str) -> Repository:
    """Return a Repository instance pointing to the session's database."""
    db_path = get_session_db_path(session_id)
    repo = Repository(db_path)
    repo.ensure_session_schema()
    return repo


def get_session(session_id: str) -> dict | None:
    """Retrieve full metadata for a session."""
    db_path = get_session_db_path(session_id)
    if not db_path.exists():
        # Check registry
        registry = _read_registry()
        return registry.get(session_id)

    try:
        repo = Repository(db_path)
        repo.ensure_session_schema()
        info = repo.get_session(session_id)
        if info:
            info["db_path"] = str(db_path)
            # Count claims & pairs
            paper_ids = info.get("paper_ids", [])
            info["claim_count"] = len(repo.get_claims_by_paper_ids(paper_ids))
            info["pair_count"] = len(repo.get_claim_pairs_by_paper_ids(paper_ids))
            return info
    except Exception as exc:
        logger.warning("Error fetching session %s from DB: %s", session_id, exc)

    registry = _read_registry()
    return registry.get(session_id)


def list_sessions() -> list[dict]:
    """
    List all available sessions from the registry and scanning databases.
    Returns newest first.
    """
    sessions_by_id: dict[str, dict] = {}

    # 1. Read registry
    registry = _read_registry()
    for sid, data in registry.items():
        if sid.startswith(("test_", "sess_", "integrity_")) or "test" in sid.lower():
            continue
        sessions_by_id[sid] = dict(data)

    # 2. Scan SESSIONS_DIR and DB_DIR for session_*.db files
    candidate_dirs = [SESSIONS_DIR, DB_DIR]
    for cdir in candidate_dirs:
        if not cdir.exists():
            continue
        for db_file in cdir.glob("session_*.db"):
            sid = db_file.stem
            if sid.startswith(("test_", "sess_", "integrity_")) or "test" in sid.lower():
                continue
            if sid not in sessions_by_id:
                sessions_by_id[sid] = {
                    "session_id": sid,
                    "name": sid.replace("_", " ").title(),
                    "created_at": datetime.fromtimestamp(db_file.stat().st_mtime).isoformat(),
                    "status": "completed",
                    "db_path": str(db_file),
                    "paper_count": 0,
                    "papers": [],
                }

    # 3. Enrich with live DB metadata if possible
    results = []
    for sid, info in sessions_by_id.items():
        db_path = Path(info.get("db_path", get_session_db_path(sid)))
        if db_path.exists():
            try:
                repo = Repository(db_path)
                repo.ensure_session_schema()
                db_session = repo.get_session(sid)
                if db_session:
                    info["status"] = db_session.get("status", info.get("status", "completed"))
                    info["name"] = db_session.get("name") or info.get("name")
                    info["paper_count"] = db_session.get("paper_count", 0)
                    info["papers"] = db_session.get("papers", [])
                    paper_ids = db_session.get("paper_ids", [])
                    info["claim_count"] = len(repo.get_claims_by_paper_ids(paper_ids))
                    info["pair_count"] = len(repo.get_claim_pairs_by_paper_ids(paper_ids))
                else:
                    papers = repo.list_papers()
                    info["paper_count"] = len(papers)
                    info["papers"] = [
                        {"paper_id": p.paper_id, "source_filename": p.source_filename, "title": p.title}
                        for p in papers
                    ]
                    info["claim_count"] = len(repo.get_all_claims())
            except Exception:
                pass
        results.append(info)

    # Sort descending by created_at
    results.sort(key=lambda s: str(s.get("created_at", "")), reverse=True)
    return results


def get_session_results(session_id: str, repo: Repository | None = None) -> dict:
    """
    Retrieve all results strictly scoped to this session's paper IDs.
    Guarantees no records from any external or global corpus are returned.
    """
    if repo is None:
        repo = get_session_repository(session_id)

    repo.ensure_session_schema()
    session_info = repo.get_session(session_id) or {"session_id": session_id, "status": "completed"}
    paper_ids = repo.get_session_paper_ids(session_id)

    if not paper_ids:
        # Fallback to all papers in this session repository
        papers = repo.list_papers()
        paper_ids = [p.paper_id for p in papers]
    else:
        papers = repo.get_session_papers(session_id)

    claims = repo.get_claims_by_paper_ids(paper_ids)
    pairs = repo.get_claim_pairs_by_paper_ids(paper_ids)
    scores = repo.get_aerf_scores_by_paper_ids(paper_ids)
    graph_edges = repo.get_graph_edges_by_paper_ids(paper_ids)
    gaps = repo.get_gaps_by_paper_ids(paper_ids)

    # Calculate summary metrics strictly within this session
    supports_cnt = sum(1 for p in pairs if p.get("validated_as") == "supports")
    contradicts_cnt = sum(1 for p in pairs if p.get("validated_as") == "contradicts")
    neutral_cnt = sum(1 for p in pairs if p.get("validated_as") == "neutral")
    mean_score = round(sum(s["score"] for s in scores) / len(scores), 4) if scores else 0.0

    return {
        "session_id": session_id,
        "session_info": session_info,
        "paper_ids": paper_ids,
        "papers": [
            {
                "paper_id": p.paper_id,
                "source_filename": p.source_filename,
                "title": p.title,
                "publication_year": p.publication_year,
                "page_count": p.page_count,
            }
            for p in papers
        ],
        "claims": claims,
        "pairs": pairs,
        "scores": scores,
        "graph_edges": graph_edges,
        "gaps": gaps,
        "summary": {
            "total_papers": len(papers),
            "total_claims": len(claims),
            "total_pairs": len(pairs),
            "supports_count": supports_cnt,
            "contradicts_count": contradicts_cnt,
            "neutral_count": neutral_cnt,
            "total_scores": len(scores),
            "mean_score": mean_score,
            "total_graph_edges": len(graph_edges),
            "total_gaps": len(gaps),
        },
    }


import inspect

def _fmt_time(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s}s"
    m = s // 60
    rem = s % 60
    return f"{m}m {rem:02d}s"


def _safe_run_pipeline(pipe, **kwargs):
    """Call pipe.run(**kwargs), omitting kwargs not supported by custom/mock pipeline objects."""
    try:
        sig = inspect.signature(pipe.run)
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_kwargs:
            return pipe.run(**kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return pipe.run(**filtered)
    except Exception:
        # Fallback to direct call if signature inspection fails
        return pipe.run(**kwargs)


def run_session_analysis(
    session_id: str,
    pdf_paths: list[Path | str],
    phase2_pipeline=None,
    phase3_pipeline=None,
    progress_callback: Callable[[dict | int, str | None], None] | None = None,
    cancel_checker: Callable[[], bool] | None = None,
    repo: Repository | None = None,
) -> dict:
    """
    Execute end-to-end multi-paper analysis in an isolated session:
      Phase 1: Ingest only uploaded PDFs into session DB
      Phase 2: Extract NER, Claims, Relations, Embeddings (with chunk checkpointing)
      Phase 3: Cross-paper NLI matching, AERF Scoring, Graph, Gaps (scoped to session)
    """
    if not pdf_paths:
        raise ValueError("At least one PDF file must be provided for session analysis.")

    if repo is None:
        repo = get_session_repository(session_id)

    def report(percent: int, message: str, raw_info: dict | None = None):
        logger.info("[%s %d%%] %s", session_id, percent, message)
        if progress_callback:
            try:
                if raw_info is not None:
                    try:
                        progress_callback(raw_info)
                    except TypeError:
                        progress_callback(percent, message)
                else:
                    progress_callback(percent, message)
            except Exception:
                pass

    try:
        repo.update_session(session_id, status="processing")
        report(5, "Initializing isolated session database...", raw_info={"stage": "init", "percent": 5, "message": "Initializing session database..."})

        # Step 1: Ingestion (Phase 1)
        uploaded_papers: list[Paper] = []
        num_files = len(pdf_paths)
        for i, pdf_path in enumerate(pdf_paths, start=1):
            if cancel_checker and cancel_checker():
                raise InterruptedError("Analysis cancelled by user.")
            path = Path(pdf_path)
            pct = 5 + int(20 * i / num_files)
            msg = f"Ingesting PDF {i}/{num_files}: {path.name}"
            report(pct, msg, raw_info={
                "stage": "phase1",
                "percent": pct,
                "message": msg,
                "file_index": i,
                "total_files": num_files,
                "filename": path.name,
            })
            paper = ingest_paper(path, repository=repo, metadata={})
            if paper is None or not getattr(paper, "paper_id", None):
                raise RuntimeError(f"Ingestion failed for {path.name}")
            repo.link_session_paper(session_id, paper.paper_id, paper.source_filename)
            uploaded_papers.append(paper)

        unique_paper_ids = sorted({p.paper_id for p in uploaded_papers})
        repo.update_session(session_id, paper_count=len(unique_paper_ids))

        # Step 2: Information Extraction (Phase 2)
        report(25, "Extracting claims, entities, relations and embeddings...", raw_info={"stage": "phase2_start", "percent": 25, "message": "Starting Phase 2 information extraction..."})
        if phase2_pipeline is None:
            from pipeline.phase2 import Phase2Pipeline
            phase2_pipeline = Phase2Pipeline(repo)
        else:
            phase2_pipeline.repo = repo

        for i, pid in enumerate(unique_paper_ids, start=1):
            if cancel_checker and cancel_checker():
                raise InterruptedError("Analysis cancelled by user.")
            paper_obj = repo.get_paper_by_id(pid)
            pname = paper_obj.source_filename if paper_obj else f"Paper {pid}"

            def p2_sub_cb(info):
                if isinstance(info, dict):
                    ch_done = info.get("processed_chunks", 0)
                    ch_tot = max(1, info.get("total_chunks", 1))
                    frac_p = ch_done / ch_tot
                    frac_total_p2 = (i - 1 + frac_p) / len(unique_paper_ids)
                    pct = int(25 + 45 * frac_total_p2)
                    sub_msg = (
                        f"[Paper {i}/{len(unique_paper_ids)}: {pname}] "
                        f"Chunk {ch_done}/{ch_tot} ({int(frac_p*100)}%) | "
                        f"{info.get('claims_found', 0)} claims | "
                        f"Elapsed: {_fmt_time(info.get('elapsed_s', 0))} | "
                        f"ETA: {_fmt_time(info.get('eta_s', 0))}"
                    )
                    report(pct, sub_msg, raw_info={
                        **info,
                        "percent": pct,
                        "message": sub_msg,
                        "paper_name": pname,
                        "paper_idx": i,
                        "total_papers": len(unique_paper_ids),
                    })

            _safe_run_pipeline(
                phase2_pipeline,
                paper_id=pid,
                progress_callback=p2_sub_cb,
                cancel_checker=cancel_checker,
            )

        # Step 3: Intra-Session Reasoning & Scored Evidence (Phase 3)
        report(70, "Evaluating cross-paper relationships, evidence, and reliability...", raw_info={"stage": "phase3_start", "percent": 70, "message": "Starting cross-paper reasoning..."})
        if phase3_pipeline is None:
            from pipeline.phase3 import Phase3Pipeline
            phase3_pipeline = Phase3Pipeline(repo)
        else:
            phase3_pipeline.repo = repo
            if hasattr(phase3_pipeline, "scorer"):
                phase3_pipeline.scorer.repo = repo
            if hasattr(phase3_pipeline, "updater"):
                phase3_pipeline.updater.repo = repo
                if hasattr(phase3_pipeline.updater, "scorer"):
                    phase3_pipeline.updater.scorer.repo = repo

        def p3_sub_cb(info):
            if isinstance(info, dict):
                cur = info.get("current_pair", 0)
                tot = max(1, info.get("total_pairs", 1))
                frac = cur / tot
                pct = int(70 + 25 * frac)
                sub_msg = (
                    f"[Evidence Reasoning] Pair {cur}/{tot} ({int(frac*100)}%) | "
                    f"Supports: {info.get('supports', 0)}, Contradicts: {info.get('contradicts', 0)} | "
                    f"Elapsed: {_fmt_time(info.get('elapsed_s', 0))} | "
                    f"ETA: {_fmt_time(info.get('eta_s', 0))}"
                )
                report(pct, sub_msg, raw_info={
                    **info,
                    "percent": pct,
                    "message": sub_msg,
                })

        # Run Phase 3 strictly scoped to the session's paper IDs!
        _safe_run_pipeline(
            phase3_pipeline,
            paper_ids=unique_paper_ids,
            progress_callback=p3_sub_cb,
            cancel_checker=cancel_checker,
        )
        report(95, "Finalizing session graph and research gaps...", raw_info={"stage": "finalize", "percent": 95, "message": "Finalizing graph and gaps..."})

        # Step 4: Finalize
        repo.update_session(session_id, status="completed", paper_count=len(unique_paper_ids))

        # Update registry
        registry = _read_registry()
        if session_id in registry:
            registry[session_id]["status"] = "completed"
            registry[session_id]["paper_count"] = len(unique_paper_ids)
            registry[session_id]["papers"] = [
                {"paper_id": p.paper_id, "source_filename": p.source_filename}
                for p in uploaded_papers
            ]
            _save_registry(registry)

        report(100, "Analysis completed successfully!", raw_info={"stage": "completed", "percent": 100, "message": "Analysis completed successfully!"})
        return get_session_results(session_id, repo=repo)

    except InterruptedError as ie:
        logger.warning("Session %s stopped: %s", session_id, ie)
        repo.update_session(session_id, status="stopped", error_message=str(ie))
        registry = _read_registry()
        if session_id in registry:
            registry[session_id]["status"] = "stopped"
            registry[session_id]["error_message"] = str(ie)
            _save_registry(registry)
        report(0, f"Analysis stopped: {ie}", raw_info={"stage": "stopped", "percent": 0, "message": str(ie)})
        raise

    except Exception as exc:
        logger.exception("Session %s failed: %s", session_id, exc)
        repo.update_session(session_id, status="failed", error_message=str(exc))
        registry = _read_registry()
        if session_id in registry:
            registry[session_id]["status"] = "failed"
            registry[session_id]["error_message"] = str(exc)
            _save_registry(registry)
        raise

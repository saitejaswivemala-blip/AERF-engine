"""
Safely re-run Phase 2 claim extraction for chosen papers with the new extractor.

  python scripts/reprocess_paper_claims.py --db experiments/runs/experiment_FINAL_2026-09-02.db --paper-id 8 --paper-id 9

What it does (nothing is done to the database file itself except these steps):
  1. Copies the DB to <name>.backup_<timestamp>.db  (never deleted/overwritten)
  2. For each paper: deletes ONLY that paper's old claims (their entity links,
     relations, embeddings, pairs, scores cascade) in ONE transaction. Entities
     are kept (INSERT OR IGNORE -> no duplicates).
  3. Re-runs Phase 2 (force) and prints counts before/after.
  4. Optional --phase3 re-runs cross-paper matching afterwards.
Use --dry-run to only print counts.
"""
import argparse, shutil, sqlite3, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db.repository import Repository


def counts(db, pid):
    c = sqlite3.connect(db)
    q = lambda sql: c.execute(sql, (pid,)).fetchone()[0]
    r = {"chunks": q("select count(*) from chunks where paper_id=?"),
         "entities": q("select count(*) from entities where paper_id=?"),
         "claims": q("select count(*) from claims where paper_id=?"),
         "embeddings": q("select count(*) from claim_embeddings where claim_id in (select claim_id from claims where paper_id=?)"),
         "relations": q("select count(*) from relations where claim_id in (select claim_id from claims where paper_id=?)")}
    c.close(); return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True); ap.add_argument("--paper-id", type=int, action="append", required=True)
    ap.add_argument("--phase3", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    db = Path(a.db)
    for pid in a.paper_id: print("BEFORE paper", pid, counts(db, pid))
    if a.dry_run: return
    bak = db.with_name(f"{db.stem}.backup_{time.strftime('%Y%m%d_%H%M%S')}{db.suffix}")
    shutil.copy2(db, bak); print("Backup:", bak)
    repo = Repository(db); repo.ensure_phase2_schema()
    try: repo.ensure_phase3_schema()
    except Exception: pass
    for pid in a.paper_id:
        conn = sqlite3.connect(db); conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn: conn.execute("DELETE FROM claims WHERE paper_id=?", (pid,))
        finally: conn.close()
    from pipeline.phase2 import Phase2Pipeline
    p2 = Phase2Pipeline(repo)
    for pid in a.paper_id: print("Phase2", p2.run(pid, force=True))
    if a.phase3:
        from pipeline.phase3 import Phase3Pipeline
        print("Phase3", Phase3Pipeline(repo).run())
    for pid in a.paper_id: print("AFTER paper", pid, counts(db, pid))

if __name__ == "__main__": main()

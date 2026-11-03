import sqlite3
from db.repository import Repository

def test_phase2_tables_are_in_same_database(tmp_path):
    db=tmp_path/"aerf.db"
    r=Repository(db); r.initialize_database(); r.ensure_phase2_schema()
    with sqlite3.connect(db) as c:
        tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"entities","claims","claim_entities","relations","claim_embeddings"} <= tables

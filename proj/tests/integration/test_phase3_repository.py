import sqlite3
from db.repository import Repository

def test_phase3_tables_are_in_same_database(tmp_path):
    db=tmp_path/'aerf.db'; r=Repository(db); r.initialize_database(); r.ensure_phase2_schema(); r.ensure_phase3_schema()
    with sqlite3.connect(db) as c:
        names={x[0] for x in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {'claim_pairs','aerf_scores','aerf_score_history','evidence_graph_edges','knowledge_evolution_snapshots','research_gaps','explanations'} <= names

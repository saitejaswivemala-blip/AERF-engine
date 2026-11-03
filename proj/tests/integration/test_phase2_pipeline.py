import sqlite3
import numpy as np
from db.repository import Repository
from db.models import Paper, Chunk
from pipeline.phase2 import Phase2Pipeline
from nlp.ner import EntityResult
from nlp.claim_extractor import ClaimResult

class FakeNER:
    def __init__(self):
        self.nlp=lambda text: None
    def extract(self,text):
        return [EntityResult("treatment","ENTITY",4,13),EntityResult("disease","ENTITY",22,29)]
class FakeClaims:
    def extract(self,text):
        return [ClaimResult(text,0,len(text),.9)]
class FakeRelations:
    def extract(self,text,entities=None): return []
class FakeEmbed:
    model_name="fake"
    def encode(self,texts): return [np.array([0.1,0.2,0.3], dtype=np.float32) for _ in texts]

def test_phase2_persists_provenance_and_embedding(tmp_path):
    db=tmp_path/"aerf.db"; repo=Repository(db); repo.initialize_database()
    p=repo.insert_paper(Paper("p.pdf","p.pdf",1))
    repo.insert_chunks([Chunk(p.paper_id,1,10,49,"The treatment reduced disease severity.")])
    pipe=object.__new__(Phase2Pipeline)
    pipe.repo=repo; pipe.ner=FakeNER(); pipe.claims=FakeClaims(); pipe.relations=FakeRelations(); pipe.embedder=FakeEmbed()
    result=pipe.run(p.paper_id)
    assert result["claims"]==1 and result["entities"]==2 and result["embeddings"]==1
    with sqlite3.connect(db) as c:
        row=c.execute("SELECT text,span_start,span_end FROM claims").fetchone()
        assert row[0]=="The treatment reduced disease severity." and row[1:]==(0,39)
        assert c.execute("SELECT COUNT(*) FROM claim_embeddings").fetchone()[0]==1

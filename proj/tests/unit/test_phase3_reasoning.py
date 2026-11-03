import numpy as np
from db.repository import Repository
from db.models import Paper, Chunk
from matching.claim_matcher import ClaimMatcher
from nlp.nli.nli_classifier import NLIResult
from aerf.scorer import AERFScorer
from aerf.adaptive_updater import AdaptiveAERFUpdater

class FakeNLI:
    def __init__(self,label='contradiction',confidence=.9): self.label=label; self.confidence=confidence
    def predict_bidirectional(self,a,b):
        e=c=n=0.0
        if self.label=='entailment': e=self.confidence
        elif self.label=='contradiction': c=self.confidence
        else: n=self.confidence
        return NLIResult(self.label,e,c,n,self.confidence,'fake-nli')

def test_similarity_alone_never_creates_contradiction():
    m=ClaimMatcher(FakeNLI('contradiction',.95), similarity_threshold=.5, entity_threshold=.2, nli_threshold=.6)
    a={'claim_id':1,'text':'A improves X'}; b={'claim_id':2,'text':'A improves X'}
    d=m.match(a,b,[{'text':'A'}],[{'text':'B'}],[],[],np.ones(3),np.ones(3))
    assert d is None

def test_aligned_nli_contradiction_is_validated():
    m=ClaimMatcher(FakeNLI('contradiction',.95), similarity_threshold=.5, entity_threshold=.2, nli_threshold=.6)
    a={'claim_id':1,'text':'A increases X'}; b={'claim_id':2,'text':'A decreases X'}
    d=m.match(a,b,[{'text':'A'},{'text':'X'}],[{'text':'A'},{'text':'X'}],[],[],np.ones(3),np.ones(3))
    assert d.validated_as=='contradicts'

def make_repo(tmp_path):
    r=Repository(tmp_path/'a.db'); r.initialize_database(); r.ensure_phase2_schema(); r.ensure_phase3_schema()
    p1=r.insert_paper(Paper('a.pdf','a.pdf',1,publication_year=2024,venue='Journal'))
    p2=r.insert_paper(Paper('b.pdf','b.pdf',1,publication_year=2024,venue='Journal'))
    r.insert_chunks([Chunk(p1.paper_id,1,0,20,'A increases X strongly.')])
    r.insert_chunks([Chunk(p2.paper_id,1,0,20,'A increases X strongly.')])
    class C: pass
    c=C(); c.text='A increases X strongly.'; c.start=0; c.end=len(c.text)
    c1=r.insert_claim_p2(p1.paper_id,1,c); c2=r.insert_claim_p2(p2.paper_id,2,c)
    for cid in (c1,c2):
        r.upsert_embedding_p2(cid,'fake',np.array([1,0,0],dtype=np.float32))
    return r,c1,c2,p1,p2

def test_aerf_is_bounded_and_deterministic(tmp_path):
    r,c1,c2,_,_=make_repo(tmp_path)
    s=AERFScorer(r)
    a=s.score(c1); b=s.score(c1)
    assert 0<=a.score<=1 and a==b

def test_adaptive_update_preserves_history_when_supporting_evidence_changes_score(tmp_path):
    r,c1,c2,p1,p2=make_repo(tmp_path)
    s=AERFScorer(r)
    base=s.score(c1); r.store_aerf_score(c1,base.score,base.factors,base.weights_version)
    r.upsert_claim_pair({'claim_a_id':c1,'claim_b_id':c2,'cosine_similarity':.95,'nli_label':'entailment','nli_confidence':.9,'entity_overlap_score':1.0,'validated_as':'supports','nli_entailment':.9,'nli_contradiction':.05,'nli_neutral':.05,'relation_alignment_score':0.0,'matching_method':'test','entity_alignment':['A'],'nli_model_name':'fake'})
    r.store_aerf_score(c2,.8,{'source':1,'methodology':1,'corroboration':0,'contradiction':0,'temporal':1,'semantic':1,'provenance':1},'test')
    changed=AdaptiveAERFUpdater(r,s).recompute({c1,c2},p1.paper_id,c2)
    assert changed
    with r._connect() as con:
        assert con.execute('SELECT COUNT(*) FROM aerf_score_history WHERE claim_id=?',(c1,)).fetchone()[0]>=1

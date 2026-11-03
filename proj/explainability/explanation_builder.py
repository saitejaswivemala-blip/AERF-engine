from __future__ import annotations

def build_claim_explanation(repo, claim_id, paper_ids=None, scoped_claim_ids=None):
    c=repo.get_claim(claim_id); score=repo.get_current_aerf(claim_id)
    if not c or not score: return None
    supports=repo.get_validated_neighbors(claim_id,'supports')
    contra=repo.get_validated_neighbors(claim_id,'contradicts')
    neutral=repo.get_validated_neighbors(claim_id,'neutral')
    if scoped_claim_ids is not None:
        supports = [x for x in supports if x['neighbor_id'] in scoped_claim_ids]
        contra = [x for x in contra if x['neighbor_id'] in scoped_claim_ids]
        neutral = [x for x in neutral if x['neighbor_id'] in scoped_claim_ids]
    elif paper_ids is not None:
        pids = set(paper_ids)
        claims = {x['claim_id'] for x in repo.get_claims_by_paper_ids(pids)}
        supports = [x for x in supports if x['neighbor_id'] in claims]
        contra = [x for x in contra if x['neighbor_id'] in claims]
        neutral = [x for x in neutral if x['neighbor_id'] in claims]
    p=repo.get_paper_by_id(c['paper_id'])
    provenance={'paper_id':c['paper_id'],'page_number':repo.get_claim_page(claim_id),'chunk_id':c['chunk_id'],'span_start':c['span_start'],'span_end':c['span_end'],'source_text':c['text'],'source_filename':None if p is None else p.source_filename}
    explanation={'claim_id':claim_id,'claim_text':c['text'],'aerf_score':score['score'],'factors':score['factor_breakdown'],'provenance':provenance,'supporting_evidence':[dict(x) for x in supports],'contradicting_evidence':[dict(x) for x in contra],'neutral_evidence':[dict(x) for x in neutral]}
    repo.store_explanation(claim_id,'aerf_reasoning',explanation)
    return explanation

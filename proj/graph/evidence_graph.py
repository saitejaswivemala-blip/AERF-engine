from __future__ import annotations

def sync_evidence_graph(repo, paper_ids=None):
    count=0
    all_claims=repo.get_all_claims()
    if paper_ids is not None:
        pids = set(paper_ids)
        claims = [c for c in all_claims if c['paper_id'] in pids]
        claim_ids = {c['claim_id'] for c in claims}
    else:
        claims = all_claims
        claim_ids = None

    for c in claims:
        for e in repo.get_claim_entities(c['claim_id']):
            repo.upsert_graph_edge('claim',c['claim_id'],'entity',e['entity_id'],'mentions',1.0,e['entity_id']); count+=1
        for r in repo.get_claim_relations(c['claim_id']):
            repo.upsert_graph_edge('entity',r['subject_entity_id'],'entity',r['object_entity_id'],r['predicate'],1.0,r['relation_id']); count+=1
        for kind,etype in (('supports','supports'),('contradicts','contradicts'),('neutral','neutral')):
            for n in repo.get_validated_neighbors(c['claim_id'],kind):
                if claim_ids is not None and n['neighbor_id'] not in claim_ids:
                    continue
                repo.upsert_graph_edge('claim',c['claim_id'],'claim',n['neighbor_id'],etype,float(n['nli_confidence'] or 0),n['pair_id']); count+=1
    return count

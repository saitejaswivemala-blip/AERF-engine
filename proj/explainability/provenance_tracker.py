def claim_provenance(repo, claim_id):
    c=repo.get_claim(claim_id)
    if not c: return None
    p=repo.get_paper_by_id(c['paper_id'])
    return {'paper_id':c['paper_id'],'source_filename':None if p is None else p.source_filename,'page_number':repo.get_claim_page(claim_id),'chunk_id':c['chunk_id'],'chunk_char_start':repo.get_chunk(c['chunk_id'])['char_start'],'chunk_char_end':repo.get_chunk(c['chunk_id'])['char_end'],'claim_span_start':c['span_start'],'claim_span_end':c['span_end'],'claim_text':c['text']}

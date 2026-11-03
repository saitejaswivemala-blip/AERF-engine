from nlp.claim_extractor import ScientificClaimExtractor
from nlp.ner import ScientificNER
from nlp.embeddings import ClaimEmbedder

class Tok:
    def __init__(self,pos): self.pos_=pos
class Sent:
    text="The treatment reduced disease severity."
    start_char=0; end_char=len(text)
    def __iter__(self): return iter([Tok("NOUN"),Tok("VERB")])
class Doc:
    sents=[Sent()]
class FakeNLP:
    def __call__(self,text): return Doc()
class FakeClassifier:
    def __call__(self,text,labels,multi_label=False):
        return {"labels":[labels[0],labels[1]],"scores":[0.91,0.03]}

def test_claim_extraction_is_semantic_and_preserves_exact_span():
    text=Sent.text
    ex=ScientificClaimExtractor(nlp=FakeNLP(), classifier=FakeClassifier(), threshold=.55)
    result=ex.extract(text)[0]
    assert result.text == text
    assert text[result.start:result.end] == result.text
    assert result.score == .91

def test_ner_model_is_configurable_without_loading_it():
    assert ScientificNER("custom-model").model_name == "custom-model"

def test_embedder_returns_empty_for_no_claims_without_loading_model():
    assert ClaimEmbedder().encode([]) == []


class RelationTok:
    def __init__(self, text, idx, lemma, dep, children=()):
        self.text=text; self.idx=idx; self.lemma_=lemma; self.dep_=dep; self.children=list(children)

class RelationDoc:
    def __init__(self, tokens): self.tokens=tokens
    def __iter__(self): return iter(self.tokens)

class RelationNLP:
    def __init__(self, doc): self.doc=doc
    def __call__(self, text): return self.doc


def test_relation_extractor_uses_dependency_subject_and_object():
    from nlp.relation_extractor import ScientificRelationExtractor
    subj=RelationTok("Vitamin",0,"vitamin","nsubj")
    obj=RelationTok("infection",22,"infection","dobj")
    pred=RelationTok("reduces",8,"reduce","ROOT",[subj,obj])
    doc=RelationDoc([subj,pred,obj])
    ents=[type("E",(),{"start_char":0,"end_char":6})(), type("E",(),{"start_char":22,"end_char":31})()]
    r=ScientificRelationExtractor(nlp=RelationNLP(doc)).extract("Vitamin reduces infection.", ents)
    assert [(x.predicate,x.subject_start,x.object_start) for x in r] == [("reduce",0,22)]


def test_relation_extractor_entity_fallback_when_dependency_links_missing():
    from nlp.relation_extractor import ScientificRelationExtractor
    pred=RelationTok("reduces",7,"reduce","ROOT",[])
    doc=RelationDoc([pred])
    ents=[type("E",(),{"start_char":0,"end_char":6})(), type("E",(),{"start_char":15,"end_char":24})()]
    r=ScientificRelationExtractor(nlp=RelationNLP(doc)).extract("Vitamin reduces infection.", ents)
    assert len(r) == 1 and r[0].predicate == "reduce"


def test_relation_extractor_does_not_create_relation_from_predicate_alone():
    from nlp.relation_extractor import ScientificRelationExtractor
    pred=RelationTok("reduces",0,"reduce","ROOT",[])
    doc=RelationDoc([pred])
    assert ScientificRelationExtractor(nlp=RelationNLP(doc)).extract("reduces", []) == []


def test_relation_extractor_ignores_generic_document_entities():
    from nlp.relation_extractor import ScientificRelationExtractor
    pred=RelationTok("treat",16,"treat","ROOT",[])
    doc=RelationDoc([pred])
    ents=[type("E",(),{"start_char":0,"end_char":14,"text":"abstract excerpt"})(), type("E",(),{"start_char":21,"end_char":38,"text":"experimental result"})()]
    r=ScientificRelationExtractor(nlp=RelationNLP(doc)).extract("abstract excerpt treat experimental result", ents)
    assert r == []


def test_relation_extractor_ignores_negated_predicate():
    from nlp.relation_extractor import ScientificRelationExtractor
    neg=RelationTok("not",0,"not","neg")
    pred=RelationTok("reduce",10,"reduce","ROOT",[neg])
    doc=RelationDoc([neg,pred])
    ents=[type("E",(),{"start_char":0,"end_char":6,"text":"Vitamin"})(), type("E",(),{"start_char":17,"end_char":21,"text":"risk"})()]
    r=ScientificRelationExtractor(nlp=RelationNLP(doc)).extract("Vitamin not reduce risk", ents)
    assert r == []

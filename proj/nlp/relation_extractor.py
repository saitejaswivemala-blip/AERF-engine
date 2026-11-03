from __future__ import annotations
from dataclasses import dataclass

PREDICATES = ("treat", "cause", "prevent", "improve", "increase", "reduce")

# Generic document-structure labels are not scientific entities and can create
# false triples when the parser/entity model links headings or abstract labels.
GENERIC_ENTITY_TERMS = {
    "abstract excerpt", "abstract", "experimental result", "result", "results",
    "discussion", "conclusion", "conclusions", "introduction", "background",
    "methods", "method", "study", "paper", "article", "figure", "table",
}
NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "didn't",
    "didnt",
    "did not",
    "isn't",
    "isnt",
    "wasn't",
    "wasnt",
    "cannot",
    "can't",
    "cant",
}

@dataclass(frozen=True)
class RelationResult:
    subject_start: int
    subject_end: int
    predicate: str
    object_start: int
    object_end: int

class ScientificRelationExtractor:
    """Conservative dependency-based extraction with an entity-anchored fallback.

    The fallback is only used when a predicate is a known verb but the parser does
    not expose an nsubj/obj dependency. It requires one entity before and one
    entity after the predicate, preventing relation creation from the predicate
    word alone.
    """
    def __init__(self, nlp=None):
        self._nlp = nlp

    def _doc(self, text):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_sci_sm")
        return self._nlp(text)

    @staticmethod
    def _entity_for_token(tok, ents):
        return next((e for e in ents if e.start_char <= tok.idx < e.end_char), None)

    @staticmethod
    def _fallback_entities(tok, ents):
        """Return nearest non-overlapping entities on either side of a predicate."""
        before = [e for e in ents if e.end_char <= tok.idx]
        after = [e for e in ents if e.start_char >= tok.idx + len(tok.text)]
        if not before or not after:
            return None, None
        # Nearest entity on each side; this is deliberately deterministic.
        return max(before, key=lambda e: e.end_char), min(after, key=lambda e: e.start_char)

    @staticmethod
    def _is_generic_entity(ent):
        text = getattr(ent, "text", "").strip().lower()
        return text in GENERIC_ENTITY_TERMS

    @staticmethod
    def _is_negated(tok):
        # Detect local dependency negation (preferred) and nearby negation words.
        if any(c.dep_ == "neg" for c in tok.children):
            return True
        start = max(0, tok.i - 4) if hasattr(tok, "i") else None
        if start is not None:
            try:
                return any(t.text.lower() in NEGATION_WORDS for t in tok.doc[start:tok.i])
            except Exception:
                pass
        return False

    def extract(self, text: str, entities=None) -> list[RelationResult]:
        doc = self._doc(text)
        ents = list(doc.ents) if entities is None else list(entities)
        ents = [e for e in ents if not self._is_generic_entity(e)]
        out = []
        seen = set()
        for tok in doc:
            pred = tok.lemma_.lower()
            if pred not in PREDICATES:
                continue
            # Do not emit a positive relation when the predicate is explicitly negated.
            # Claim-level NLI retains the full negated proposition for Phase 3.
            if self._is_negated(tok):
                continue

            subs = [c for c in tok.children if c.dep_ in {"nsubj", "nsubjpass"}]
            objs = [c for c in tok.children if c.dep_ in {"dobj", "obj", "pobj", "attr"}]
            s = self._entity_for_token(subs[0], ents) if subs else None
            o = self._entity_for_token(objs[0], ents) if objs else None

            # Some scientific parser configurations expose the predicate but do
            # not assign usable subject/object dependencies. Recover the relation
            # only when entity spans provide an unambiguous local subject/object.
            if s is None or o is None:
                fs, fo = self._fallback_entities(tok, ents)
                if fs is not None and fo is not None:
                    s, o = fs, fo

            if s is not None and o is not None:
                key = (s.start_char, s.end_char, pred, o.start_char, o.end_char)
                if key not in seen:
                    out.append(RelationResult(*key))
                    seen.add(key)
        return out

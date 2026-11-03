from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class EntityResult:
    text: str
    label: str
    start: int
    end: int

class ScientificNER:
    """Thin wrapper around a configured SciSpaCy pipeline."""
    def __init__(self, model_name: str = "en_core_sci_sm", nlp=None):
        self.model_name = model_name
        self._nlp = nlp

    def load(self):
        import spacy
        try:
            self._nlp = spacy.load(self.model_name)
        except OSError as exc:
            install_cmd = (
                "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
                if self.model_name == "en_core_sci_sm"
                else f"python -m spacy download {self.model_name} or pip install <model_url>"
            )
            raise RuntimeError(
                f"SciSpaCy model '{self.model_name}' is not installed. "
                f"Install the configured model before running Phase 2.\n"
                f"To install for this environment, run:\n"
                f"  {install_cmd}"
            ) from exc
        return self

    @property
    def nlp(self):
        return self._nlp if self._nlp is not None else self.load()._nlp

    def extract(self, text: str) -> list[EntityResult]:
        doc = self.nlp(text)
        return [EntityResult(e.text, e.label_, e.start_char, e.end_char) for e in doc.ents]

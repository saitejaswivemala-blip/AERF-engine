from __future__ import annotations
import numpy as np

class ClaimEmbedder:
    """Sentence-Transformer encoder for persisted claim embeddings."""
    def __init__(self, model_name="all-MiniLM-L6-v2", model=None):
        self.model_name = model_name
        self._model = model

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        vectors = self._load().encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return [np.asarray(v, dtype=np.float32) for v in vectors]

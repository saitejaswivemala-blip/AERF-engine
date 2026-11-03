from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class NLIResult:
    label: str
    entailment: float
    contradiction: float
    neutral: float
    confidence: float
    model_name: str


class NLIClassifier:
    """Real NLI classifier with robust label handling and bidirectional aggregation.

    The default Phase-3 model is DeBERTa-v3 MNLI/FEVER/ANLI rather than the
    smaller DistilBART model used by the original implementation.  The model
    name remains configurable so experiments can swap models without changing
    the pipeline.
    """

    DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    def __init__(self, model_name: str | None = None, classifier=None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._classifier = classifier

    def _load(self):
        if self._classifier is None:
            import os
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from transformers import pipeline
            self._classifier = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
            )
        return self._classifier

    @staticmethod
    def _normalise(raw):
        """Normalize Hugging Face output across model label conventions."""
        scores = raw[0] if raw and isinstance(raw[0], list) else raw
        vals = {str(x["label"]).lower(): float(x["score"]) for x in scores}

        # Prefer the model's explicit semantic labels.  Fall back to the
        # standard MNLI LABEL_0/1/2 convention used by many checkpoints.
        mapping = {
            "label_0": "contradiction",
            "label_1": "neutral",
            "label_2": "entailment",
        }
        normalized = {mapping.get(k, k): v for k, v in vals.items()}

        # Some checkpoints use capitalized labels or alternative entailment
        # spellings.  Keep this narrow and deterministic.
        aliases = {
            "entail": "entailment",
            "entails": "entailment",
            "contradict": "contradiction",
            "contradicts": "contradiction",
        }
        normalized = {aliases.get(k, k): v for k, v in normalized.items()}

        e = normalized.get("entailment", 0.0)
        c = normalized.get("contradiction", 0.0)
        n = normalized.get("neutral", 0.0)
        label = max(((e, "entailment"), (c, "contradiction"), (n, "neutral")))[1]
        return NLIResult(label, e, c, n, max(e, c, n), "unknown")

    def predict(self, premise, hypothesis):
        raw = self._load()(premise, text_pair=hypothesis, top_k=None)
        result = self._normalise(raw)
        return NLIResult(
            result.label,
            result.entailment,
            result.contradiction,
            result.neutral,
            result.confidence,
            self.model_name,
        )

    def predict_bidirectional(self, a, b):
        """Run NLI in both directions and combine the probabilities safely.

        Entailment is directional: if either claim entails the other, that is
        useful evidence for a support relationship.  Contradiction is treated
        as symmetric and therefore averaged across both directions.  Neutral
        is retained as the fallback when neither directional signal is strong.
        """
        x = self.predict(a, b)
        y = self.predict(b, a)

        contradiction = (x.contradiction + y.contradiction) / 2.0
        # Support is directional, so do not average it away when one claim is
        # a specific statement and the other is a more general statement.
        entailment = max(x.entailment, y.entailment)
        neutral = (x.neutral + y.neutral) / 2.0

        label = max(
            ((entailment, "entailment"),
             (contradiction, "contradiction"),
             (neutral, "neutral")),
            key=lambda item: item[0],
        )[1]
        confidence = max(entailment, contradiction, neutral)
        return NLIResult(
            label,
            entailment,
            contradiction,
            neutral,
            confidence,
            self.model_name,
        )

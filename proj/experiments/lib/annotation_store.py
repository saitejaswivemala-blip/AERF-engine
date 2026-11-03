"""
Human-annotation format and storage for cross-document claim-pair ground
truth.

Design rule (non-negotiable per the experiment brief): a system prediction
must NEVER silently become ground truth. `AnnotationRecord.human_label`
starts as None and is only set through `annotate()`, which requires an
explicit label. `system_prediction` is stored purely for later comparison
in metrics.evaluate() and is never read by `annotate()`.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

VALID_LABELS = ("supports", "contradicts", "neutral", "uncertain")


@dataclass
class AnnotationRecord:
    pair_id: str
    paper_a: str
    paper_b: str
    claim_a: str
    claim_b: str
    source_span_a: Optional[dict] = None  # {chunk_id, page, char_start, char_end}
    source_span_b: Optional[dict] = None
    system_prediction: Optional[str] = None
    human_label: Optional[str] = None
    human_notes: str = ""
    annotator: Optional[str] = None

    def annotate(self, label: str, annotator: str, notes: str = "") -> None:
        if label not in VALID_LABELS:
            raise ValueError(f"label must be one of {VALID_LABELS}, got {label!r}")
        self.human_label = label
        self.annotator = annotator
        self.human_notes = notes

    def is_annotated(self) -> bool:
        return self.human_label is not None


class AnnotationStore:
    """Append-only JSON-backed store of AnnotationRecords for one experiment run."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._records: dict[str, AnnotationRecord] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self.path.read_text())
        for row in data:
            self._records[row["pair_id"]] = AnnotationRecord(**row)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(r) for r in self._records.values()], indent=2))

    def add(self, record: AnnotationRecord, overwrite: bool = False) -> None:
        if record.pair_id in self._records and not overwrite:
            raise ValueError(f"pair_id {record.pair_id!r} already exists; pass overwrite=True to replace")
        self._records[record.pair_id] = record

    def get(self, pair_id: str) -> AnnotationRecord:
        return self._records[pair_id]

    def all(self) -> list[AnnotationRecord]:
        return list(self._records.values())

    def annotated(self) -> list[AnnotationRecord]:
        return [r for r in self._records.values() if r.is_annotated()]

    def unannotated(self) -> list[AnnotationRecord]:
        return [r for r in self._records.values() if not r.is_annotated()]

    def to_metric_records(self) -> list[dict]:
        """Only annotated records, in the shape metrics.evaluate() expects.
        Raises if any annotated record lacks a system_prediction, since a
        pair that was never scored by the system can't be evaluated."""
        out = []
        for r in self.annotated():
            if r.system_prediction is None:
                raise ValueError(f"pair_id {r.pair_id!r} has a human_label but no "
                                  f"system_prediction; run the system on it before evaluating")
            out.append({"system_prediction": r.system_prediction, "human_label": r.human_label})
        return out

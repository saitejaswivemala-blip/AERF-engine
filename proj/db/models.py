"""
models.py

Plain dataclasses mirroring the Phase 1 database rows (`papers`, `chunks`).
These are read/write DTOs used by repository.py and the pipeline -- they
intentionally have no behavior of their own, so business logic stays out
of the data layer.

Only Paper and Chunk are defined here in Phase 1. Dataclasses for
entities/claims/relations/etc. will be added in db/models.py during
Phase 2+ without needing to change these.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Paper:
    source_filename: str
    source_path: str
    page_count: int
    title: str | None = None
    authors: str | None = None
    publication_year: int | None = None
    venue: str | None = None
    paper_id: int | None = None          # set after insert
    ingested_at: datetime | str | None = None


@dataclass
class Chunk:
    paper_id: int
    page_number: int
    char_start: int
    char_end: int
    text: str
    chunk_id: int | None = None          # set after insert
    created_at: datetime | str | None = None

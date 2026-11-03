"""
exceptions.py

Custom exception types for the ingestion pipeline. Using typed exceptions
(instead of letting raw ValueError/OSError propagate everywhere) lets
callers -- and the orchestrator's error handling -- distinguish "this PDF
is unreadable" from "this database write failed" from "this config is
broken", and makes unit tests assert on the right thing.
"""


class AERFError(Exception):
    """Base class for all AERF-engine-specific errors."""


class PDFLoadError(AERFError):
    """Raised when a PDF file cannot be opened or has no extractable text."""


class TextCleaningError(AERFError):
    """Raised when text cleaning fails on malformed input."""


class ChunkingError(AERFError):
    """Raised when chunk segmentation fails or produces an invalid state."""


class DatabaseError(AERFError):
    """Raised when a database read/write operation fails."""

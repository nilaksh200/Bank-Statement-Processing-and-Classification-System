"""Ingestion module: Handles file validation, secure staging, and PDF type detection."""

from .detector import (
    PDFTypeDetector,
    PDFDetectionError,
    PDFPasswordProtectedError,
    PDFCorruptedError,
    PDFEmptyError,
)

__all__ = [
    "PDFTypeDetector",
    "PDFDetectionError",
    "PDFPasswordProtectedError",
    "PDFCorruptedError",
    "PDFEmptyError",
]

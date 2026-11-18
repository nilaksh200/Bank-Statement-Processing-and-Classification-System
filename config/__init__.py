"""Configuration package for Bank Statement Processing & Classification System."""

from .settings import (
    BASE_DIR,
    DATA_DIR,
    TEMP_DIR,
    EXPORTS_DIR,
    MODELS_DIR,
    SUPPORTED_FILE_EXTENSIONS,
    SUPPORTED_MIME_TYPES,
    MAX_FILE_SIZE_MB,
    STANDARD_SCHEMA_COLUMNS,
    OCR_SETTINGS,
    DEFAULT_CATEGORIES,
)

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "TEMP_DIR",
    "EXPORTS_DIR",
    "MODELS_DIR",
    "SUPPORTED_FILE_EXTENSIONS",
    "SUPPORTED_MIME_TYPES",
    "MAX_FILE_SIZE_MB",
    "STANDARD_SCHEMA_COLUMNS",
    "OCR_SETTINGS",
    "DEFAULT_CATEGORIES",
]


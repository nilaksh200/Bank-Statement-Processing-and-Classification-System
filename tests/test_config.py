"""Unit tests for system configuration and schema constants."""

from config.settings import (
    BASE_DIR,
    DATA_DIR,
    TEMP_DIR,
    EXPORTS_DIR,
    STANDARD_SCHEMA_COLUMNS,
    SUPPORTED_FILE_EXTENSIONS,
    OCR_SETTINGS,
    FORBIDDEN_PACKAGES,
)


def test_standard_schema_columns():
    expected = ["date", "description", "debit", "credit", "balance", "category"]
    assert STANDARD_SCHEMA_COLUMNS == expected


def test_runtime_directories_exist():
    assert DATA_DIR.exists()
    assert TEMP_DIR.exists()
    assert EXPORTS_DIR.exists()


def test_supported_extensions():
    assert ".pdf" in SUPPORTED_FILE_EXTENSIONS


def test_ocr_settings_structure():
    assert "enabled" in OCR_SETTINGS
    assert "default_engine" in OCR_SETTINGS
    assert OCR_SETTINGS["render_dpi"] == 300


def test_strict_non_llm_definitions():
    assert "openai" in FORBIDDEN_PACKAGES
    assert "langchain" in FORBIDDEN_PACKAGES


"""Unit tests for ExtractionOrchestrator and Data Sanitiser."""

from pathlib import Path
import pytest
import pandas as pd

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore

from extraction.sanitiser import clean_date, clean_amount, sanitise_dataframe
from extraction.orchestrator import ExtractionOrchestrator
from tests.test_extraction import create_statement_with_metadata, create_grid_table_pdf
from tests.test_detector import create_corrupted_pdf_file


@pytest.fixture
def orchestrator() -> ExtractionOrchestrator:
    return ExtractionOrchestrator()


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


# ==============================================================================
# Sanitiser Unit Tests
# ==============================================================================
def test_clean_date_formats():
    assert clean_date("15/01/2026") == "2026-01-15"
    assert clean_date("2026-01-15") == "2026-01-15"
    assert clean_date("15-Jan-2026") == "2026-01-15"
    assert clean_date("Jan 15, 2026") == "2026-01-15"
    assert clean_date("15.01.2026") == "2026-01-15"
    assert clean_date("") == ""
    assert clean_date("-") == ""
    assert clean_date(None) == ""


def test_clean_amount_currencies():
    assert clean_amount("$1,234.56") == 1234.56
    assert clean_amount("₹ 5,000.00") == 5000.00
    assert clean_amount("€ 120.50") == 120.50
    assert clean_amount("£45.99") == 45.99
    assert clean_amount("(150.00)") == 150.00
    assert clean_amount("- 75.25") == 75.25
    assert clean_amount("") == 0.0
    assert clean_amount("-") == 0.0
    assert clean_amount(None) == 0.0
    assert clean_amount(1500) == 1500.0


def test_sanitise_dataframe():
    raw_df = pd.DataFrame([
        {"date": "15/01/2026", "description": "  GROCERY STORE \n", "debit": "$45.50", "credit": "-", "balance": "$1,000.00"},
        {"date": "16-Jan-2026", "description": "SALARY   DEP", "debit": "0.0", "credit": "₹2,500.00", "balance": "3,500.00"},
        {"date": "", "description": "", "debit": "0.0", "credit": "0.0", "balance": "0.0"},  # empty noise row
    ])

    clean_df = sanitise_dataframe(raw_df)
    assert len(clean_df) == 2
    assert clean_df.iloc[0]["date"] == "2026-01-15"
    assert clean_df.iloc[0]["description"] == "GROCERY STORE"
    assert clean_df.iloc[0]["debit"] == 45.50
    assert clean_df.iloc[0]["credit"] == 0.0
    assert clean_df.iloc[0]["balance"] == 1000.00

    assert clean_df.iloc[1]["date"] == "2026-01-16"
    assert clean_df.iloc[1]["description"] == "SALARY DEP"
    assert clean_df.iloc[1]["credit"] == 2500.00


# ==============================================================================
# Orchestrator Unit Tests
# ==============================================================================
def test_orchestrator_process_text_statement(orchestrator: ExtractionOrchestrator, temp_dir: Path):
    pdf_path = create_grid_table_pdf(temp_dir / "grid_statement.pdf")
    result = orchestrator.process_statement(pdf_path)

    assert result["status"] == "success"
    assert result["pdf_type"] == "text_based"
    assert isinstance(result["metadata"], dict)
    assert isinstance(result["transactions"], pd.DataFrame)
    assert len(result["transactions"]) == 5
    assert list(result["transactions"].columns) == ["date", "description", "debit", "credit", "balance"]
    assert result["transactions"].iloc[0]["date"] == "2026-01-02"


def test_orchestrator_file_not_found(orchestrator: ExtractionOrchestrator, temp_dir: Path):
    non_existent = temp_dir / "missing.pdf"
    result = orchestrator.process_statement(non_existent)

    assert result["status"] == "error"
    assert result["pdf_type"] == "unknown"
    assert "not found" in result["error_message"].lower()


def test_orchestrator_corrupted_file(orchestrator: ExtractionOrchestrator, temp_dir: Path):
    corrupt_pdf = create_corrupted_pdf_file(temp_dir / "broken.pdf")
    result = orchestrator.process_statement(corrupt_pdf)

    assert result["status"] == "error"
    assert result["pdf_type"] == "unknown"
    assert "error_message" in result


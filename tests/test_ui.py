"""Unit tests for Streamlit application components and processing functions."""

from pathlib import Path
import pytest
import pandas as pd

from app import process_uploaded_pdf, get_classification_engine
from tests.test_extraction import create_rich_statement_pdf


def test_get_classification_engine():
    """Verify that the cached classification engine initializes properly."""
    engine = get_classification_engine()
    assert engine is not None
    cat, conf, method = engine.classify_single("WALMART GROCERY")
    assert cat == "Groceries & Supermarket"
    assert conf > 0


def test_process_uploaded_pdf_e2e(tmp_path: Path):
    """Verify end-to-end processing helper function used by Streamlit UI."""
    pdf_path = create_rich_statement_pdf(tmp_path / "test_statement.pdf")
    pdf_bytes = pdf_path.read_bytes()

    result = process_uploaded_pdf(
        file_bytes=pdf_bytes,
        filename="test_statement.pdf",
        ocr_engine="EasyOCR",
    )

    assert result["status"] == "success"
    assert result["pdf_type"] == "text_based"
    assert result["page_count"] == 1
    assert isinstance(result["metadata"], dict)
    assert result["metadata"]["account_holder"] == "Alice M. Walker"

    df = result["transactions"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "category" in df.columns
    assert "confidence" in df.columns
    assert "classification_method" in df.columns


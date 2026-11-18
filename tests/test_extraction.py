"""Comprehensive Unit Test Suite for Extraction Pipeline and Data Sanitisation.

Tests:
1. `clean_date` and `clean_amount` edge cases (commas, currency symbols, missing values, unusual date representations).
2. `ExtractionOrchestrator` on synthetic text PDFs verifying accurate column extraction and metadata parsing.
3. Rigorous validation of DataFrame schema contract, non-null dates in ISO format, and numerical float integrity.
"""

from pathlib import Path
import pytest
import pandas as pd

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore

from extraction.sanitiser import clean_date, clean_amount, sanitise_dataframe
from extraction.text_parser import TextStatementParser, TRANSACTION_COLUMNS
from extraction.orchestrator import ExtractionOrchestrator
from tests.test_detector import create_corrupted_pdf_file


@pytest.fixture
def orchestrator() -> ExtractionOrchestrator:
    return ExtractionOrchestrator()


@pytest.fixture
def parser() -> TextStatementParser:
    return TextStatementParser()


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


# ==============================================================================
# 1. Synthetic PDF Generators
# ==============================================================================
def create_rich_statement_pdf(path: Path) -> Path:
    """Generate a realistic bank statement with header metadata and grid transaction table."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4

    # Account Metadata Header
    page.insert_text((50, 45), "GLOBAL TRUST BANK - ACCOUNT STATEMENT", fontsize=15)
    page.insert_text((50, 68), "Account Holder Name: Alice M. Walker", fontsize=11)
    page.insert_text((50, 83), "Account Number: 5544-3322-1100", fontsize=11)
    page.insert_text((50, 98), "IFSC Code: GTBK0004589", fontsize=11)
    page.insert_text((50, 113), "Statement Period: 01/01/2026 to 31/01/2026", fontsize=11)

    # Grid Table Coordinates
    col_x = [50, 130, 340, 410, 480, 550]
    y_start = 145
    row_height = 24

    table_data = [
        ["Date", "Description", "Debit", "Credit", "Balance"],
        ["02/01/2026", "OFFICE DEPOT SUPPLIES", "$125.50", "$0.00", "$4,874.50"],
        ["05-Jan-2026", "CLIENT INVOICE #8842 PAYMENT", "$0.00", "$2,500.00", "$7,374.50"],
        ["10/01/2026", "STARBUCKS STORE #129", "$12.75", "$0.00", "$7,361.75"],
        ["18-Jan-2026", "ELECTRIC UTILITY AUTO-PAY", "$142.30", "$0.00", "$7,219.45"],
        ["25/01/2026", "CONSULTING FEE RETAINER", "$0.00", "$1,800.00", "$9,019.45"],
    ]

    for r_idx, row in enumerate(table_data):
        y_top = y_start + (r_idx * row_height)
        y_bot = y_top + row_height
        page.draw_rect(fitz.Rect(col_x[0], y_top, col_x[-1], y_bot), color=(0, 0, 0), width=0.8)
        for x in col_x[1:-1]:
            page.draw_line(fitz.Point(x, y_top), fitz.Point(x, y_bot), color=(0, 0, 0), width=0.8)

        for c_idx, cell in enumerate(row):
            page.insert_text((col_x[c_idx] + 4, y_top + 16), cell, fontsize=9)

    doc.save(str(path))
    doc.close()
    return path


# Aliases for backward compatibility across test fixtures
create_statement_with_metadata = create_rich_statement_pdf
create_grid_table_pdf = create_rich_statement_pdf


def create_multiline_statement_pdf(path: Path) -> Path:
    """Generate a table with description wrapping over multiple rows."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    col_x = [50, 130, 340, 410, 480, 550]
    y_start = 100
    row_height = 24

    table_data = [
        ["Date", "Description", "Debit", "Credit", "Balance"],
        ["2026-01-03", "AMAZON RETAIL PURCHASE", "89.90", "0.00", "2,110.10"],
        ["", "ORDER #114-992384 SEATTLE WA", "", "", ""],  # Wrapped text
        ["2026-01-08", "SALARY PAYMENT ACME CORP", "0.00", "3,500.00", "5,610.10"],
        ["2026-01-15", "VERIZON WIRELESS BILL", "110.00", "0.00", "5,500.10"],
        ["", "AUTO DEBIT MONTHLY PLAN", "", "", ""],       # Wrapped text
    ]

    for r_idx, row in enumerate(table_data):
        y_top = y_start + (r_idx * row_height)
        y_bot = y_top + row_height
        page.draw_rect(fitz.Rect(col_x[0], y_top, col_x[-1], y_bot), color=(0, 0, 0), width=0.8)
        for x in col_x[1:-1]:
            page.draw_line(fitz.Point(x, y_top), fitz.Point(x, y_bot), color=(0, 0, 0), width=0.8)

        for c_idx, cell in enumerate(row):
            if cell:
                page.insert_text((col_x[c_idx] + 4, y_top + 16), cell, fontsize=9)

    doc.save(str(path))
    doc.close()
    return path


# Aliases for backward compatibility across test fixtures
create_statement_with_metadata = create_rich_statement_pdf
create_grid_table_pdf = create_rich_statement_pdf


def create_single_amount_statement_pdf(path: Path) -> Path:
    """Generate a table with a single Amount column using DR/CR/+/-."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    col_x = [50, 140, 360, 460, 550]
    y_start = 100
    row_height = 24

    table_data = [
        ["Date", "Description", "Amount", "Balance"],
        ["2026-01-02", "GROCERY OUTLET", "65.50 DR", "3,934.50"],
        ["2026-01-05", "PAYROLL DIRECT DEP", "1,800.00 CR", "5,734.50"],
        ["2026-01-12", "RESTAURANT DINING", "-42.00", "5,692.50"],
        ["2026-01-18", "CASHBACK BONUS", "+25.00", "5,717.50"],
    ]

    for r_idx, row in enumerate(table_data):
        y_top = y_start + (r_idx * row_height)
        y_bot = y_top + row_height
        page.draw_rect(fitz.Rect(col_x[0], y_top, col_x[-1], y_bot), color=(0, 0, 0), width=0.8)
        for x in col_x[1:-1]:
            page.draw_line(fitz.Point(x, y_top), fitz.Point(x, y_bot), color=(0, 0, 0), width=0.8)

        for c_idx, cell in enumerate(row):
            page.insert_text((col_x[c_idx] + 4, y_top + 16), cell, fontsize=9)

    doc.save(str(path))
    doc.close()
    return path


# Aliases for backward compatibility across test fixtures
create_statement_with_metadata = create_rich_statement_pdf
create_grid_table_pdf = create_rich_statement_pdf


# ==============================================================================
# 2. Mock Tests for clean_date
# ==============================================================================
def test_clean_date_edge_cases():
    """Verify clean_date handles various date representations, unusual formats, and missing values."""
    # Standard ISO and European representations
    assert clean_date("2026-01-15") == "2026-01-15"
    assert clean_date("15/01/2026") == "2026-01-15"
    assert clean_date("15-01-2026") == "2026-01-15"
    assert clean_date("15.01.2026") == "2026-01-15"

    # Month strings & abbreviations
    assert clean_date("15-Jan-2026") == "2026-01-15"
    assert clean_date("15 January 2026") == "2026-01-15"
    assert clean_date("Jan 15, 2026") == "2026-01-15"
    assert clean_date("January 15 2026") == "2026-01-15"

    # Two-digit year resolution
    assert clean_date("15/01/26") == "2026-01-15"

    # Whitespace and newline padding
    assert clean_date("   2026-01-15 \n\t") == "2026-01-15"
    assert clean_date("  15-Jan-2026  ") == "2026-01-15"

    # Missing, empty, and non-date inputs
    assert clean_date("") == ""
    assert clean_date("   ") == ""
    assert clean_date("-") == ""
    assert clean_date("--") == ""
    assert clean_date("None") == ""
    assert clean_date(None) == ""
    assert clean_date("N/A") == ""
    assert clean_date("nan") == ""
    assert clean_date(float("nan")) == ""


# ==============================================================================
# 3. Mock Tests for clean_amount
# ==============================================================================
def test_clean_amount_edge_cases():
    """Verify clean_amount handles currency symbols, commas, accounting notation, and nulls."""
    # Commas and thousands separators
    assert clean_amount("1,234.56") == 1234.56
    assert clean_amount("1,000,500.75") == 1000500.75
    assert clean_amount("12,345") == 12345.0

    # Multi-currency symbols
    assert clean_amount("$1,234.56") == 1234.56
    assert clean_amount("₹ 5,000.00") == 5000.00
    assert clean_amount("€ 120.50") == 120.50
    assert clean_amount("£45.99") == 45.99
    assert clean_amount("¥ 10,000") == 10000.0
    assert clean_amount("USD 250.00") == 250.00
    assert clean_amount("CAD 75.25") == 75.25

    # Negative amounts and accounting parentheses
    assert clean_amount("-75.25") == 75.25
    assert clean_amount("(150.00)") == 150.00
    assert clean_amount("-$ 320.00") == 320.00

    # Whitespace handling
    assert clean_amount("   $ 1,500.25 \n") == 1500.25

    # Direct numeric types
    assert clean_amount(150) == 150.0
    assert clean_amount(125.75) == 125.75

    # Empty and missing values
    assert clean_amount("") == 0.0
    assert clean_amount("   ") == 0.0
    assert clean_amount("-") == 0.0
    assert clean_amount("--") == 0.0
    assert clean_amount("None") == 0.0
    assert clean_amount(None) == 0.0
    assert clean_amount("N/A") == 0.0
    assert clean_amount("nil") == 0.0
    assert clean_amount(float("nan")) == 0.0


# ==============================================================================
# 4. Orchestrator End-to-End Extraction & DataFrame Assertion
# ==============================================================================
def test_orchestrator_on_synthetic_text_pdf(orchestrator: ExtractionOrchestrator, temp_dir: Path):
    """Test ExtractionOrchestrator on synthetic text PDF and assert strict schema and type contracts."""
    pdf_path = create_rich_statement_pdf(temp_dir / "statement.pdf")
    result = orchestrator.process_statement(pdf_path)

    # 1. Output Dictionary Structure
    assert result["status"] == "success"
    assert result["pdf_type"] == "text_based"
    assert isinstance(result["metadata"], dict)
    assert isinstance(result["transactions"], pd.DataFrame)

    # 2. Metadata Verification
    metadata = result["metadata"]
    assert metadata["account_holder"] == "Alice M. Walker"
    assert metadata["account_number"] == "5544-3322-1100"
    assert metadata["routing_code"] == "GTBK0004589"
    assert "01/01/2026" in str(metadata["statement_period"])

    df = result["transactions"]

    # 3. Exact Column Names Contract
    assert list(df.columns) == ["date", "description", "debit", "credit", "balance"]
    assert len(df) == 5

    # 4. Non-Null, ISO Formatted Dates
    assert df["date"].isna().sum() == 0
    assert (df["date"] == "").sum() == 0
    for d in df["date"]:
        assert len(d) == 10
        assert d[4] == "-" and d[7] == "-"
        # Check standard ISO YYYY-MM-DD
        year, month, day = d.split("-")
        assert year == "2026"
        assert 1 <= int(month) <= 12
        assert 1 <= int(day) <= 31

    # 5. Non-Empty Descriptions
    assert df["description"].isna().sum() == 0
    assert (df["description"] == "").sum() == 0
    assert all(isinstance(desc, str) for desc in df["description"])

    # 6. Valid Numerical Float Values for Debit, Credit, and Balance
    for col in ["debit", "credit", "balance"]:
        assert pd.api.types.is_float_dtype(df[col])
        assert df[col].isna().sum() == 0
        assert (df[col] < 0).sum() == 0  # Absolute non-negative floats

    # 7. Check Numerical Accuracy of Extracted Rows
    row0 = df.iloc[0]
    assert row0["date"] == "2026-01-02"
    assert "OFFICE DEPOT SUPPLIES" in row0["description"]
    assert row0["debit"] == 125.50
    assert row0["credit"] == 0.0
    assert row0["balance"] == 4874.50

    row1 = df.iloc[1]
    assert row1["date"] == "2026-01-05"
    assert "CLIENT INVOICE" in row1["description"]
    assert row1["debit"] == 0.0
    assert row1["credit"] == 2500.00
    assert row1["balance"] == 7374.50


# ==============================================================================
# 5. Multiline & Single Amount Table Extraction Tests
# ==============================================================================
def test_multiline_narrative_merging(parser: TextStatementParser, temp_dir: Path):
    """Verify wrapped multiline narratives are concatenated onto the primary date row."""
    pdf_path = create_multiline_statement_pdf(temp_dir / "multiline.pdf")
    df = parser.extract_transactions(pdf_path)

    assert len(df) == 3
    assert "AMAZON RETAIL PURCHASE ORDER #114-992384 SEATTLE WA" in df.iloc[0]["description"]
    assert "VERIZON WIRELESS BILL AUTO DEBIT MONTHLY PLAN" in df.iloc[2]["description"]


def test_single_amount_splitting(parser: TextStatementParser, temp_dir: Path):
    """Verify single amount column values with DR/CR/+/- are split into debit and credit."""
    pdf_path = create_single_amount_statement_pdf(temp_dir / "single_amt.pdf")
    df = parser.extract_transactions(pdf_path)

    assert len(df) == 4
    # Row 0: 65.50 DR
    assert df.iloc[0]["debit"] == 65.50
    assert df.iloc[0]["credit"] == 0.0
    # Row 1: 1800.00 CR
    assert df.iloc[1]["debit"] == 0.0
    assert df.iloc[1]["credit"] == 1800.00
    # Row 2: -42.00
    assert df.iloc[2]["debit"] == 42.00
    assert df.iloc[2]["credit"] == 0.0
    # Row 3: +25.00
    assert df.iloc[3]["debit"] == 0.0
    assert df.iloc[3]["credit"] == 25.00


def test_orchestrator_error_handling(orchestrator: ExtractionOrchestrator, temp_dir: Path):
    """Verify orchestrator returns error status for corrupted or nonexistent files."""
    missing_path = temp_dir / "does_not_exist_file.pdf"
    res1 = orchestrator.process_statement(missing_path)
    assert res1["status"] == "error"
    assert res1["pdf_type"] == "unknown"

    corrupted_path = create_corrupted_pdf_file(temp_dir / "corrupted_file.pdf")
    res2 = orchestrator.process_statement(corrupted_path)
    assert res2["status"] == "error"
    assert res2["pdf_type"] == "unknown"

"""Unit tests for StatementExporter (CSV and styled multi-tab Excel)."""

from pathlib import Path
import pytest
import openpyxl
import pandas as pd

from export.exporter import StatementExporter, COLOR_NAVY, COLOR_CREDIT_GREEN, COLOR_DEBIT_RED


@pytest.fixture
def exporter() -> StatementExporter:
    return StatementExporter(currency_symbol="$")


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, dict]:
    df = pd.DataFrame([
        {
            "date": "2026-01-02",
            "description": "WALMART GROCERY MART",
            "debit": 85.50,
            "credit": 0.00,
            "balance": 4914.50,
            "category": "Groceries & Supermarket",
        },
        {
            "date": "2026-01-05",
            "description": "CITY ELECTRICITY BILL AUTO-PAY",
            "debit": 142.20,
            "credit": 0.00,
            "balance": 4772.30,
            "category": "Utilities & Bills",
        },
        {
            "date": "2026-01-10",
            "description": "SALARY DIRECT DEPOSIT ACME CORP",
            "debit": 0.00,
            "credit": 4500.00,
            "balance": 9272.30,
            "category": "Salary / Income",
        },
    ])
    metadata = {
        "account_holder": "Alice M. Walker",
        "account_number": "5544-3322-1100",
        "routing_code": "GTBK0004589",
        "statement_period": "01/01/2026 to 31/01/2026",
    }
    return df, metadata


# ==============================================================================
# 1. CSV Export Tests
# ==============================================================================
def test_to_csv_export(exporter: StatementExporter, sample_data, temp_dir: Path):
    df, _ = sample_data
    csv_file = temp_dir / "transactions.csv"

    out_path = exporter.to_csv(df, csv_file)
    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 0

    # Read back and verify rows and columns
    read_df = pd.read_csv(out_path)
    assert len(read_df) == 3
    assert list(read_df.columns) == list(df.columns)
    assert read_df.iloc[0]["description"] == "WALMART GROCERY MART"


# ==============================================================================
# 2. Excel Multi-Tab & Styling Tests
# ==============================================================================
def test_to_excel_multitab_structure(exporter: StatementExporter, sample_data, temp_dir: Path):
    df, meta = sample_data
    xlsx_file = temp_dir / "statement_report.xlsx"

    out_path = exporter.to_excel(df, meta, xlsx_file)
    assert Path(out_path).exists()

    wb = openpyxl.load_workbook(out_path)
    assert wb.sheetnames == ["Transactions", "Category Summary", "Account Metadata"]


def test_transactions_sheet_styling(exporter: StatementExporter, sample_data, temp_dir: Path):
    df, meta = sample_data
    xlsx_file = temp_dir / "transactions_styled.xlsx"
    exporter.to_excel(df, meta, xlsx_file)

    wb = openpyxl.load_workbook(xlsx_file)
    ws = wb["Transactions"]

    # 1. Header Styling
    header_cell = ws.cell(row=1, column=1)
    assert header_cell.font.bold is True
    assert header_cell.font.color.rgb == "00FFFFFF" or header_cell.font.color.rgb == "FFFFFF"
    # Navy header fill
    assert header_cell.fill.start_color.rgb.endswith(COLOR_NAVY)

    # 2. Number format verification for currency
    debit_cell = ws.cell(row=2, column=3)  # Row 2 (first transaction), Column 3 (Debit)
    assert "$" in debit_cell.number_format
    assert debit_cell.value == 85.50

    # 3. Highlight colors: Row 2 has debit > 0 -> Light Red fill
    assert debit_cell.fill.start_color.rgb.endswith(COLOR_DEBIT_RED)

    # Row 4 (third transaction) has credit = 4500.00 -> Light Green fill
    credit_cell = ws.cell(row=4, column=4)
    assert credit_cell.fill.start_color.rgb.endswith(COLOR_CREDIT_GREEN)


def test_category_summary_sheet_content(exporter: StatementExporter, sample_data, temp_dir: Path):
    df, meta = sample_data
    xlsx_file = temp_dir / "category_summary.xlsx"
    exporter.to_excel(df, meta, xlsx_file)

    wb = openpyxl.load_workbook(xlsx_file)
    ws = wb["Category Summary"]

    # Title Banner
    assert "Category Spend" in ws.cell(row=1, column=1).value

    # Check that categories exist in table
    categories_found = [ws.cell(row=r, column=1).value for r in range(3, 10) if ws.cell(row=r, column=1).value]
    assert "Groceries & Supermarket" in categories_found
    assert "Utilities & Bills" in categories_found
    assert "Total" in categories_found


def test_account_metadata_sheet_content(exporter: StatementExporter, sample_data, temp_dir: Path):
    df, meta = sample_data
    xlsx_file = temp_dir / "metadata_test.xlsx"
    exporter.to_excel(df, meta, xlsx_file)

    wb = openpyxl.load_workbook(xlsx_file)
    ws = wb["Account Metadata"]

    sheet_text = [str(ws.cell(row=r, column=c).value) for r in range(1, 15) for c in (1, 2)]
    joined = " ".join(sheet_text)

    assert "Alice M. Walker" in joined
    assert "5544-3322-1100" in joined
    assert "GTBK0004589" in joined
    assert "01/01/2026 to 31/01/2026" in joined


def test_empty_dataframe_export(exporter: StatementExporter, temp_dir: Path):
    empty_df = pd.DataFrame(columns=["date", "description", "debit", "credit", "balance", "category"])
    xlsx_file = temp_dir / "empty.xlsx"

    out = exporter.to_excel(empty_df, {}, xlsx_file)
    assert Path(out).exists()
    wb = openpyxl.load_workbook(out)
    assert "Transactions" in wb.sheetnames


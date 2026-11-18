"""Integration and verification test suite for StatementExporter and UI export pipelines.

Tests:
1. `StatementExporter.to_csv()` and `StatementExporter.to_excel()` on a populated sample DataFrame.
2. Verification of all required worksheets ('Transactions', 'Category Summary', 'Account Metadata') via `openpyxl.load_workbook()`.
3. Strict verification of monetary numerical data types and currency formatting post-export.
"""

from pathlib import Path
import pytest
import openpyxl
import pandas as pd

from export.exporter import (
    StatementExporter,
    COLOR_NAVY,
    COLOR_CREDIT_GREEN,
    COLOR_DEBIT_RED,
)


@pytest.fixture
def exporter() -> StatementExporter:
    return StatementExporter(currency_symbol="$")


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """Populated DataFrame representing processed and classified transactions."""
    return pd.DataFrame([
        {
            "date": "2026-01-03",
            "description": "WALMART SUPERCENTER GROCERY",
            "debit": 124.50,
            "credit": 0.00,
            "balance": 4875.50,
            "category": "Groceries & Supermarket",
        },
        {
            "date": "2026-01-07",
            "description": "PACIFIC GAS & ELECTRIC BILL",
            "debit": 185.30,
            "credit": 0.00,
            "balance": 4690.20,
            "category": "Utilities & Bills",
        },
        {
            "date": "2026-01-12",
            "description": "STARBUCKS CAFE LATTE",
            "debit": 14.75,
            "credit": 0.00,
            "balance": 4675.45,
            "category": "Dining & Food",
        },
        {
            "date": "2026-01-15",
            "description": "MONTHLY PAYROLL DIRECT DEPOSIT",
            "debit": 0.00,
            "credit": 4500.00,
            "balance": 9175.45,
            "category": "Salary / Income",
        },
        {
            "date": "2026-01-20",
            "description": "ATM CASH WITHDRAWAL #441",
            "debit": 200.00,
            "credit": 0.00,
            "balance": 8975.45,
            "category": "ATM & Cash",
        },
    ])


@pytest.fixture
def sample_metadata() -> dict:
    """Sample bank statement header metadata."""
    return {
        "account_holder": "Robert C. Davis",
        "account_number": "9988-7766-5544",
        "routing_code": "021000021",
        "statement_period": "01/01/2026 to 31/01/2026",
    }


# ==============================================================================
# 1. CSV Exporter Integration Tests
# ==============================================================================
def test_csv_export_integrity(
    exporter: StatementExporter,
    sample_transactions: pd.DataFrame,
    tmp_path: Path,
):
    """Verify to_csv exports clean records with correct schemas and numerical types."""
    csv_file = tmp_path / "export_test.csv"
    exported_path = exporter.to_csv(sample_transactions, csv_file)

    assert Path(exported_path).exists()
    assert Path(exported_path).stat().st_size > 0

    # Read back and verify columns and types
    df_read = pd.read_csv(exported_path)
    assert len(df_read) == 5
    assert list(df_read.columns) == ["date", "description", "debit", "credit", "balance", "category"]

    # Verify numerical float types are preserved
    assert pd.api.types.is_float_dtype(df_read["debit"])
    assert pd.api.types.is_float_dtype(df_read["credit"])
    assert pd.api.types.is_float_dtype(df_read["balance"])

    assert df_read.iloc[0]["debit"] == 124.50
    assert df_read.iloc[3]["credit"] == 4500.00


# ==============================================================================
# 2. Excel Worksheet Structure Tests
# ==============================================================================
def test_excel_export_all_required_worksheets(
    exporter: StatementExporter,
    sample_transactions: pd.DataFrame,
    sample_metadata: dict,
    tmp_path: Path,
):
    """Assert that generated .xlsx contains all required worksheets using openpyxl."""
    excel_file = tmp_path / "multi_tab_statement.xlsx"
    out_path = exporter.to_excel(sample_transactions, sample_metadata, excel_file)

    assert Path(out_path).exists()

    # Load workbook using openpyxl
    wb = openpyxl.load_workbook(out_path)
    expected_sheets = ["Transactions", "Category Summary", "Account Metadata"]

    assert wb.sheetnames == expected_sheets
    for sheet_name in expected_sheets:
        assert sheet_name in wb


# ==============================================================================
# 3. Monetary Number Formatting and Type Preservation Tests
# ==============================================================================
def test_excel_monetary_formatting_and_data_types(
    exporter: StatementExporter,
    sample_transactions: pd.DataFrame,
    sample_metadata: dict,
    tmp_path: Path,
):
    """Verify that debit, credit, balance maintain numeric data types and currency formatting."""
    excel_file = tmp_path / "formatted_types.xlsx"
    exporter.to_excel(sample_transactions, sample_metadata, excel_file)

    wb = openpyxl.load_workbook(excel_file)
    ws_tx = wb["Transactions"]

    # Row 1 is header, Rows 2 to 6 are data rows
    for r_idx in range(2, 7):
        debit_cell = ws_tx.cell(row=r_idx, column=3)
        credit_cell = ws_tx.cell(row=r_idx, column=4)
        balance_cell = ws_tx.cell(row=r_idx, column=5)

        # 1. Type preservation: cell data_type is numeric ('n')
        assert debit_cell.data_type == "n", f"Row {r_idx} debit is not numeric"
        assert credit_cell.data_type == "n", f"Row {r_idx} credit is not numeric"
        assert balance_cell.data_type == "n", f"Row {r_idx} balance is not numeric"

        # 2. Values are python floats/ints, not strings
        assert isinstance(debit_cell.value, (float, int))
        assert isinstance(credit_cell.value, (float, int))
        assert isinstance(balance_cell.value, (float, int))

        # 3. Currency number formatting contains '$' and '#,##0.00'
        assert "$" in debit_cell.number_format
        assert "#,##0.00" in debit_cell.number_format
        assert "$" in credit_cell.number_format
        assert "$" in balance_cell.number_format

    # 4. Check specific cell highlight fills
    # Row 2 (debit = 124.50 > 0) -> soft red fill
    assert ws_tx.cell(row=2, column=3).fill.start_color.rgb.endswith(COLOR_DEBIT_RED)

    # Row 5 (credit = 4500.00 > 0) -> soft green fill
    assert ws_tx.cell(row=5, column=4).fill.start_color.rgb.endswith(COLOR_CREDIT_GREEN)


def test_excel_category_summary_pivot_verification(
    exporter: StatementExporter,
    sample_transactions: pd.DataFrame,
    sample_metadata: dict,
    tmp_path: Path,
):
    """Verify Category Summary pivot table sums, percentage formatting, and totals."""
    excel_file = tmp_path / "category_pivot.xlsx"
    exporter.to_excel(sample_transactions, sample_metadata, excel_file)

    wb = openpyxl.load_workbook(excel_file)
    ws_cat = wb["Category Summary"]

    # Header check
    assert ws_cat.cell(row=2, column=1).value == "Category"
    assert "Total Spend" in str(ws_cat.cell(row=2, column=2).value)
    assert "% of Total Spend" in str(ws_cat.cell(row=2, column=5).value)

    # Check Total Spend row at the bottom
    found_total = False
    for r in range(3, 12):
        label = ws_cat.cell(row=r, column=1).value
        if label == "Total":
            found_total = True
            total_spend = ws_cat.cell(row=r, column=2).value
            total_credit = ws_cat.cell(row=r, column=3).value
            total_count = ws_cat.cell(row=r, column=4).value
            pct_val = ws_cat.cell(row=r, column=5).value

            # Total debits = 124.50 + 185.30 + 14.75 + 200.00 = 524.55
            assert pytest.approx(total_spend, 0.01) == 524.55
            assert pytest.approx(total_credit, 0.01) == 4500.00
            assert total_count == 5
            assert pct_val == 1.0
            break

    assert found_total, "Summary 'Total' row not found in Category Summary tab"


def test_excel_metadata_tab_accuracy(
    exporter: StatementExporter,
    sample_transactions: pd.DataFrame,
    sample_metadata: dict,
    tmp_path: Path,
):
    """Verify Account Metadata tab records exact header metadata values."""
    excel_file = tmp_path / "metadata_verify.xlsx"
    exporter.to_excel(sample_transactions, sample_metadata, excel_file)

    wb = openpyxl.load_workbook(excel_file)
    ws_meta = wb["Account Metadata"]

    meta_dict = {}
    for r in range(3, 12):
        k = ws_meta.cell(row=r, column=1).value
        v = ws_meta.cell(row=r, column=2).value
        if k and v:
            meta_dict[k] = str(v)

    assert meta_dict.get("Account Holder Name") == "Robert C. Davis"
    assert meta_dict.get("Account Number") == "9988-7766-5544"
    assert meta_dict.get("IFSC / SWIFT / Routing Code") == "021000021"
    assert meta_dict.get("Statement Period") == "01/01/2026 to 31/01/2026"
    assert "5" in meta_dict.get("Total Transactions Processed", "")


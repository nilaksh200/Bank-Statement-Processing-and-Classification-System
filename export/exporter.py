"""Styled Excel and CSV Exporter Module for Bank Statements.

Builds formatted CSV files and multi-tab Excel workbooks containing:
- Tab 1: "Transactions" with navy headers, alternating row colors, currency formatting,
         and soft color highlighting for credits (light green) and debits (light red).
- Tab 2: "Category Summary" with spend breakdown, transaction counts, and percentage distribution.
- Tab 3: "Account Metadata" presenting account headers, period bounds, and processing statistics.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd

logger = logging.getLogger(__name__)

# Standard Styling Constants
COLOR_NAVY = "1F4E79"
COLOR_WHITE = "FFFFFF"
COLOR_ZEBRA_LIGHT = "F2F5F9"
COLOR_CREDIT_GREEN = "E2EFDA"
COLOR_DEBIT_RED = "FCE4D6"
COLOR_BORDER = "D9D9D9"
COLOR_META_HEADER = "D9E1F2"


class StatementExporter:
    """Exports processed bank statement transactions to formatted CSV and multi-tab Excel files."""

    def __init__(self, currency_symbol: str = "$") -> None:
        """Initialize exporter with currency configuration.

        Args:
            currency_symbol: Currency symbol used in number formatting (e.g. '$' or '₹').
        """
        self.currency_symbol = currency_symbol
        self.currency_format = f'"{self.currency_symbol}"#,##0.00'

        # Common reusable styles
        self.font_header = Font(name="Segoe UI", size=11, bold=True, color=COLOR_WHITE)
        self.font_bold = Font(name="Segoe UI", size=10, bold=True)
        self.font_regular = Font(name="Segoe UI", size=10)

        self.fill_header = PatternFill(start_color=COLOR_NAVY, end_color=COLOR_NAVY, fill_type="solid")
        self.fill_zebra = PatternFill(start_color=COLOR_ZEBRA_LIGHT, end_color=COLOR_ZEBRA_LIGHT, fill_type="solid")
        self.fill_credit = PatternFill(start_color=COLOR_CREDIT_GREEN, end_color=COLOR_CREDIT_GREEN, fill_type="solid")
        self.fill_debit = PatternFill(start_color=COLOR_DEBIT_RED, end_color=COLOR_DEBIT_RED, fill_type="solid")
        self.fill_meta_key = PatternFill(start_color=COLOR_META_HEADER, end_color=COLOR_META_HEADER, fill_type="solid")

        self.thin_border = Border(
            left=Side(style="thin", color=COLOR_BORDER),
            right=Side(style="thin", color=COLOR_BORDER),
            top=Side(style="thin", color=COLOR_BORDER),
            bottom=Side(style="thin", color=COLOR_BORDER),
        )
        self.double_bottom_border = Border(
            top=Side(style="thin", color=COLOR_BORDER),
            bottom=Side(style="double", color="000000"),
        )

    # ==========================================================================
    # 1. CSV Export
    # ==========================================================================
    def to_csv(self, df: pd.DataFrame, output_path: Union[str, Path]) -> str:
        """Export clean DataFrame to CSV format.

        Args:
            df: Transactions DataFrame.
            output_path: Target path for .csv file.

        Returns:
            Resolved absolute file path string.
        """
        dest = Path(output_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        export_df = df.copy() if df is not None else pd.DataFrame()
        export_df.to_csv(dest, index=False, encoding="utf-8")
        logger.info("Exported CSV to %s (%d rows)", dest, len(export_df))
        return str(dest)

    # ==========================================================================
    # 2. Multi-Tab Formatted Excel Export
    # ==========================================================================
    def to_excel(
        self,
        df: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
        output_path: Union[str, Path] = "statement_export.xlsx",
    ) -> str:
        """Build multi-tab styled Excel workbook using openpyxl.

        Tabs:
        - Tab 1: 'Transactions' (Navy header, currency formatting, zebra & debit/credit colors)
        - Tab 2: 'Category Summary' (Pivot spend breakdown, totals, percentages)
        - Tab 3: 'Account Metadata' (Account attributes, period, summary metrics)

        Args:
            df: Standardized transactions DataFrame.
            metadata: Optional account header metadata dictionary.
            output_path: Target file path.

        Returns:
            Resolved absolute file path string.
        """
        dest = Path(output_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        meta = metadata or {}
        tx_df = df.copy() if df is not None else pd.DataFrame()

        wb = openpyxl.Workbook()

        # Tab 1: Transactions
        ws_tx = wb.active
        ws_tx.title = "Transactions"
        self._build_transactions_sheet(ws_tx, tx_df)

        # Tab 2: Category Summary
        ws_cat = wb.create_sheet(title="Category Summary")
        self._build_category_summary_sheet(ws_cat, tx_df)

        # Tab 3: Account Metadata
        ws_meta = wb.create_sheet(title="Account Metadata")
        self._build_metadata_sheet(ws_meta, meta, tx_df)

        wb.save(dest)
        logger.info("Exported styled Excel workbook to %s", dest)
        return str(dest)

    # --------------------------------------------------------------------------
    # Tab 1 Builder: Transactions
    # --------------------------------------------------------------------------
    def _build_transactions_sheet(self, ws: openpyxl.worksheet.worksheet.Worksheet, df: pd.DataFrame) -> None:
        """Populate and format the Transactions sheet."""
        # Standard display columns
        columns = [c for c in ["date", "description", "debit", "credit", "balance", "category"] if c in df.columns]
        if not columns and not df.empty:
            columns = list(df.columns)
        elif not columns:
            columns = ["date", "description", "debit", "credit", "balance", "category"]

        # 1. Header Row
        headers = [col.capitalize() for col in columns]
        ws.append(headers)
        ws.row_dimensions[1].height = 28

        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = self.font_header
            cell.fill = self.fill_header
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = self.thin_border

        # Freeze header row
        ws.freeze_panes = "A2"

        # 2. Data Rows
        for r_idx, (_, row) in enumerate(df[columns].iterrows(), start=2):
            ws.row_dimensions[r_idx].height = 20
            is_even = (r_idx % 2 == 0)
            debit_val = float(row.get("debit", 0.0) or 0.0)
            credit_val = float(row.get("credit", 0.0) or 0.0)

            for c_idx, col_name in enumerate(columns, start=1):
                val = row[col_name]
                cell = ws.cell(row=r_idx, column=c_idx)
                cell.font = self.font_regular
                cell.border = self.thin_border

                # Alternating zebra background base
                cell.fill = self.fill_zebra if is_even else PatternFill(fill_type=None)

                # Column-specific formatting & highlighting
                if col_name == "date":
                    cell.value = str(val) if pd.notna(val) else ""
                    cell.alignment = Alignment(horizontal="center", vertical="center")

                elif col_name == "description":
                    cell.value = str(val) if pd.notna(val) else ""
                    cell.alignment = Alignment(horizontal="left", vertical="center")

                elif col_name in ("debit", "credit", "balance"):
                    try:
                        num_val = float(val) if pd.notna(val) else 0.0
                    except (ValueError, TypeError):
                        num_val = 0.0
                    cell.value = num_val
                    cell.number_format = self.currency_format
                    cell.alignment = Alignment(horizontal="right", vertical="center")

                    # Soft color highlighting
                    if col_name == "credit" and credit_val > 0:
                        cell.fill = self.fill_credit
                    elif col_name == "debit" and debit_val > 0:
                        cell.fill = self.fill_debit

                elif col_name == "category":
                    cell.value = str(val) if pd.notna(val) else "Uncategorized"
                    cell.alignment = Alignment(horizontal="center", vertical="center")

        # 3. Auto-fit column widths
        self._autofit_columns(ws, min_width=12)

    # --------------------------------------------------------------------------
    # Tab 2 Builder: Category Summary
    # --------------------------------------------------------------------------
    def _build_category_summary_sheet(self, ws: openpyxl.worksheet.worksheet.Worksheet, df: pd.DataFrame) -> None:
        """Populate Category Summary pivot breakdown."""
        # Title Banner
        ws.merge_cells("A1:E1")
        title_cell = ws.cell(row=1, column=1, value="Category Spend & Income Breakdown")
        title_cell.font = Font(name="Segoe UI", size=13, bold=True, color=COLOR_WHITE)
        title_cell.fill = self.fill_header
        title_cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 32

        # Header Row
        headers = ["Category", f"Total Spend ({self.currency_symbol})", f"Total Received ({self.currency_symbol})", "Count", "% of Total Spend"]
        ws.append(headers)
        ws.row_dimensions[2].height = 24

        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=2, column=col_idx)
            cell.font = Font(name="Segoe UI", size=10, bold=True, color=COLOR_NAVY)
            cell.fill = self.fill_meta_key
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = self.thin_border

        if df.empty or "category" not in df.columns:
            self._autofit_columns(ws)
            return

        # Aggregate data
        cat_col = "category"
        grouped = df.groupby(cat_col).agg(
            total_debit=("debit", "sum"),
            total_credit=("credit", "sum"),
            count=(cat_col, "count"),
        ).reset_index()

        total_spend_all = grouped["total_debit"].sum()
        grouped["pct_spend"] = grouped["total_debit"].apply(
            lambda x: (x / total_spend_all) if total_spend_all > 0 else 0.0
        )
        # Sort by total spend descending
        grouped = grouped.sort_values(by="total_debit", ascending=False)

        # Populate rows
        start_row = 3
        for r_offset, (_, row) in enumerate(grouped.iterrows()):
            curr_row = start_row + r_offset
            ws.row_dimensions[curr_row].height = 20
            is_even = (curr_row % 2 == 0)

            cells = [
                ws.cell(row=curr_row, column=1, value=str(row[cat_col])),
                ws.cell(row=curr_row, column=2, value=float(row["total_debit"])),
                ws.cell(row=curr_row, column=3, value=float(row["total_credit"])),
                ws.cell(row=curr_row, column=4, value=int(row["count"])),
                ws.cell(row=curr_row, column=5, value=float(row["pct_spend"])),
            ]

            for c_idx, c in enumerate(cells, start=1):
                c.font = self.font_regular
                c.fill = self.fill_zebra if is_even else PatternFill(fill_type=None)
                c.border = self.thin_border

                if c_idx == 1:
                    c.alignment = Alignment(horizontal="left", vertical="center")
                elif c_idx in (2, 3):
                    c.number_format = self.currency_format
                    c.alignment = Alignment(horizontal="right", vertical="center")
                elif c_idx == 4:
                    c.number_format = "#,##0"
                    c.alignment = Alignment(horizontal="center", vertical="center")
                elif c_idx == 5:
                    c.number_format = "0.0%"
                    c.alignment = Alignment(horizontal="right", vertical="center")

        # Summary Total Row
        tot_row = start_row + len(grouped)
        ws.row_dimensions[tot_row].height = 24
        tot_cells = [
            ws.cell(row=tot_row, column=1, value="Total"),
            ws.cell(row=tot_row, column=2, value=float(total_spend_all)),
            ws.cell(row=tot_row, column=3, value=float(grouped["total_credit"].sum())),
            ws.cell(row=tot_row, column=4, value=int(grouped["count"].sum())),
            ws.cell(row=tot_row, column=5, value=1.0 if total_spend_all > 0 else 0.0),
        ]

        for c_idx, c in enumerate(tot_cells, start=1):
            c.font = self.font_bold
            c.fill = self.fill_meta_key
            c.border = self.double_bottom_border
            if c_idx in (2, 3):
                c.number_format = self.currency_format
                c.alignment = Alignment(horizontal="right", vertical="center")
            elif c_idx == 4:
                c.number_format = "#,##0"
                c.alignment = Alignment(horizontal="center", vertical="center")
            elif c_idx == 5:
                c.number_format = "0.0%"
                c.alignment = Alignment(horizontal="right", vertical="center")

        self._autofit_columns(ws, min_width=14)

    # --------------------------------------------------------------------------
    # Tab 3 Builder: Account Metadata
    # --------------------------------------------------------------------------
    def _build_metadata_sheet(
        self,
        ws: openpyxl.worksheet.worksheet.Worksheet,
        meta: Dict[str, Any],
        df: pd.DataFrame,
    ) -> None:
        """Populate Account Metadata sheet."""
        # Title Banner
        ws.merge_cells("A1:B1")
        banner = ws.cell(row=1, column=1, value="Bank Statement Summary & Account Metadata")
        banner.font = Font(name="Segoe UI", size=13, bold=True, color=COLOR_WHITE)
        banner.fill = self.fill_header
        banner.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 32

        # Calculate high-level summary metrics
        total_tx = len(df)
        total_debits = float(df["debit"].sum()) if not df.empty and "debit" in df.columns else 0.0
        total_credits = float(df["credit"].sum()) if not df.empty and "credit" in df.columns else 0.0
        closing_balance = float(df.iloc[-1]["balance"]) if not df.empty and "balance" in df.columns else 0.0

        metadata_items = [
            ("Account Holder Name", meta.get("account_holder") or "N/A"),
            ("Account Number", meta.get("account_number") or "N/A"),
            ("IFSC / SWIFT / Routing Code", meta.get("routing_code") or "N/A"),
            ("Statement Period", meta.get("statement_period") or "N/A"),
            ("Total Transactions Processed", f"{total_tx:,}"),
            (f"Total Debits / Withdrawals ({self.currency_symbol})", f"{self.currency_symbol}{total_debits:,.2f}"),
            (f"Total Credits / Deposits ({self.currency_symbol})", f"{self.currency_symbol}{total_credits:,.2f}"),
            (f"Closing Account Balance ({self.currency_symbol})", f"{self.currency_symbol}{closing_balance:,.2f}"),
            ("Export Timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ]

        for idx, (label, val) in enumerate(metadata_items, start=3):
            ws.row_dimensions[idx].height = 22

            cell_lbl = ws.cell(row=idx, column=1, value=label)
            cell_lbl.font = self.font_bold
            cell_lbl.fill = self.fill_meta_key
            cell_lbl.border = self.thin_border
            cell_lbl.alignment = Alignment(horizontal="left", vertical="center")

            cell_val = ws.cell(row=idx, column=2, value=str(val))
            cell_val.font = self.font_regular
            cell_val.border = self.thin_border
            cell_val.alignment = Alignment(horizontal="left", vertical="center")

        ws.column_dimensions["A"].width = 34
        ws.column_dimensions["B"].width = 40

    # --------------------------------------------------------------------------
    # Utility: Column Auto-Fit
    # --------------------------------------------------------------------------
    def _autofit_columns(self, ws: openpyxl.worksheet.worksheet.Worksheet, min_width: int = 12) -> None:
        """Auto-adjust worksheet column widths based on maximum content length."""
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val = cell.value
                if val is not None:
                    max_len = max(max_len, len(str(val)))
            ws.column_dimensions[col_letter].width = max(max_len + 4, min_width)


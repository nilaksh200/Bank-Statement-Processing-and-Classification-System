"""Native Text and Table Extraction Module for Bank Statements.

Uses pdfplumber and PyMuPDF to extract account metadata from statement headers
and parse tabular transaction data with multi-line description merging and
single-amount column splitting into canonical format.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore

import pdfplumber
import pandas as pd

logger = logging.getLogger(__name__)

# Standard target columns
TRANSACTION_COLUMNS: List[str] = ["date", "description", "debit", "credit", "balance"]

# Date matchers for line detection and validation
DATE_PATTERNS = [
    # YYYY-MM-DD or YYYY/MM/DD
    r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}",
    # DD-MM-YYYY or DD/MM/YYYY or MM/DD/YYYY
    r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}",
    # 15 Jan 2026 or 15-Jan-2026 or 15 January 2026
    r"^\d{1,2}[-\s]+[A-Za-z]{3,9}[-\s]+\d{2,4}",
    # Jan 15, 2026 or January 15 2026
    r"^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}",
]
DATE_REGEX = re.compile("|".join(f"(?:{p})" for p in DATE_PATTERNS), re.IGNORECASE)


class TextStatementParser:
    """Extracts account metadata and tabular transactions from text-based bank statements."""

    def __init__(self, table_settings: Optional[Dict[str, Any]] = None) -> None:
        """Initialize parser with optional pdfplumber table extraction settings.

        Args:
            table_settings: Custom settings dict for pdfplumber.extract_tables().
        """
        self.primary_table_settings: Dict[str, Any] = table_settings or {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 3,
        }
        self.fallback_table_settings: Dict[str, Any] = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "min_words_vertical": 2,
        }

    # ==========================================================================
    # 1. Header Metadata Extraction
    # ==========================================================================
    def extract_metadata(self, pdf_path: Union[str, Path]) -> Dict[str, Optional[str]]:
        """Extract key account details from the first page of the statement.

        Extracts:
        - Account Holder Name
        - Account Number
        - IFSC / SWIFT / Routing Code / Sort Code
        - Statement Period

        Args:
            pdf_path: Path to the target PDF statement.

        Returns:
            Dictionary containing extracted metadata fields.
        """
        path = Path(pdf_path).resolve()
        first_page_text = self._extract_first_page_text(path)

        metadata: Dict[str, Optional[str]] = {
            "account_holder": self._extract_account_holder(first_page_text),
            "account_number": self._extract_account_number(first_page_text),
            "routing_code": self._extract_routing_code(first_page_text),
            "statement_period": self._extract_statement_period(first_page_text),
        }
        return metadata

    def _extract_first_page_text(self, path: Path) -> str:
        """Extract text from the first page using PyMuPDF."""
        try:
            doc = fitz.open(str(path))
            if len(doc) == 0:
                doc.close()
                return ""
            text = doc[0].get_text() or ""
            doc.close()
            return text
        except Exception as exc:
            logger.warning("Could not extract first page text with PyMuPDF: %s", exc)
            return ""

    def _extract_account_holder(self, text: str) -> Optional[str]:
        """Extract account holder name using heuristic patterns."""
        patterns = [
            r"(?i)(?:Account\s+Holder|Customer|Account)\s*Name\s*[:\-]?\s*([A-Za-z\s\.\,\-]+)",
            r"(?i)Account\s+Holder\s*[:\-]?\s*([A-Za-z\s\.\,\-]+)",
            r"(?i)Name\s*[:\-]\s*([A-Z][a-zA-Z\s\.\,\-]+)",
            r"(?i)(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-zA-Z\s]+)",
        ]
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                raw_name = match.group(1).split("\n")[0].strip()
                # Clean delimiters or trailing identifiers
                cleaned = re.sub(r"[|;].*$", "", raw_name).strip()
                if 2 <= len(cleaned) <= 60 and not re.search(r"\b(number|no|date|period|branch)\b", cleaned, re.I):
                    return cleaned
        return None

    def _extract_account_number(self, text: str) -> Optional[str]:
        """Extract account number (plain or masked)."""
        patterns = [
            r"(?i)(?:Account|A/c|Acct)\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([X\*\d\-]{6,26})",
            r"(?i)\bAcc(?:t)?\s*#?\s*[:\-]?\s*([X\*\d\-]{6,26})\b",
            r"(?i)\b(?:Account\s+ID)\s*[:\-]?\s*([X\*\d\-]{6,26})\b",
        ]
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                acc = match.group(1).strip()
                # Must contain at least 4 digits
                if sum(c.isdigit() for c in acc) >= 4:
                    return acc
        return None

    def _extract_routing_code(self, text: str) -> Optional[str]:
        """Extract routing, IFSC, SWIFT, or sort code."""
        # 1. Indian Financial System Code (IFSC)
        ifsc_match = re.search(r"\bIFSC(?:\s*Code)?\s*[:\-]?\s*([A-Z]{4}0[A-Z0-9]{6})\b", text, re.I)
        if ifsc_match:
            return ifsc_match.group(1).upper()

        # 2. SWIFT / BIC Code
        swift_match = re.search(r"\b(?:SWIFT|BIC)(?:\s*Code)?\s*[:\-]?\s*([A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)\b", text, re.I)
        if swift_match:
            return swift_match.group(1).upper()

        # 3. US ABA Routing Number (9 digits)
        routing_match = re.search(r"(?i)\b(?:Routing(?:\s*No\.?|\s*Number)?|ABA)\s*[:\-]?\s*(\d{9})\b", text)
        if routing_match:
            return routing_match.group(1)

        # 4. UK Sort Code (6 digits or DD-DD-DD)
        sort_match = re.search(r"(?i)\b(?:Sort\s*Code)\s*[:\-]?\s*(\d{2}-\d{2}-\d{2}|\d{6})\b", text)
        if sort_match:
            return sort_match.group(1)

        return None

    def _extract_statement_period(self, text: str) -> Optional[str]:
        """Extract statement period range."""
        patterns = [
            r"(?i)Statement\s+Period\s*[:\-]?\s*([A-Za-z0-9\/\-\.]+\s*(?:to|-|–)\s*[A-Za-z0-9\/\-\.]+)",
            r"(?i)Period\s*[:\-]?\s*([A-Za-z0-9\/\-\.]+\s*(?:to|-|–)\s*[A-Za-z0-9\/\-\.]+)",
            r"(?i)From\s*[:\-]?\s*([A-Za-z0-9\/\-\.]+)\s*To\s*[:\-]?\s*([A-Za-z0-9\/\-\.]+)",
            r"(?i)Date\s*Range\s*[:\-]?\s*([A-Za-z0-9\/\-\.]+\s*(?:to|-|–)\s*[A-Za-z0-9\/\-\.]+)",
        ]
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                if len(match.groups()) == 2:
                    return f"{match.group(1).strip()} to {match.group(2).strip()}"
                return match.group(1).strip()
        return None

    # ==========================================================================
    # 2. Table and Transaction Extraction
    # ==========================================================================
    def extract_transactions(self, pdf_path: Union[str, Path]) -> pd.DataFrame:
        """Extract tabular transactions across all pages into a standardized DataFrame.

        Standard columns: ['date', 'description', 'debit', 'credit', 'balance']

        Args:
            pdf_path: Path to target PDF.

        Returns:
            pd.DataFrame with standard schema and parsed numerical fields.
        """
        path = Path(pdf_path).resolve()
        raw_tables = self._extract_raw_tables_pdfplumber(path)

        if not raw_tables:
            logger.info("No line-bounded tables found; trying text-stream extraction.")
            raw_tables = self._extract_raw_tables_pdfplumber(path, use_fallback=True)

        if not raw_tables:
            # Fallback to line-by-line regex stream parser
            logger.info("Falling back to text stream line parsing.")
            return self._parse_from_text_stream(path)

        # Parse collected tables
        transactions = self._parse_table_grid(raw_tables)
        if not transactions:
            return self._parse_from_text_stream(path)

        df = pd.DataFrame(transactions)
        return self._format_dataframe(df)

    def _extract_raw_tables_pdfplumber(
        self,
        path: Path,
        use_fallback: bool = False,
    ) -> List[List[List[Optional[str]]]]:
        """Extract raw tables from all pages using pdfplumber."""
        collected_tables: List[List[List[Optional[str]]]] = []
        settings = self.fallback_table_settings if use_fallback else self.primary_table_settings

        try:
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables(settings)
                    for t in tables:
                        if t and len(t) >= 2:  # Must have header + at least 1 row
                            collected_tables.append(t)
        except Exception as exc:
            logger.warning("Error in pdfplumber table extraction for %s: %s", path, exc)

        return collected_tables

    # ==========================================================================
    # 3. Table Grid Parsing & Normalization
    # ==========================================================================
    def _parse_table_grid(self, tables: List[List[List[Optional[str]]]]) -> List[Dict[str, Any]]:
        """Parse raw 2D tables, identify columns, merge multi-lines, and split amounts."""
        transactions: List[Dict[str, Any]] = []
        current_column_map: Optional[Dict[str, int]] = None

        for table in tables:
            for row in table:
                if not row or all(c is None or str(c).strip() == "" for c in row):
                    continue

                cleaned_row = [str(c).strip() if c is not None else "" for c in row]

                # Check if this row is a header row
                col_map = self._detect_header_mapping(cleaned_row)
                if col_map:
                    current_column_map = col_map
                    continue

                if not current_column_map:
                    continue

                # Process transaction row with active column mapping
                parsed_row = self._parse_row(cleaned_row, current_column_map)
                if not parsed_row:
                    continue

                # Multi-line handling:
                # If date is present, it's a primary transaction row
                if parsed_row["date"]:
                    transactions.append(parsed_row)
                elif transactions and parsed_row["description"]:
                    # Wrapped narrative: concatenate to previous transaction
                    prev = transactions[-1]
                    prev["description"] = f"{prev['description']} {parsed_row['description']}".strip()
                    # If debit/credit/balance were missing previously but found here, backfill
                    if prev["debit"] == 0.0 and parsed_row["debit"] > 0:
                        prev["debit"] = parsed_row["debit"]
                    if prev["credit"] == 0.0 and parsed_row["credit"] > 0:
                        prev["credit"] = parsed_row["credit"]
                    if prev["balance"] == 0.0 and parsed_row["balance"] != 0.0:
                        prev["balance"] = parsed_row["balance"]

        return transactions

    def _detect_header_mapping(self, row: List[str]) -> Optional[Dict[str, int]]:
        """Identify if a row is a table header and map column names to column indices."""
        mapping: Dict[str, int] = {}
        for idx, col in enumerate(row):
            norm = col.lower().replace("\n", " ").strip()
            if not norm:
                continue

            if re.search(r"\b(?:date|txn\s*date|value\s*date|post\s*date)\b", norm):
                mapping.setdefault("date", idx)
            elif re.search(r"\b(?:description|narration|particulars|details|transaction\s*details|remarks)\b", norm):
                mapping.setdefault("description", idx)
            elif re.search(r"\b(?:debit|dr|withdrawal|withdrawals|spend|paid\s*out)\b", norm):
                mapping.setdefault("debit", idx)
            elif re.search(r"\b(?:credit|cr|deposit|deposits|received|paid\s*in)\b", norm):
                mapping.setdefault("credit", idx)
            elif re.search(r"\b(?:amount|txn\s*amount|trans\s*amount)\b", norm):
                mapping.setdefault("amount", idx)
            elif re.search(r"\b(?:balance|closing\s*balance|available\s*balance|bal)\b", norm):
                mapping.setdefault("balance", idx)
            elif re.search(r"\b(?:type|d/c|cr/dr|dr/cr)\b", norm):
                mapping.setdefault("type", idx)

        # A valid bank table header must identify at least (date + description) and (amount OR debit/credit)
        has_date = "date" in mapping
        has_desc = "description" in mapping
        has_amt = "amount" in mapping or ("debit" in mapping or "credit" in mapping)

        if has_date and (has_desc or has_amt):
            return mapping
        return None

    def _parse_row(self, row: List[str], col_map: Dict[str, int]) -> Optional[Dict[str, Any]]:
        """Parse a single row using column mapping, resolving multi-lines and amount splitting."""
        num_cols = len(row)

        def get_val(col_key: str) -> str:
            idx = col_map.get(col_key)
            if idx is not None and idx < num_cols:
                return row[idx].strip()
            return ""

        date_val = get_val("date")
        desc_val = get_val("description")
        balance_val = get_val("balance")

        # Validate date
        is_date = bool(DATE_REGEX.search(date_val)) if date_val else False
        clean_date = date_val if is_date else ""

        # Debit & Credit resolution
        debit = 0.0
        credit = 0.0

        if "amount" in col_map and ("debit" not in col_map and "credit" not in col_map):
            # Combined Single Amount column
            amt_raw = get_val("amount")
            type_raw = get_val("type")
            debit, credit = self._split_single_amount(amt_raw, type_raw)
        else:
            # Separate Debit / Credit columns
            debit_raw = get_val("debit")
            credit_raw = get_val("credit")
            debit = self._parse_amount_value(debit_raw)
            credit = self._parse_amount_value(credit_raw)

            # If only one of debit or credit was present in header or filled, check DR/CR tags
            if debit == 0.0 and "dr" in debit_raw.lower():
                debit = self._parse_amount_value(re.sub(r"(?i)dr", "", debit_raw))
            if credit == 0.0 and "cr" in credit_raw.lower():
                credit = self._parse_amount_value(re.sub(r"(?i)cr", "", credit_raw))

        parsed_balance = self._parse_amount_value(balance_val)

        # Ignore empty summary or blank separator lines
        if not clean_date and not desc_val and debit == 0.0 and credit == 0.0:
            return None

        # Ignore repeating header texts inside the data
        if any(h in desc_val.lower() for h in ["balance brought forward", "opening balance"]):
            # Still valid transaction or opening line
            pass
        elif "description" in desc_val.lower() and "date" in date_val.lower():
            return None

        return {
            "date": clean_date,
            "description": desc_val,
            "debit": float(debit),
            "credit": float(credit),
            "balance": float(parsed_balance),
        }

    # ==========================================================================
    # 4. Single "Amount" Splitting Heuristics
    # ==========================================================================
    def _split_single_amount(self, amount_str: str, type_str: str = "") -> Tuple[float, float]:
        """Split a combined single 'Amount' column into debit and credit amounts.

        Rules:
        - Contains 'DR', 'Debit', '-' sign, or (parentheses) -> Debit
        - Contains 'CR', 'Credit', '+' sign -> Credit
        - Check secondary type_str indicator if present
        - Defaults: positive amounts with no explicit sign default to debit if expense or credit
        """
        combined = f"{amount_str} {type_str}".strip()
        if not combined:
            return 0.0, 0.0

        upper = combined.upper()
        # 1. Indicator for Credit
        is_credit = bool(
            "CR" in upper
            or "CREDIT" in upper
            or "+" in upper
            or "DEP" in upper
        )

        # 2. Indicator for Debit
        is_debit = bool(
            "DR" in upper
            or "DEBIT" in upper
            or "-" in upper
            or re.search(r"\(.*\)", amount_str)  # accounting parentheses, e.g. (150.00)
            or "WD" in upper
        )

        numeric_val = self._parse_amount_value(amount_str)

        if is_credit and not is_debit:
            return 0.0, numeric_val
        if is_debit and not is_credit:
            return numeric_val, 0.0

        # Sign-based default: negative is debit, positive is credit
        if "-" in amount_str:
            return numeric_val, 0.0
        elif "+" in amount_str:
            return 0.0, numeric_val

        # If indeterminate, treat positive amount without signs as Debit (most frequent in card/checking)
        # or Credit if indicated by parent column
        return numeric_val, 0.0

    # ==========================================================================
    # 5. Fallback Text Stream Parser (Line-by-Line)
    # ==========================================================================
    def _parse_from_text_stream(self, path: Path) -> pd.DataFrame:
        """Parse borderless statements from raw text stream line by line."""
        transactions: List[Dict[str, Any]] = []

        try:
            doc = fitz.open(str(path))
        except Exception:
            return pd.DataFrame(columns=TRANSACTION_COLUMNS)

        try:
            for page in doc:
                text = page.get_text() or ""
                lines = text.splitlines()

                for line in lines:
                    line_str = line.strip()
                    if not line_str:
                        continue

                    # Check if line starts with a date
                    date_match = DATE_REGEX.search(line_str)
                    if date_match and date_match.start() == 0:
                        # Parse date
                        date_str = date_match.group(0).strip()
                        remainder = line_str[len(date_str):].strip()

                        # Extract numbers from remainder
                        amounts = re.findall(r"[-+]?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}", remainder)

                        if amounts:
                            # Usually: [debit/credit, balance] or [debit, credit, balance]
                            balance = self._parse_amount_value(amounts[-1])
                            debit = 0.0
                            credit = 0.0

                            if len(amounts) >= 3:
                                debit = self._parse_amount_value(amounts[-3])
                                credit = self._parse_amount_value(amounts[-2])
                            elif len(amounts) == 2:
                                # Could be (debit/credit, balance)
                                amt_val = self._parse_amount_value(amounts[0])
                                if "-" in amounts[0] or "dr" in remainder.lower():
                                    debit = amt_val
                                else:
                                    credit = amt_val
                            elif len(amounts) == 1:
                                balance = self._parse_amount_value(amounts[0])

                            # Clean description by removing amounts
                            desc_candidate = remainder
                            for a in amounts:
                                desc_candidate = desc_candidate.replace(a, "")
                            desc_candidate = re.sub(r"\s+", " ", desc_candidate).strip()

                            transactions.append({
                                "date": date_str,
                                "description": desc_candidate,
                                "debit": float(debit),
                                "credit": float(credit),
                                "balance": float(balance),
                            })
                    elif transactions and len(line_str) > 2:
                        # Multi-line wrapped description text
                        if not re.search(r"(?i)\b(page \d|total|balance forward|statement)\b", line_str):
                            transactions[-1]["description"] = f"{transactions[-1]['description']} {line_str}".strip()
        finally:
            doc.close()

        df = pd.DataFrame(transactions)
        return self._format_dataframe(df)

    # ==========================================================================
    # 6. Cleaning & Formatting Helpers
    # ==========================================================================
    def _parse_amount_value(self, val_str: Any) -> float:
        """Parse string amount to float, stripping currencies, commas, and negative signs."""
        if val_str is None:
            return 0.0
        val = str(val_str).strip()
        if not val or val in ("-", "--", "N/A", "None"):
            return 0.0

        # Remove currency symbols ($ € £ ₹ CAD USD)
        val = re.sub(r"[\$€£₹A-Za-z\s]", "", val)
        # Remove commas in thousand separators: 1,234.56 -> 1234.56
        val = val.replace(",", "")
        # Remove parenthetical negatives: (120.50) -> 120.50
        val = val.replace("(", "").replace(")", "")

        try:
            return abs(float(val))
        except ValueError:
            return 0.0

    def _format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure standard column structure, correct dtypes, and clean values."""
        if df.empty:
            return pd.DataFrame(columns=TRANSACTION_COLUMNS)

        # Ensure all columns exist
        for col in TRANSACTION_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0 if col in ("debit", "credit", "balance") else ""

        # Filter out rows with empty date or description
        df = df[df["date"].astype(str).str.strip() != ""].copy()
        df = df[df["description"].astype(str).str.strip() != ""].copy()

        # Clean string columns
        df["date"] = df["date"].astype(str).str.strip()
        df["description"] = df["description"].astype(str).str.strip()

        # Clean float columns
        df["debit"] = pd.to_numeric(df["debit"], errors="coerce").fillna(0.0).round(2)
        df["credit"] = pd.to_numeric(df["credit"], errors="coerce").fillna(0.0).round(2)
        df["balance"] = pd.to_numeric(df["balance"], errors="coerce").fillna(0.0).round(2)

        # Reset index and enforce column ordering
        df = df[TRANSACTION_COLUMNS].reset_index(drop=True)
        return df


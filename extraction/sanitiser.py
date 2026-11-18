"""Data Sanitisation Module for Extracted Bank Statements.

Standardizes dates into ISO YYYY-MM-DD format, normalizes financial amounts
by stripping currencies and delimiters, cleans merchant narratives, and
enforces schema integrity on extracted DataFrames.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Optional, Union

import pandas as pd
from dateutil import parser as date_parser

from extraction.text_parser import TRANSACTION_COLUMNS

logger = logging.getLogger(__name__)

# Common non-numeric currency identifiers
CURRENCY_REGEX = re.compile(r"[\$€£₹¥A-Za-z\s]")

# Regex to capture date components
EXPLICIT_ISO_REGEX = re.compile(r"^\s*(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\s*$")


def clean_date(date_str: Any, dayfirst: bool = True) -> str:
    """Parse various date formats into standard YYYY-MM-DD ISO format.

    Supports:
    - ISO formats: `YYYY-MM-DD`, `YYYY/MM/DD`, `YYYY.MM.DD`
    - European/Standard: `DD/MM/YYYY`, `DD-MM-YYYY`, `DD.MM.YYYY`, `DD/MM/YY`
    - Month words: `DD-MMM-YYYY`, `DD MMM YYYY`, `MMM DD, YYYY`, `15 January 2026`
    - US format fallback: `MM/DD/YYYY`

    Args:
        date_str: Raw input date representation.
        dayfirst: Whether to interpret the first numerical field as day (standard in banking).

    Returns:
        Formatted date string `YYYY-MM-DD`, or empty string `""` if invalid.
    """
    if date_str is None or pd.isna(date_str):
        return ""

    raw = str(date_str).strip()
    if not raw or raw.lower() in ("-", "--", "none", "nan", "nat", "n/a"):
        return ""

    # 1. Fast path: already YYYY-MM-DD
    iso_match = EXPLICIT_ISO_REGEX.match(raw)
    if iso_match:
        y, m, d = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
        try:
            return datetime(y, m, d).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # 2. Extract primary date token if surrounded by noise
    date_token_match = re.search(
        r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}[-\s]+[A-Za-z]{3,9}[-\s]+\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})\b",
        raw,
    )
    target_str = date_token_match.group(0) if date_token_match else raw

    # 3. Flexible parsing via python-dateutil
    try:
        dt = date_parser.parse(target_str, dayfirst=dayfirst)
        # Year sanity check (1970 to 2099)
        if 1970 <= dt.year <= 2099:
            return dt.strftime("%Y-%m-%d")
        # Handle 2-digit years
        if dt.year < 100:
            century = 2000 if dt.year < 70 else 1900
            return dt.replace(year=century + dt.year).strftime("%Y-%m-%d")
    except (ValueError, TypeError, OverflowError):
        pass

    # 4. Fallback: try parsing with dayfirst=False for US-style MM/DD/YYYY
    try:
        dt = date_parser.parse(target_str, dayfirst=False)
        if 1970 <= dt.year <= 2099:
            return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError, OverflowError):
        pass

    return raw


def clean_amount(amount_str: Any) -> float:
    """Strip currency symbols, commas, and spaces, converting to float.

    Converts empty strings, dashes, or unparseable inputs to `0.0`.
    Handles parenthetical accounting representations `(100.00)` as positive magnitude.

    Args:
        amount_str: Raw string or numeric representation.

    Returns:
        Rounded float value (2 decimal places).
    """
    if amount_str is None or pd.isna(amount_str):
        return 0.0

    if isinstance(amount_str, (int, float)):
        return round(abs(float(amount_str)), 2)

    val = str(amount_str).strip()
    if not val or val.lower() in ("-", "--", "none", "nan", "null", "n/a", "nil"):
        return 0.0

    # Strip currency symbols and letters: $, €, £, ₹, USD, CAD, CR, DR
    cleaned = CURRENCY_REGEX.sub("", val)

    # Remove commas and accounting parentheses
    cleaned = cleaned.replace(",", "").replace("(", "").replace(")", "").strip()

    if not cleaned:
        return 0.0

    try:
        num = float(cleaned)
        return round(abs(num), 2)
    except ValueError:
        logger.debug("Could not parse amount string to float: %r", amount_str)
        return 0.0


def sanitise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitise and standardize raw DataFrame columns.

    - Cleans and normalizes `date` to YYYY-MM-DD.
    - Cleans and formats `debit`, `credit`, `balance` to 2-decimal floats.
    - Strips whitespace and linebreaks from `description`.
    - Eliminates empty rows.

    Args:
        df: Raw input DataFrame.

    Returns:
        Sanitised DataFrame conforming to canonical columns.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=TRANSACTION_COLUMNS)

    clean_df = df.copy()

    # Ensure all canonical columns exist
    for col in TRANSACTION_COLUMNS:
        if col not in clean_df.columns:
            clean_df[col] = 0.0 if col in ("debit", "credit", "balance") else ""

    # 1. Clean Dates
    clean_df["date"] = clean_df["date"].apply(clean_date)

    # 2. Clean Descriptions
    clean_df["description"] = (
        clean_df["description"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # 3. Clean Amounts
    clean_df["debit"] = clean_df["debit"].apply(clean_amount)
    clean_df["credit"] = clean_df["credit"].apply(clean_amount)
    clean_df["balance"] = clean_df["balance"].apply(clean_amount)

    # 4. Remove empty noise rows
    mask = (
        (clean_df["date"] != "")
        | (clean_df["description"] != "")
        | (clean_df["debit"] > 0)
        | (clean_df["credit"] > 0)
    )
    clean_df = clean_df[mask].reset_index(drop=True)

    return clean_df[TRANSACTION_COLUMNS]


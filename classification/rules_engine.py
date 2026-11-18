"""Deterministic Heuristic and Regex Classification Engine for Bank Transactions.

Strictly Non-LLM rule engine utilizing keyword heuristics and compiled regular
expressions to classify transaction narratives with deterministic confidence scores.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from config.settings import CONFIG_DIR

logger = logging.getLogger(__name__)

# Fallback default rule patterns if categories.json is absent
DEFAULT_RULE_PATTERNS: Dict[str, str] = {
    "Salary / Income": r"(?i)(salary|payroll|direct deposit|stipend|wages)",
    "Utilities & Bills": r"(?i)(electricity|water|gas|broadband|wifi|mobile|recharge|bill)",
    "Groceries & Supermarket": r"(?i)(mart|supermarket|grocery|trader|walmart|whole foods)",
    "Dining & Food": r"(?i)(restaurant|cafe|swiggy|zomato|starbucks|uber eats|mcdonald)",
    "Transfers & UPI": r"(?i)(upi|neft|rtgs|imps|zelle|venmo|transfer)",
    "ATM & Cash": r"(?i)(atm|cash withdrawal)",
    "Investment & Insurance": r"(?i)(mutual fund|lic|insurance|zerodha|sip|stock)",
    "Rent & Housing": r"(?i)(rent|lease|mortgage|housing|landlord|maintenance)",
    "Entertainment & Subscriptions": r"(?i)(netflix|spotify|prime|hulu|disney|theatre|cinema|movie)",
    "Travel & Transportation": r"(?i)(uber|lyft|airline|flight|metro|transit|railway|train|fuel|petrol|gas station|shell)",
    "Health & Medical": r"(?i)(pharmacy|hospital|clinic|doctor|medical|health|cvs|walgreens|dental)",
    "Bank Fees & Charges": r"(?i)(fee|charge|interest charge|overdraft|service charge|penalty)",
}


class HeuristicClassifier:
    """Classifies transaction descriptions using deterministic regular expressions."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        custom_rules: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize heuristic classifier with category rules.

        Args:
            config_path: Optional path to JSON category definitions file.
            custom_rules: Optional explicit mapping of {category_name: regex_pattern}.
        """
        self.rules: Dict[str, re.Pattern] = {}
        self.default_category: str = "Uncategorized"

        # 1. Load rules from custom dictionary, JSON config, or default fallback
        raw_rules: Dict[str, str] = {}

        json_file = Path(config_path) if config_path else CONFIG_DIR / "categories.json"
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("categories", []):
                        raw_rules[item["name"]] = item["regex"]
                    self.default_category = data.get("default_category", "Uncategorized")
            except Exception as exc:
                logger.warning("Could not read category rules from %s: %s", json_file, exc)
                raw_rules = dict(DEFAULT_RULE_PATTERNS)
        else:
            raw_rules = dict(DEFAULT_RULE_PATTERNS)

        if custom_rules:
            raw_rules.update(custom_rules)

        # 2. Pre-compile regular expressions
        for category, pattern in raw_rules.items():
            try:
                self.rules[category] = re.compile(pattern, re.IGNORECASE)
            except re.error as err:
                logger.error("Invalid regex for category '%s' (%s): %s", category, pattern, err)

    def classify_description(self, description: str) -> Tuple[str, float]:
        """Classify a single transaction narrative string.

        Args:
            description: Raw or cleaned transaction description.

        Returns:
            Tuple of `(category, confidence_score)`:
            - `(category, 1.0)` for a positive regex match.
            - `("Uncategorized", 0.0)` if no rule matches.
        """
        if not description or not isinstance(description, str) or not description.strip():
            return self.default_category, 0.0

        text = description.strip()

        # Iterate rules in defined order
        for category, compiled_regex in self.rules.items():
            if compiled_regex.search(text):
                return category, 1.0

        return self.default_category, 0.0

    def classify_batch(self, descriptions: List[str]) -> List[Tuple[str, float]]:
        """Classify a list of transaction narratives."""
        return [self.classify_description(d) for d in descriptions]

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        description_col: str = "description",
        output_col: str = "category",
        include_confidence: bool = False,
    ) -> pd.DataFrame:
        """Classify transactions in a pandas DataFrame.

        Args:
            df: Input transactions DataFrame.
            description_col: Column name containing merchant text.
            output_col: Column name to write assigned category.
            include_confidence: Whether to add a 'confidence' column.

        Returns:
            DataFrame with assigned category column.
        """
        if df.empty or description_col not in df.columns:
            out_df = df.copy()
            out_df[output_col] = self.default_category
            if include_confidence:
                out_df["confidence"] = 0.0
            return out_df

        results = self.classify_batch(df[description_col].fillna("").astype(str).tolist())

        out_df = df.copy()
        out_df[output_col] = [cat for cat, _ in results]
        if include_confidence:
            out_df["confidence"] = [conf for _, conf in results]

        return out_df

    def add_rule(self, category: str, pattern: str) -> None:
        """Add or override a category regex pattern at runtime."""
        self.rules[category] = re.compile(pattern, re.IGNORECASE)


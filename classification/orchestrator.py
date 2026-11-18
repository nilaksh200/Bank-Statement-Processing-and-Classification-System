"""Dual-Engine Hybrid Classification Orchestrator.

Combines deterministic regex heuristics (Layer 1) with traditional scikit-learn
TF-IDF machine learning (Layer 2) to categorize transactions with high throughput,
thread safety, and confidence gating without using Large Language Models (LLMs).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from classification.ml_classifier import TraditionalMLClassifier
from classification.rules_engine import HeuristicClassifier

logger = logging.getLogger(__name__)


class TransactionClassificationEngine:
    """Hybrid two-tier classification orchestrator for bank transactions.

    Tier 1: HeuristicClassifier (deterministic regex/keyword rules, confidence = 1.0).
    Tier 2: TraditionalMLClassifier (TF-IDF + MultinomialNB for ambiguous descriptions).
    Fallback: Marks transactions with confidence < 0.4 as 'Uncategorized'.
    """

    def __init__(
        self,
        heuristic_classifier: Optional[HeuristicClassifier] = None,
        ml_classifier: Optional[TraditionalMLClassifier] = None,
        heuristic_threshold: float = 0.8,
        ml_threshold: float = 0.4,
    ) -> None:
        """Initialize dual-engine classification orchestrator.

        Args:
            heuristic_classifier: Optional pre-configured HeuristicClassifier instance.
            ml_classifier: Optional pre-configured TraditionalMLClassifier instance.
            heuristic_threshold: Minimum confidence required from Layer 1 (default: 0.8).
            ml_threshold: Minimum confidence required from Layer 2 (default: 0.4).
        """
        self.heuristic = heuristic_classifier or HeuristicClassifier()
        self.ml = ml_classifier or TraditionalMLClassifier()
        self.heuristic_threshold = heuristic_threshold
        self.ml_threshold = ml_threshold
        self._lock = threading.RLock()

    # ==========================================================================
    # 1. Single Transaction Classification
    # ==========================================================================
    def classify_single(self, description: str) -> Tuple[str, float, str]:
        """Classify a single transaction narrative.

        Args:
            description: Narrative text of the transaction.

        Returns:
            Tuple of `(category, confidence_score, engine_used)` where engine_used is
            'heuristic', 'ml', or 'fallback'.
        """
        if not description or not isinstance(description, str) or not description.strip():
            return "Uncategorized", 0.0, "fallback"

        clean_desc = description.strip()

        # Tier 1: Regex & Heuristic Engine
        h_cat, h_conf = self.heuristic.classify_description(clean_desc)
        if h_cat != "Uncategorized" and h_conf >= self.heuristic_threshold:
            return h_cat, h_conf, "heuristic"

        # Tier 2: Traditional Machine Learning Engine
        with self._lock:
            ml_cat, ml_conf = self.ml.predict(clean_desc)

        if ml_cat != "Uncategorized" and ml_conf >= self.ml_threshold:
            return ml_cat, ml_conf, "ml"

        return "Uncategorized", ml_conf, "fallback"

    # ==========================================================================
    # 2. High-Throughput Batch DataFrame Classification
    # ==========================================================================
    def classify_transactions(
        self,
        df: pd.DataFrame,
        description_col: str = "description",
        category_col: str = "category",
        confidence_col: str = "confidence",
        include_method: bool = False,
    ) -> pd.DataFrame:
        """Classify transactions in a pandas DataFrame using vectorized hybrid routing.

        Adds `category` and `confidence` columns. Evaluates descriptions through
        HeuristicClassifier first; routes unmatched/low-confidence items to
        TraditionalMLClassifier in high-speed vectorized batches.

        Args:
            df: Input transactions DataFrame.
            description_col: Name of column containing narrative descriptions.
            category_col: Target column name for assigned category.
            confidence_col: Target column name for classification confidence.
            include_method: If True, adds 'classification_method' ('heuristic', 'ml', 'fallback').

        Returns:
            Enriched DataFrame with category and confidence columns.
        """
        if df.empty or description_col not in df.columns:
            out_df = df.copy()
            out_df[category_col] = "Uncategorized"
            out_df[confidence_col] = 0.0
            if include_method:
                out_df["classification_method"] = "fallback"
            return out_df

        out_df = df.copy()
        raw_descriptions = out_df[description_col].fillna("").astype(str).tolist()

        # Step 1: Batch Tier 1 Heuristic Evaluation
        h_results = self.heuristic.classify_batch(raw_descriptions)

        categories: List[str] = [cat for cat, _ in h_results]
        confidences: List[float] = [conf for _, conf in h_results]
        methods: List[str] = [
            "heuristic" if cat != "Uncategorized" and conf >= self.heuristic_threshold else "pending"
            for cat, conf in h_results
        ]

        # Step 2: Identify indices requiring Tier 2 ML inference
        ml_indices = [i for i, m in enumerate(methods) if m == "pending"]

        # Step 3: Batch Tier 2 ML Evaluation (Thread-safe)
        if ml_indices:
            ml_descriptions = [raw_descriptions[i] for i in ml_indices]
            with self._lock:
                ml_results = self.ml.predict_batch(ml_descriptions)

            for idx, (ml_cat, ml_conf) in zip(ml_indices, ml_results):
                if ml_cat != "Uncategorized" and ml_conf >= self.ml_threshold:
                    categories[idx] = ml_cat
                    confidences[idx] = round(ml_conf, 4)
                    methods[idx] = "ml"
                else:
                    categories[idx] = "Uncategorized"
                    confidences[idx] = round(ml_conf, 4) if ml_conf > 0 else 0.0
                    methods[idx] = "fallback"

        # Step 4: Populate enriched DataFrame
        out_df[category_col] = categories
        out_df[confidence_col] = confidences
        if include_method:
            out_df["classification_method"] = methods

        return out_df


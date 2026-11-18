"""Traditional Machine Learning Classification Engine for Bank Transactions.

Combines scikit-learn TfidfVectorizer (n-grams 1-2) with a Multinomial Naive Bayes
or Random Forest classifier to categorize transaction descriptions with probabilistic
confidence scoring without using Large Language Models (LLMs).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from config.settings import CLASSIFICATION_DIR

logger = logging.getLogger(__name__)

# Default model directory and serialization path
SAVED_MODELS_DIR = CLASSIFICATION_DIR / "saved_models"
DEFAULT_MODEL_PATH = SAVED_MODELS_DIR / "classifier.joblib"


class TraditionalMLClassifier:
    """Traditional ML classifier utilizing TF-IDF vectorization and classical classifiers."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        auto_bootstrap: bool = True,
    ) -> None:
        """Initialize ML classifier.

        Args:
            model_path: Optional path to serialized joblib model.
            auto_bootstrap: If True, automatically trains synthetic model if absent.
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.pipeline: Optional[Pipeline] = None
        self._load_or_bootstrap(auto_bootstrap)

    # ==========================================================================
    # Model Loading & Bootstrapping
    # ==========================================================================
    def _load_or_bootstrap(self, auto_bootstrap: bool) -> None:
        """Attempt loading pre-trained model; bootstrap synthetic training if missing."""
        if self.model_path.exists():
            try:
                self.pipeline = joblib.load(self.model_path)
                logger.info("Loaded pre-trained ML classifier from %s", self.model_path)
                return
            except Exception as exc:
                logger.warning("Failed to load model from %s: %s", self.model_path, exc)

        if auto_bootstrap:
            logger.info("No saved model found at %s. Triggering synthetic model training.", self.model_path)
            try:
                from classification.train_synthetic import train_synthetic_model
                self.pipeline = train_synthetic_model(self.model_path)
            except Exception as exc:
                logger.warning("Auto-bootstrap training failed: %s", exc)

    # ==========================================================================
    # Model Training
    # ==========================================================================
    def train(
        self,
        training_data: pd.DataFrame,
        model_path: Optional[Union[str, Path]] = None,
    ) -> Pipeline:
        """Train TF-IDF vectorizer and classifier pipeline on labeled data.

        Args:
            training_data: DataFrame with 'description' and 'category' columns.
            model_path: Optional destination to save trained joblib pipeline.

        Returns:
            Fitted scikit-learn Pipeline.
        """
        if "description" not in training_data.columns or "category" not in training_data.columns:
            raise ValueError("Training data must contain 'description' and 'category' columns.")

        clean_data = training_data.dropna(subset=["description", "category"]).copy()
        clean_data["description"] = clean_data["description"].astype(str).str.strip()
        clean_data = clean_data[clean_data["description"] != ""]

        if len(clean_data) < 5:
            raise ValueError("Training data requires at least 5 valid samples.")

        X = clean_data["description"]
        y = clean_data["category"]

        # Build Pipeline: TF-IDF (1-2 ngrams) + MultinomialNB
        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    lowercase=True,
                    sublinear_tf=True,
                    min_df=1,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
            ("clf", MultinomialNB(alpha=0.1)),
        ])

        pipeline.fit(X, y)
        self.pipeline = pipeline

        # Save to disk
        dest_path = Path(model_path) if model_path else self.model_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, dest_path)
        logger.info("Saved trained ML classifier pipeline to %s", dest_path)

        return pipeline

    # ==========================================================================
    # Prediction Methods
    # ==========================================================================
    def predict(self, description: str) -> Tuple[str, float]:
        """Predict category and confidence score for a single transaction narrative.

        Args:
            description: Narrative text of the transaction.

        Returns:
            Tuple of `(predicted_category, confidence_probability)`.
        """
        if not description or not isinstance(description, str) or not description.strip():
            return "Uncategorized", 0.0

        if self.pipeline is None:
            # Try reloading once
            if self.model_path.exists():
                try:
                    self.pipeline = joblib.load(self.model_path)
                except Exception:
                    pass

        if self.pipeline is None:
            return "Uncategorized", 0.0

        clean_desc = description.strip()
        try:
            probs = self.pipeline.predict_proba([clean_desc])[0]
            classes = self.pipeline.classes_
            max_idx = int(np.argmax(probs))
            predicted_cat = str(classes[max_idx])
            confidence = float(round(probs[max_idx], 4))
            return predicted_cat, confidence
        except Exception as exc:
            logger.debug("Prediction failed for description %r: %s", description, exc)
            return "Uncategorized", 0.0

    def predict_batch(self, descriptions: List[str]) -> List[Tuple[str, float]]:
        """Predict categories for a batch of transaction descriptions."""
        if not descriptions:
            return []

        if self.pipeline is None:
            return [self.predict(d) for d in descriptions]

        clean_descs = [d.strip() if isinstance(d, str) and d.strip() else "" for d in descriptions]

        # Use predict_proba for vector batch efficiency
        try:
            probs_matrix = self.pipeline.predict_proba(clean_descs)
            classes = self.pipeline.classes_
            results: List[Tuple[str, float]] = []

            for idx, probs in enumerate(probs_matrix):
                if not clean_descs[idx]:
                    results.append(("Uncategorized", 0.0))
                else:
                    max_idx = int(np.argmax(probs))
                    results.append((str(classes[max_idx]), float(round(probs[max_idx], 4))))
            return results
        except Exception:
            return [self.predict(d) for d in descriptions]

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        description_col: str = "description",
        output_col: str = "category",
        include_confidence: bool = False,
    ) -> pd.DataFrame:
        """Classify transactions in a pandas DataFrame.

        Args:
            df: Input DataFrame.
            description_col: Column containing transaction descriptions.
            output_col: Column to populate with predicted category.
            include_confidence: Whether to add a 'ml_confidence' column.

        Returns:
            DataFrame with predicted categories.
        """
        if df.empty or description_col not in df.columns:
            out_df = df.copy()
            out_df[output_col] = "Uncategorized"
            if include_confidence:
                out_df["ml_confidence"] = 0.0
            return out_df

        predictions = self.predict_batch(df[description_col].fillna("").astype(str).tolist())

        out_df = df.copy()
        out_df[output_col] = [cat for cat, _ in predictions]
        if include_confidence:
            out_df["ml_confidence"] = [conf for _, conf in predictions]

        return out_df


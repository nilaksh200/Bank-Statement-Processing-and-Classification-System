"""Unit tests for TraditionalMLClassifier and synthetic training pipeline."""

from pathlib import Path
import pytest
import pandas as pd
from sklearn.pipeline import Pipeline

from classification.ml_classifier import TraditionalMLClassifier
from classification.train_synthetic import generate_synthetic_training_data, train_synthetic_model


@pytest.fixture
def classifier() -> TraditionalMLClassifier:
    return TraditionalMLClassifier()


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


# ==============================================================================
# 1. Dataset Generation & Training Tests
# ==============================================================================
def test_synthetic_training_data_generator():
    """Verify generate_synthetic_training_data produces > 200 labeled samples across 8 categories."""
    df = generate_synthetic_training_data(multiplier=2)

    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 200
    assert set(df.columns) == {"description", "category"}

    # Must contain 8 standard categories
    unique_cats = df["category"].nunique()
    assert unique_cats >= 8
    assert "Salary / Income" in df["category"].values
    assert "Groceries & Supermarket" in df["category"].values
    assert "Utilities & Bills" in df["category"].values
    assert "Dining & Food" in df["category"].values
    assert "Transfers & UPI" in df["category"].values
    assert "ATM & Cash" in df["category"].values
    assert "Investment & Insurance" in df["category"].values
    assert "Entertainment & Subscriptions" in df["category"].values


def test_train_and_save_model(temp_dir: Path):
    """Verify that training creates and serializes a scikit-learn Pipeline."""
    model_file = temp_dir / "custom_classifier.joblib"
    df_train = generate_synthetic_training_data(multiplier=1)

    clf = TraditionalMLClassifier(model_path=model_file, auto_bootstrap=False)
    pipeline = clf.train(df_train, model_path=model_file)

    assert isinstance(pipeline, Pipeline)
    assert model_file.exists()
    assert model_file.stat().st_size > 1000  # Non-trivial model file


# ==============================================================================
# 2. Prediction Tests
# ==============================================================================
def test_predict_categories(classifier: TraditionalMLClassifier):
    """Verify probabilistic predictions on realistic merchant descriptions."""
    test_cases = [
        ("WHOLE FOODS ORGANIC VEGETABLES #402", "Groceries & Supermarket"),
        ("CON EDISON ELECTRICITY AUTO-PAY BILL", "Utilities & Bills"),
        ("ACME SYSTEMS DIRECT DEPOSIT SALARY", "Salary / Income"),
        ("STARBUCKS CAFE LATTE STORE #129", "Dining & Food"),
        ("ATM CASH DISPENSE 5TH AVE", "ATM & Cash"),
        ("NETFLIX STREAMING MONTHLY MEMBERSHIP", "Entertainment & Subscriptions"),
        ("VANGUARD INDEX MUTUAL FUND SIP", "Investment & Insurance"),
        ("ZELLE PAYMENT SENT TO FRIEND", "Transfers & UPI"),
    ]

    for desc, expected_cat in test_cases:
        cat, conf = classifier.predict(desc)
        assert cat == expected_cat, f"Expected {expected_cat} for '{desc}', got {cat}"
        assert 0.0 < conf <= 1.0


def test_predict_empty_and_unparseable(classifier: TraditionalMLClassifier):
    """Verify empty or non-string inputs safely resolve to Uncategorized with 0.0 confidence."""
    assert classifier.predict("") == ("Uncategorized", 0.0)
    assert classifier.predict("   ") == ("Uncategorized", 0.0)
    assert classifier.predict(None) == ("Uncategorized", 0.0)


def test_predict_dataframe(classifier: TraditionalMLClassifier):
    """Verify DataFrame batch predictions and probability column inclusion."""
    df = pd.DataFrame([
        {"date": "2026-01-02", "description": "WALMART STORE GROCERY", "debit": 45.0},
        {"date": "2026-01-05", "description": "CITY WATER BILL", "debit": 80.0},
        {"date": "2026-01-08", "description": "MONTHLY SALARY DISBURSEMENT", "credit": 4000.0},
    ])

    pred_df = classifier.predict_dataframe(df, include_confidence=True)

    assert "category" in pred_df.columns
    assert "ml_confidence" in pred_df.columns

    assert pred_df.iloc[0]["category"] == "Groceries & Supermarket"
    assert pred_df.iloc[1]["category"] == "Utilities & Bills"
    assert pred_df.iloc[2]["category"] == "Salary / Income"

    assert all(pred_df["ml_confidence"] > 0.3)


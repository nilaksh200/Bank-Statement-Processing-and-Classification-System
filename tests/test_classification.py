"""Comprehensive Classification Test Suite (Heuristic, ML, and Hybrid Orchestrator).

Tests:
1. HeuristicClassifier against exact sample descriptions ('SALARY CREDIT FOR OCT', 'SWIGGY BANGALORE', 'ATM CASH WITHDRAWAL').
2. TraditionalMLClassifier training and probabilistic inference loop on synthetic test rows.
3. TransactionClassificationEngine on sample transaction DataFrames ensuring non-empty categories and numerical confidence scores.
"""

from pathlib import Path
import pytest
import pandas as pd
from sklearn.pipeline import Pipeline

from classification.rules_engine import HeuristicClassifier
from classification.ml_classifier import TraditionalMLClassifier
from classification.orchestrator import TransactionClassificationEngine
from classification.train_synthetic import generate_synthetic_training_data


# ==============================================================================
# Pytest Fixtures
# ==============================================================================
@pytest.fixture
def heuristic_classifier() -> HeuristicClassifier:
    return HeuristicClassifier()


@pytest.fixture
def ml_classifier() -> TraditionalMLClassifier:
    return TraditionalMLClassifier()


@pytest.fixture
def hybrid_engine() -> TransactionClassificationEngine:
    return TransactionClassificationEngine()


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


# ==============================================================================
# 1. Heuristic Classifier Verification Tests
# ==============================================================================
def test_heuristic_required_sample_descriptions(heuristic_classifier: HeuristicClassifier):
    """Verify exact prompt sample descriptions match target categories with confidence 1.0."""
    # 1. Salary Credit
    cat_salary, conf_salary = heuristic_classifier.classify_description("SALARY CREDIT FOR OCT")
    assert cat_salary == "Salary / Income"
    assert conf_salary == 1.0

    # 2. Food Delivery / Dining
    cat_dining, conf_dining = heuristic_classifier.classify_description("SWIGGY BANGALORE")
    assert cat_dining == "Dining & Food"
    assert conf_dining == 1.0

    # 3. ATM Cash
    cat_atm, conf_atm = heuristic_classifier.classify_description("ATM CASH WITHDRAWAL")
    assert cat_atm == "ATM & Cash"
    assert conf_atm == 1.0


def test_heuristic_additional_standard_categories(heuristic_classifier: HeuristicClassifier):
    """Verify category heuristics across all standard banking buckets."""
    test_cases = [
        ("CITY ELECTRICITY AND WATER BILL AUTO-PAY", "Utilities & Bills"),
        ("WALMART SUPERCENTER STORE #1042", "Groceries & Supermarket"),
        ("TRADER JOE'S GROCERY", "Groceries & Supermarket"),
        ("STARBUCKS CAFE LATTE STORE #342", "Dining & Food"),
        ("UPI / 402918239102 / PAYMENT TO VENDOR", "Transfers & UPI"),
        ("ZELLE TRANSFER TO JOHN DOE", "Transfers & UPI"),
        ("ZERODHA BROKING MUTUAL FUND SIP", "Investment & Insurance"),
        ("LIC LIFE INSURANCE ANNUAL POLICY", "Investment & Insurance"),
        ("APARTMENT LEASE RENT PAYMENT", "Rent & Housing"),
        ("NETFLIX STREAMING SUBSCRIPTION", "Entertainment & Subscriptions"),
        ("UBER RIDE SAN FRANCISCO", "Travel & Transportation"),
        ("CVS PHARMACY PRESCRIPTION", "Health & Medical"),
        ("MONTHLY ACCOUNT SERVICE FEE", "Bank Fees & Charges"),
    ]

    for desc, expected_cat in test_cases:
        cat, conf = heuristic_classifier.classify_description(desc)
        assert cat == expected_cat, f"Description '{desc}' categorized as '{cat}' instead of '{expected_cat}'"
        assert conf == 1.0


def test_heuristic_uncategorized_and_empty(heuristic_classifier: HeuristicClassifier):
    """Verify unmatched and blank descriptions return Uncategorized with 0.0 confidence."""
    assert heuristic_classifier.classify_description("RANDOM OBSCURE NARRATIVE 99482") == ("Uncategorized", 0.0)
    assert heuristic_classifier.classify_description("") == ("Uncategorized", 0.0)
    assert heuristic_classifier.classify_description("   ") == ("Uncategorized", 0.0)
    assert heuristic_classifier.classify_description(None) == ("Uncategorized", 0.0)


# ==============================================================================
# 2. Traditional ML Classifier Training & Inference Tests
# ==============================================================================
def test_traditional_ml_training_and_inference_loop(temp_dir: Path):
    """Verify TraditionalMLClassifier training loop, model saving, and probabilistic prediction."""
    # 1. Create synthetic training rows
    sample_training_data = pd.DataFrame([
        {"description": "PAYROLL DIRECT DEP ACME CORP", "category": "Salary / Income"},
        {"description": "MONTHLY SALARY PAYMENT CORP", "category": "Salary / Income"},
        {"description": "CITY ELECTRIC UTILITY BILL", "category": "Utilities & Bills"},
        {"description": "WATER DEPARTMENT MONTHLY BILL", "category": "Utilities & Bills"},
        {"description": "WHOLE FOODS GROCERY MARKET", "category": "Groceries & Supermarket"},
        {"description": "KROGER FRESH FOOD GROCERIES", "category": "Groceries & Supermarket"},
        {"description": "STARBUCKS COFFEE CAFE SHOP", "category": "Dining & Food"},
        {"description": "MCDONALD'S DRIVE-THRU BURGER", "category": "Dining & Food"},
        {"description": "ATM CASH WITHDRAWAL MAIN ST", "category": "ATM & Cash"},
        {"description": "CHASE ATM DISPENSE CASH W/D", "category": "ATM & Cash"},
    ])

    model_path = temp_dir / "test_model.joblib"
    clf = TraditionalMLClassifier(model_path=model_path, auto_bootstrap=False)

    # 2. Train on sample rows
    fitted_pipeline = clf.train(sample_training_data, model_path=model_path)
    assert isinstance(fitted_pipeline, Pipeline)
    assert model_path.exists()

    # 3. Test Inference
    cat, conf = clf.predict("ACME CORP SALARY")
    assert cat == "Salary / Income"
    assert 0.0 < conf <= 1.0

    cat2, conf2 = clf.predict("ELECTRIC BILL")
    assert cat2 == "Utilities & Bills"
    assert 0.0 < conf2 <= 1.0

    cat3, conf3 = clf.predict("WHOLE FOODS")
    assert cat3 == "Groceries & Supermarket"
    assert 0.0 < conf3 <= 1.0


def test_traditional_ml_full_synthetic_dataset_inference(ml_classifier: TraditionalMLClassifier):
    """Verify pre-trained traditional ML classifier accurately categorizes unseen descriptions."""
    test_samples = [
        ("KROGER FOOD AND PHARMACY GROCERIES #88", "Groceries & Supermarket"),
        ("PACIFIC GAS & ELECTRIC UTILITY BILL", "Utilities & Bills"),
        ("DISBURSEMENT OF SALARY TECHCORP", "Salary / Income"),
        ("ATM DISPENSE CASH WITHDRAWAL", "ATM & Cash"),
        ("SPOTIFY MUSIC MONTHLY SUBSCRIPTION", "Entertainment & Subscriptions"),
    ]

    for desc, expected in test_samples:
        pred_cat, pred_conf = ml_classifier.predict(desc)
        assert pred_cat == expected, f"For '{desc}', expected {expected}, got {pred_cat}"
        assert 0.0 < pred_conf <= 1.0


# ==============================================================================
# 3. TransactionClassificationEngine Hybrid Pipeline Tests
# ==============================================================================
def test_hybrid_engine_dataframe_classification(hybrid_engine: TransactionClassificationEngine):
    """Verify that all rows in a DataFrame receive a valid non-empty category and numerical confidence."""
    test_df = pd.DataFrame([
        {"date": "2026-01-02", "description": "WALMART SUPERCENTER STORE #1042", "debit": 85.50, "credit": 0.0, "balance": 4914.50},
        {"date": "2026-01-05", "description": "CON EDISON ELECTRICITY AUTO-PAY", "debit": 142.20, "credit": 0.0, "balance": 4772.30},
        {"date": "2026-01-08", "description": "SWIGGY BANGALORE RESTAURANT", "debit": 32.00, "credit": 0.0, "balance": 4740.30},
        {"date": "2026-01-10", "description": "ATM CASH WITHDRAWAL DOWNTOWN", "debit": 200.00, "credit": 0.0, "balance": 4540.30},
        {"date": "2026-01-15", "description": "SALARY CREDIT FOR OCT ACME CORP", "debit": 0.0, "credit": 4500.00, "balance": 9040.30},
        {"date": "2026-01-18", "description": "RANDOM TOKEN XZ99834 UNKNOWN", "debit": 15.00, "credit": 0.0, "balance": 9025.30},
    ])

    enriched_df = hybrid_engine.classify_transactions(test_df)

    # 1. Check required schema columns
    assert "category" in enriched_df.columns
    assert "confidence" in enriched_df.columns
    assert len(enriched_df) == len(test_df)

    # 2. Verify all categories are non-empty strings
    assert enriched_df["category"].isna().sum() == 0
    assert (enriched_df["category"] == "").sum() == 0
    for cat in enriched_df["category"]:
        assert isinstance(cat, str)
        assert len(cat.strip()) > 0

    # 3. Verify all confidence scores are numerical floats in [0.0, 1.0]
    assert pd.api.types.is_float_dtype(enriched_df["confidence"])
    assert enriched_df["confidence"].isna().sum() == 0
    for conf in enriched_df["confidence"]:
        assert 0.0 <= conf <= 1.0

    # 4. Check specific row classifications
    assert enriched_df.iloc[0]["category"] == "Groceries & Supermarket"
    assert enriched_df.iloc[0]["confidence"] >= 0.40

    assert enriched_df.iloc[1]["category"] == "Utilities & Bills"
    assert enriched_df.iloc[1]["confidence"] == 1.0

    assert enriched_df.iloc[2]["category"] == "Dining & Food"
    assert enriched_df.iloc[2]["confidence"] == 1.0

    assert enriched_df.iloc[3]["category"] == "ATM & Cash"
    assert enriched_df.iloc[3]["confidence"] == 1.0

    assert enriched_df.iloc[4]["category"] == "Salary / Income"
    assert enriched_df.iloc[4]["confidence"] == 1.0

    assert enriched_df.iloc[5]["category"] == "Uncategorized"


def test_hybrid_engine_empty_and_missing_dataframe(hybrid_engine: TransactionClassificationEngine):
    """Verify hybrid engine handles empty DataFrames safely without exceptions."""
    empty_df = pd.DataFrame(columns=["date", "description", "debit", "credit", "balance"])
    result = hybrid_engine.classify_transactions(empty_df)

    assert "category" in result.columns
    assert "confidence" in result.columns
    assert len(result) == 0

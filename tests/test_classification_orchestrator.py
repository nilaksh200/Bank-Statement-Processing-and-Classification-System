"""Unit tests for TransactionClassificationEngine hybrid dual-layer orchestrator."""

from concurrent.futures import ThreadPoolExecutor
import time
import pytest
import pandas as pd

from classification.orchestrator import TransactionClassificationEngine
from classification.rules_engine import HeuristicClassifier
from classification.ml_classifier import TraditionalMLClassifier


@pytest.fixture
def engine() -> TransactionClassificationEngine:
    return TransactionClassificationEngine()


# ==============================================================================
# 1. Hybrid Routing Logic Tests
# ==============================================================================
def test_hybrid_routing_heuristic_tier(engine: TransactionClassificationEngine):
    """Verify regex/keyword descriptions are matched via heuristic Tier 1."""
    cat, conf, method = engine.classify_single("MONTHLY PAYROLL DIRECT DEPOSIT ACME CORP")
    assert cat == "Salary / Income"
    assert conf == 1.0
    assert method == "heuristic"


def test_hybrid_routing_ml_tier(engine: TransactionClassificationEngine):
    """Verify descriptions without exact regex keywords fall back to ML Tier 2."""
    # Description without 'mart', 'supermarket', 'grocery', etc., but has strong grocery tokens
    cat, conf, method = engine.classify_single("SPROUTS FARMERS MARKET PRODUCE SOMA")
    assert cat == "Groceries & Supermarket"
    assert conf >= 0.40
    assert method == "ml"


def test_hybrid_routing_fallback_uncategorized(engine: TransactionClassificationEngine):
    """Verify completely ambiguous or random descriptions resolve to Uncategorized."""
    cat, conf, method = engine.classify_single("XZYQ7782 RANDOM STRING 9901")
    assert cat == "Uncategorized"
    assert method == "fallback"


# ==============================================================================
# 2. DataFrame Batch Processing Tests
# ==============================================================================
def test_classify_transactions_dataframe(engine: TransactionClassificationEngine):
    """Verify DataFrame batch classification, column additions, and threshold gating."""
    df = pd.DataFrame([
        {"date": "2026-01-02", "description": "WALMART SUPERCENTER STORE #1042", "debit": 85.50},
        {"date": "2026-01-05", "description": "CON EDISON ELECTRICITY BILL", "debit": 120.00},
        {"date": "2026-01-08", "description": "RANDOM TOKEN XZ99834 UNKNOWN", "debit": 10.00},
        {"date": "2026-01-12", "description": "DIRECT DEP SALARY DISBURSEMENT", "credit": 3500.00},
    ])

    result_df = engine.classify_transactions(df, include_method=True)

    assert "category" in result_df.columns
    assert "confidence" in result_df.columns
    assert "classification_method" in result_df.columns

    # Row 0: Grocery
    assert result_df.iloc[0]["category"] == "Groceries & Supermarket"
    assert result_df.iloc[0]["confidence"] >= 0.40

    # Row 1: Utilities
    assert result_df.iloc[1]["category"] == "Utilities & Bills"
    assert result_df.iloc[1]["confidence"] == 1.0
    assert result_df.iloc[1]["classification_method"] == "heuristic"

    # Row 2: Uncategorized fallback
    assert result_df.iloc[2]["category"] == "Uncategorized"
    assert result_df.iloc[2]["classification_method"] == "fallback"

    # Row 3: Salary
    assert result_df.iloc[3]["category"] == "Salary / Income"
    assert result_df.iloc[3]["confidence"] == 1.0


# ==============================================================================
# 3. High-Throughput & Thread-Safety Tests
# ==============================================================================
def test_batch_throughput_benchmark(engine: TransactionClassificationEngine):
    """Verify batch processing over 1,000 transaction rows executes rapidly."""
    base_rows = [
        {"description": "WALMART GROCERY MART", "debit": 50.0},
        {"description": "PACIFIC GAS ELECTRIC BILL", "debit": 100.0},
        {"description": "STARBUCKS CAFE COFFEE", "debit": 7.50},
        {"description": "ATM CASH WITHDRAWAL #12", "debit": 200.0},
        {"description": "ACME DIRECT DEP PAYROLL", "credit": 2500.0},
    ]

    # Repeat 200 times to create 1,000 rows
    large_df = pd.DataFrame(base_rows * 200)
    assert len(large_df) == 1000

    start_time = time.perf_counter()
    classified_df = engine.classify_transactions(large_df)
    duration = time.perf_counter() - start_time

    assert len(classified_df) == 1000
    assert "category" in classified_df.columns
    # 1,000 transactions should process in well under 1.5 seconds with vectorized inference
    assert duration < 2.0, f"Batch processing took {duration:.2f}s, expected < 2.0s"


def test_thread_safety_concurrent_execution(engine: TransactionClassificationEngine):
    """Verify thread-safety when multiple worker threads execute concurrent classifications."""
    descriptions = [
        "WALMART SUPERCENTER",
        "STARBUCKS CAFE",
        "CHASE ATM CASH WITHDRAWAL",
        "CON EDISON ELECTRIC BILL",
        "SALARY DIRECT DEPOSIT",
    ]

    def worker_task(thread_id: int):
        df = pd.DataFrame({"description": descriptions * 10})
        out = engine.classify_transactions(df)
        assert len(out) == 50
        assert out["category"].isna().sum() == 0
        return True

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker_task, i) for i in range(5)]
        results = [f.result() for f in futures]

    assert all(results)


"""Classification module: Non-LLM rule-based heuristics and traditional scikit-learn ML categorization."""

from .rules_engine import HeuristicClassifier, DEFAULT_RULE_PATTERNS
from .ml_classifier import TraditionalMLClassifier, DEFAULT_MODEL_PATH
from .train_synthetic import generate_synthetic_training_data, train_synthetic_model
from .orchestrator import TransactionClassificationEngine

__all__ = [
    "HeuristicClassifier",
    "TraditionalMLClassifier",
    "TransactionClassificationEngine",
    "DEFAULT_RULE_PATTERNS",
    "DEFAULT_MODEL_PATH",
    "generate_synthetic_training_data",
    "train_synthetic_model",
]

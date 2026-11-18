"""Global configuration and settings for Bank Statement Processing & Classification System.

Defines:
- Filesystem paths and directories
- Supported file types and size limits
- Standard output schema columns
- OCR fallback configurations (EasyOCR / PyTesseract)
- Default transaction category definitions
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# ==============================================================================
# Path Configurations
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent.parent
CONFIG_DIR: Path = BASE_DIR / "config"
INGESTION_DIR: Path = BASE_DIR / "ingestion"
EXTRACTION_DIR: Path = BASE_DIR / "extraction"
CLASSIFICATION_DIR: Path = BASE_DIR / "classification"
EXPORT_DIR: Path = BASE_DIR / "export"
UI_DIR: Path = BASE_DIR / "ui"
TESTS_DIR: Path = BASE_DIR / "tests"

# Runtime & data directories
DATA_DIR: Path = BASE_DIR / "data"
TEMP_DIR: Path = DATA_DIR / "temp"
EXPORTS_DIR: Path = DATA_DIR / "exports"
MODELS_DIR: Path = CLASSIFICATION_DIR / "models"

# Ensure runtime directories exist
for path in (DATA_DIR, TEMP_DIR, EXPORTS_DIR, MODELS_DIR):
    path.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# File Ingestion & Validation Settings
# ==============================================================================
SUPPORTED_FILE_EXTENSIONS: List[str] = [".pdf"]
SUPPORTED_MIME_TYPES: List[str] = ["application/pdf"]
MAX_FILE_SIZE_MB: int = 50

# Threshold: Minimum characters extracted per page before triggering OCR fallback
SCANNED_PAGE_CHAR_THRESHOLD: int = 50

# ==============================================================================
# Standard Schema Column Definitions
# ==============================================================================
STANDARD_SCHEMA_COLUMNS: List[str] = [
    "date",
    "description",
    "debit",
    "credit",
    "balance",
    "category",
]

# Type casting specifications for canonical DataFrame
COLUMN_DTYPES: Dict[str, str] = {
    "date": "string",
    "description": "string",
    "debit": "float64",
    "credit": "float64",
    "balance": "float64",
    "category": "string",
}

# ==============================================================================
# OCR Fallback Configurations
# ==============================================================================
OCR_SETTINGS: Dict[str, Any] = {
    "enabled": True,
    # Preferred OCR engine: 'easyocr' (pure-python / pytorch) or 'pytesseract' (requires system tesseract-ocr)
    "default_engine": os.getenv("DEFAULT_OCR_ENGINE", "easyocr"),
    # Languages for EasyOCR
    "easyocr_languages": ["en"],
    # System path to tesseract binary if not in system PATH
    "tesseract_cmd": os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe" if os.name == "nt" else "/usr/bin/tesseract"),
    # Resolution (DPI) when rasterizing PDF pages for OCR
    "render_dpi": 300,
    # Preprocessing options for image enhancement before OCR
    "enable_preprocessing": True,
    "contrast_enhancement": True,
    "binarization_threshold": 180,
}

# ==============================================================================
# Default Transaction Categories (Heuristics & ML Target Classes)
# ==============================================================================
DEFAULT_CATEGORIES: List[str] = [
    "Groceries & Food",
    "Dining & Restaurants",
    "Utilities & Bills",
    "Rent & Housing",
    "Salary & Income",
    "Transfers & Payments",
    "Shopping & Retail",
    "Entertainment & Subscriptions",
    "Health & Medical",
    "Travel & Transportation",
    "Investment & Savings",
    "Bank Fees & Charges",
    "Uncategorized",
]

# ==============================================================================
# Strict Non-LLM Architecture Policy
# ==============================================================================
ENFORCE_NON_LLM: bool = True
FORBIDDEN_PACKAGES: List[str] = [
    "openai",
    "langchain",
    "anthropic",
    "google-generativeai",
    "ollama",
    "llama_cpp",
    "litellm",
    "transformers",
    "huggingface_hub",
]


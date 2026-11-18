# Bank Statement Processing & Classification System

An automated, privacy-first bank statement processing pipeline and interactive web dashboard. Built with a modular multi-subagent architecture, this system ingests PDF bank statements, detects document layout (vector text vs. scanned raster images), extracts structured tabular data with OCR fallback, classifies transactions using deterministic heuristics and traditional machine learning, and exports validated records to CSV and styled multi-tab Excel workbooks.

---

## 🔒 Strict Non-LLM Architecture

> [!IMPORTANT]
> **Zero LLM Dependency**: This system strictly forbids the import or invocation of Large Language Models (OpenAI, LangChain, Anthropic, Ollama, etc.) for extraction or classification. All categorization relies solely on regular expression heuristics and traditional `scikit-learn` machine learning models (TF-IDF Vectorization + Naïve Bayes / Linear Classifiers). All processing occurs **100% on-device** ensuring maximum financial data privacy.

---

## 🚀 Quick-Start Guide for Reviewers

Follow these end-to-end steps to set up and run the system:

### 1. Environment Setup
Clone or unzip the project repository and install required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Initialize Model
Bootstrap and train the traditional machine learning classifier pipeline using 360+ synthetic labeled bank transactions across 8 standard categories:
```bash
python classification/train_synthetic.py
```
*Note: This generates `classification/saved_models/classifier.joblib`.*

### 3. Run Test Suite
Execute the comprehensive automated test suite (54+ unit and integration tests across ingestion, extraction, classification, export, and UI):
```bash
pytest tests/
```
Or for verbose test logging:
```bash
pytest tests/ -v
```

### 4. Launch Web UI
Start the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
Open your web browser at `http://localhost:8501`.

---

## 🔍 Note on OCR Configuration

The pipeline supports two OCR backends for scanned and image-based PDF statements:

- **EasyOCR (Default / Recommended)**:
  - **Works out-of-the-box without system installs.**
  - Pure-Python PyTorch pipeline that downloads optical character recognition weights automatically on first run.
- **Tesseract OCR (`pytesseract`)**:
  - Optional secondary fallback.
  - Can be selected from the web UI sidebar if Tesseract OCR is installed on your host system:
    - **Windows**: Install [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) (default: `C:\Program Files\Tesseract-OCR\tesseract.exe`).
    - **Linux**: `sudo apt install tesseract-ocr`
    - **macOS**: `brew install tesseract`

---

## 🏛️ Subagent Architecture

The system is organized into four specialized subagents:

1. **`IngestionSubAgent`** (`ingestion/`):
   - File intake, MIME type validation, file integrity verification, and structural PDF type detection.
   - Evaluates selectable character count (`< 50` threshold) and visual image area coverage (`> 80%` threshold) to categorize documents into `text_based` vs `image_based`.
2. **`ExtractionSubAgent`** (`extraction/`):
   - **`TextStatementParser`**: Layout-aware vector parsing with `pdfplumber` and `PyMuPDF` (`fitz`), header metadata extraction (Account Holder, Account No, IFSC/Routing Code, Statement Period), multi-line narrative merging, and single-amount column splitting (`DR`/`CR`/`+`/`-`).
   - **`OCRStatementParser`**: 300 DPI high-resolution page rendering, Pillow pre-processing (grayscale, contrast enhancement, sharpening, binarization), and spatial 2D table reconstruction from bounding boxes.
   - **`ExtractionOrchestrator` & `DataSanitiser`**: Dynamic routing and data standardization (`YYYY-MM-DD` ISO dates, normalized 2-decimal floats).
3. **`ClassificationSubAgent`** (`classification/`):
   - **`HeuristicClassifier`**: Deterministic keyword and regex rule engine loaded from `config/categories.json` for instant $1.0$ confidence classification.
   - **`TraditionalMLClassifier`**: `scikit-learn` Pipeline combining `TfidfVectorizer(ngram_range=(1, 2))` with `MultinomialNB(alpha=0.1)` for probabilistic categorization.
   - **`TransactionClassificationEngine`**: Thread-safe hybrid two-tier orchestrator providing high-throughput batch evaluation ($1,000+$ rows in $<0.1$s) with confidence gating ($<0.4$ fallback to `Uncategorized`).
4. **`UISubAgent`** (`ui/`, `export/`, `app.py`):
   - **Streamlit Web Dashboard**: Document type badge, metadata expander, metric cards (Total Debits, Total Credits, Net Cash Flow, Counts), inline interactive `st.data_editor` for manual category overrides, and interactive Plotly spend analytics (horizontal bar chart and distribution pie chart).
   - **`StatementExporter`**: Single-click downloads for clean CSV and professionally styled multi-tab Excel workbooks:
     - **Tab 1 ("Transactions")**: Navy headers (`#1F4E79`), alternating zebra rows, currency formatting (`$#,##0.00`), and soft conditional formatting fills (light green for credits, light red for debits).
     - **Tab 2 ("Category Summary")**: Pivot spend table with counts and percentage breakdown.
     - **Tab 3 ("Account Metadata")**: Account headers and high-level financial metrics.

---

## 📁 Repository Architecture

```text
befree/
├── .antigravityrules         # Master project directives and constraints
├── requirements.txt          # Production Python dependencies
├── README.md                 # Complete system guide & setup instructions
├── app.py                    # Primary Streamlit web application
│
├── config/                   # Global configuration & category rules
│   ├── __init__.py
│   ├── settings.py           # Paths, schema columns, OCR parameters
│   └── categories.json       # Category definitions and regex patterns
│
├── ingestion/                # IngestionSubAgent: PDF detection & validation
│   ├── __init__.py
│   └── detector.py           # PDFTypeDetector (text_based vs image_based)
│
├── extraction/               # ExtractionSubAgent: Layout parsers & OCR
│   ├── __init__.py
│   ├── text_parser.py        # Native vector text & grid table extractor
│   ├── ocr_parser.py         # 300 DPI rasterizer & spatial table reconstructor
│   ├── sanitiser.py          # ISO date & currency sanitiser
│   └── orchestrator.py       # Unified extraction pipeline orchestrator
│
├── classification/           # ClassificationSubAgent: Dual-layer non-LLM engine
│   ├── __init__.py
│   ├── rules_engine.py       # Tier 1 regex & keyword heuristics
│   ├── ml_classifier.py      # Tier 2 scikit-learn TF-IDF + Naive Bayes pipeline
│   ├── train_synthetic.py    # Synthetic bank description generator & trainer
│   ├── orchestrator.py       # Hybrid dual-tier classification orchestrator
│   └── saved_models/         # Serialized classifier model (.joblib)
│
├── export/                   # UISubAgent: Formatted Excel & CSV exporters
│   ├── __init__.py
│   └── exporter.py           # Multi-tab styled Excel and CSV exporter
│
├── ui/                       # UISubAgent: Streamlit interface package
│   ├── __init__.py
│   └── app.py                # Dashboard entrypoint wrapper
│
└── tests/                    # Complete automated pytest test suite
    ├── __init__.py
    ├── test_config.py        # Settings and schema validation tests
    ├── test_detector.py      # PDFTypeDetector tests
    ├── test_extraction.py    # Native extraction and sanitiser tests
    ├── test_ocr_parser.py    # OCR rasterization and reconstruction tests
    ├── test_orchestrator.py  # Extraction orchestrator integration tests
    ├── test_classification.py # Heuristic and ML classification tests
    ├── test_classification_orchestrator.py # Hybrid classification benchmark tests
    ├── test_ml_classifier.py # ML classifier training and prediction tests
    ├── test_export.py        # Styled Excel and CSV export tests
    ├── test_ui.py            # Streamlit pipeline helper tests
    └── test_ui_export.py     # UI integration & export verification tests
```

---

## 📋 Standard Output Schema

All extracted and parsed bank transactions conform to the canonical 6-column schema:

| Column | Type | Description |
| :--- | :--- | :--- |
| `date` | `string` | Transaction date normalized to `YYYY-MM-DD` |
| `description` | `string` | Cleaned transaction description / merchant narrative |
| `debit` | `float64` | Withdrawal / payment amount (or `0.00`) |
| `credit` | `float64` | Deposit / refund amount (or `0.00`) |
| `balance` | `float64` | Running account balance after transaction |
| `category` | `string` | Assigned expense/income category |

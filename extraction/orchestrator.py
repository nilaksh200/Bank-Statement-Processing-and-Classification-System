"""Unified Extraction Pipeline Orchestrator.

Orchestrates statement processing by detecting PDF document representation
(native text vs scanned image), routing to the appropriate parser
(TextStatementParser or OCRStatementParser), and sanitising all extracted records.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from extraction.ocr_parser import OCRStatementParser
from extraction.sanitiser import sanitise_dataframe
from extraction.text_parser import TRANSACTION_COLUMNS, TextStatementParser
from ingestion.detector import PDFTypeDetector, PDFDetectionError

logger = logging.getLogger(__name__)


class ExtractionOrchestrator:
    """Unified coordinator for bank statement ingestion and transaction extraction.

    Workflow:
    1. Detect PDF representation via PDFTypeDetector ('text_based' vs 'image_based').
    2. Extract header metadata (account holder, account number, routing code, period).
    3. Route to TextStatementParser (for native vector text) or OCRStatementParser (for scans).
    4. Automatically fallback to OCR if native parser extracts zero transactions.
    5. Clean and normalize data using sanitiser.py into standard schema.
    """

    def __init__(
        self,
        detector: Optional[PDFTypeDetector] = None,
        text_parser: Optional[TextStatementParser] = None,
        ocr_parser: Optional[OCRStatementParser] = None,
    ) -> None:
        """Initialize orchestrator with sub-parsers and detector.

        Args:
            detector: Optional instance of PDFTypeDetector.
            text_parser: Optional instance of TextStatementParser.
            ocr_parser: Optional instance of OCRStatementParser.
        """
        self.detector = detector or PDFTypeDetector()
        self.text_parser = text_parser or TextStatementParser()
        self.ocr_parser = ocr_parser or OCRStatementParser()

    def process_statement(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a bank statement PDF and extract structured, sanitised transactions.

        Args:
            pdf_path: Path to the target bank statement PDF.

        Returns:
            Dictionary matching specification:
            {
                "metadata": dict,
                "transactions": pd.DataFrame,
                "pdf_type": "text_based" | "image_based",
                "status": "success" | "error"
            }
        """
        path = Path(pdf_path).resolve()

        if not path.exists():
            return {
                "metadata": {},
                "transactions": pd.DataFrame(columns=TRANSACTION_COLUMNS),
                "pdf_type": "unknown",
                "status": "error",
                "error_message": f"File not found: {path}",
            }

        try:
            # Step 1: Detect PDF structure & type
            detection_info = self.detector.detect_pdf_type(path)
            pdf_type = detection_info.get("pdf_type", "text_based")
            logger.info("PDF %s classified as '%s' (confidence: %.2f)", path.name, pdf_type, detection_info.get("confidence_score", 0.0))

            # Step 2: Extract header metadata
            metadata = self.text_parser.extract_metadata(path)

            # Step 3: Route extraction based on PDF type
            raw_df: pd.DataFrame
            if pdf_type == "text_based":
                raw_df = self.text_parser.extract_transactions(path)
                # Fallback to OCR if text extraction yields no transaction records
                if raw_df.empty:
                    logger.info("Text parser yielded 0 transactions for %s. Triggering OCR fallback.", path.name)
                    raw_df = self.ocr_parser.extract_transactions(path)
            else:
                raw_df = self.ocr_parser.extract_transactions(path)

            # Step 4: Sanitise and normalize extracted records
            sanitised_df = sanitise_dataframe(raw_df)

            return {
                "metadata": metadata,
                "transactions": sanitised_df,
                "pdf_type": pdf_type,
                "status": "success",
            }

        except PDFDetectionError as exc:
            logger.error("Ingestion/detection error for %s: %s", path.name, exc)
            return {
                "metadata": {},
                "transactions": pd.DataFrame(columns=TRANSACTION_COLUMNS),
                "pdf_type": "unknown",
                "status": "error",
                "error_message": str(exc),
            }
        except Exception as exc:
            logger.exception("Unexpected error processing statement %s: %s", path.name, exc)
            return {
                "metadata": {},
                "transactions": pd.DataFrame(columns=TRANSACTION_COLUMNS),
                "pdf_type": "unknown",
                "status": "error",
                "error_message": str(exc),
            }


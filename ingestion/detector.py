"""PDF Type Auto-Detector Module.

Evaluates bank statement PDFs for selectable text density, character counts,
and image bounds coverage to determine whether a document is native text-based
or a rasterized/scanned image-based document requiring OCR fallback.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore

import pdfplumber

from config.settings import SCANNED_PAGE_CHAR_THRESHOLD

logger = logging.getLogger(__name__)


# ==============================================================================
# Custom Exceptions for PDF Ingestion
# ==============================================================================
class PDFDetectionError(Exception):
    """Base exception for PDF detection failures."""


class PDFPasswordProtectedError(PDFDetectionError):
    """Raised when the target PDF is encrypted or password-protected."""


class PDFCorruptedError(PDFDetectionError):
    """Raised when the target PDF is corrupted, malformed, or unreadable."""


class PDFEmptyError(PDFDetectionError):
    """Raised when the target PDF contains zero pages or is empty."""


# ==============================================================================
# PDF Type Detector Class
# ==============================================================================
class PDFTypeDetector:
    """Detects whether a PDF bank statement is native text-based or scanned/image-based.

    Uses a hybrid analysis combining PyMuPDF (`fitz`) and `pdfplumber` to inspect:
    - Selectable character count per page
    - Text character density
    - Visual image bounding boxes relative to total page surface area

    Classification Criteria:
    - Page is classified as 'image_based' if selectable chars < 50 OR image coverage > 80%.
    - Document is classified as 'image_based' if >= 50% of pages are image_based.
    - Otherwise classified as 'text_based'.
    """

    def __init__(
        self,
        min_char_count: int = SCANNED_PAGE_CHAR_THRESHOLD,
        max_image_area_ratio: float = 0.80,
    ) -> None:
        """Initialize detector with detection thresholds.

        Args:
            min_char_count: Minimum selectable characters per page before triggering OCR fallback.
            max_image_area_ratio: Maximum proportion of page area (0.0 to 1.0) covered by images
                                  before flagging the page as image-based/scanned.
        """
        self.min_char_count = min_char_count
        self.max_image_area_ratio = max_image_area_ratio

    def detect_pdf_type(
        self,
        pdf_path: Union[str, Path],
        include_page_details: bool = False,
    ) -> Dict[str, Any]:
        """Analyze a PDF document and classify its representation type.

        Args:
            pdf_path: Path to the target PDF file.
            include_page_details: If True, includes granular per-page breakdown in output.

        Returns:
            Dictionary matching the required schema:
            {
                "pdf_type": "text_based" | "image_based",
                "page_count": int,
                "has_selectable_text": bool,
                "confidence_score": float,
                # Optional extended detail if requested:
                "page_details": [...]
            }

        Raises:
            FileNotFoundError: If the file does not exist.
            PDFPasswordProtectedError: If the document is password-protected or encrypted.
            PDFCorruptedError: If the document is corrupted or malformed.
            PDFEmptyError: If the document contains 0 pages or has 0 bytes.
        """
        path = Path(pdf_path).resolve()
        self._validate_file_path(path)

        # 1. Structural inspection & PyMuPDF analysis
        fitz_doc = self._open_with_fitz(path)
        try:
            total_pages = len(fitz_doc)
            if total_pages == 0:
                raise PDFEmptyError(f"PDF contains 0 pages: {path}")

            # 2. Extract per-page metrics using both PyMuPDF and pdfplumber
            page_metrics: List[Dict[str, Any]] = self._evaluate_pages(path, fitz_doc)
        finally:
            fitz_doc.close()

        # 3. Aggregate document-level classifications
        return self._aggregate_results(page_metrics, include_page_details=include_page_details)

    # --------------------------------------------------------------------------
    # Internal Validation & File Handling
    # --------------------------------------------------------------------------
    def _validate_file_path(self, path: Path) -> None:
        """Verify file existence and basic validity."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise PDFCorruptedError(f"Path is not a regular file: {path}")

        if path.stat().st_size == 0:
            raise PDFEmptyError(f"File is 0 bytes (empty): {path}")

    def _open_with_fitz(self, path: Path) -> fitz.Document:
        """Open PDF with PyMuPDF with corruption and encryption checks."""
        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            logger.error("PyMuPDF failed to open %s: %s", path, exc)
            raise PDFCorruptedError(f"Corrupted or invalid PDF format: {path}") from exc

        if doc.is_encrypted or doc.needs_pass:
            doc.close()
            raise PDFPasswordProtectedError(f"File is password-protected or encrypted: {path}")

        return doc

    # --------------------------------------------------------------------------
    # Per-Page Layout and Character Analysis
    # --------------------------------------------------------------------------
    def _evaluate_pages(self, path: Path, fitz_doc: fitz.Document) -> List[Dict[str, Any]]:
        """Evaluate text density and image coverage for each page."""
        metrics: List[Dict[str, Any]] = []

        # Open pdfplumber session for corroborative layout and character inspection
        plumber_doc: Optional[pdfplumber.PDF] = None
        try:
            plumber_doc = pdfplumber.open(str(path))
        except Exception as exc:
            logger.warning("pdfplumber could not open %s (proceeding with fitz only): %s", path, exc)

        try:
            for page_idx in range(len(fitz_doc)):
                fitz_page = fitz_doc[page_idx]

                # PyMuPDF character extraction
                fitz_text = fitz_page.get_text() or ""
                fitz_char_count = len(fitz_text.strip())

                # pdfplumber corroboration (if available)
                plumber_char_count = 0
                if plumber_doc and page_idx < len(plumber_doc.pages):
                    try:
                        p_page = plumber_doc.pages[page_idx]
                        plumber_text = p_page.extract_text() or ""
                        plumber_char_count = len(plumber_text.strip())
                    except Exception:
                        plumber_char_count = 0

                # Max character count between both parsers ensures robust detection
                char_count = max(fitz_char_count, plumber_char_count)

                # Compute image bounding boxes coverage ratio
                image_ratio = self._calculate_image_coverage_ratio(fitz_page)

                # Page-level classification rule:
                # If char_count < min_char_count OR image_ratio > max_image_area_ratio => image_based
                is_image_based = (
                    char_count < self.min_char_count
                    or image_ratio > self.max_image_area_ratio
                )

                metrics.append({
                    "page_index": page_idx,
                    "char_count": char_count,
                    "image_area_ratio": round(image_ratio, 4),
                    "page_classification": "image_based" if is_image_based else "text_based",
                    "has_text": char_count > 0,
                })

        finally:
            if plumber_doc:
                try:
                    plumber_doc.close()
                except Exception:
                    pass

        return metrics

    def _calculate_image_coverage_ratio(self, page: fitz.Page) -> float:
        """Calculate total bounding box area of images relative to total page area."""
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height

        if page_area <= 0:
            return 0.0

        total_image_area = 0.0
        try:
            # Query all placed image rectangles on this page
            image_info = page.get_image_info(xrefs=True)
            for img in image_info:
                bbox = img.get("bbox")
                if not bbox:
                    continue
                # Clip to page boundary
                img_rect = fitz.Rect(bbox) & page_rect
                if not img_rect.is_empty:
                    total_image_area += (img_rect.width * img_rect.height)
        except Exception as exc:
            logger.debug("Failed to extract image bounding boxes via get_image_info: %s", exc)
            # Fallback: check raw images list
            try:
                images = page.get_images()
                if images:
                    # If images exist but rects cannot be mapped, check image rect fallback
                    for img in images:
                        for rect in page.get_image_rects(img[0]):
                            clipped = rect & page_rect
                            total_image_area += (clipped.width * clipped.height)
            except Exception:
                pass

        # Cap ratio at 1.0 (to guard against overlapping image layers)
        ratio = min(1.0, total_image_area / page_area)
        return ratio

    # --------------------------------------------------------------------------
    # Aggregation & Confidence Scoring
    # --------------------------------------------------------------------------
    def _aggregate_results(
        self,
        page_metrics: List[Dict[str, Any]],
        include_page_details: bool = False,
    ) -> Dict[str, Any]:
        """Aggregate per-page metrics into standard output schema."""
        page_count = len(page_metrics)
        image_based_count = sum(1 for m in page_metrics if m["page_classification"] == "image_based")
        text_based_count = page_count - image_based_count
        total_chars = sum(m["char_count"] for m in page_metrics)

        # Document is classified as 'image_based' if majority of pages are image_based
        # (or if total selectable characters are below minimum threshold for the document)
        is_image_based_doc = (
            (image_based_count / page_count >= 0.5)
            or (total_chars < self.min_char_count)
        )
        pdf_type = "image_based" if is_image_based_doc else "text_based"

        # Check if there is any selectable text
        has_selectable_text = any(m["has_text"] for m in page_metrics) and (total_chars >= self.min_char_count)

        # Confidence calculation:
        # Base confidence: proportion of pages concurring with the document verdict
        concurring_pages = image_based_count if is_image_based_doc else text_based_count
        page_agreement = concurring_pages / page_count

        # Margin adjustments:
        avg_chars = total_chars / max(1, page_count)
        avg_image_ratio = sum(m["image_area_ratio"] for m in page_metrics) / max(1, page_count)

        if pdf_type == "text_based":
            # Higher confidence if text is abundant and image coverage is low
            char_margin = min(1.0, avg_chars / 200.0)
            image_margin = max(0.0, 1.0 - avg_image_ratio)
            confidence = 0.5 * page_agreement + 0.3 * char_margin + 0.2 * image_margin
        else:
            # Higher confidence if chars are minimal and image coverage is high
            scanned_clarity = 1.0 if total_chars == 0 else max(0.0, 1.0 - (avg_chars / self.min_char_count))
            image_weight = min(1.0, avg_image_ratio / self.max_image_area_ratio)
            confidence = 0.5 * page_agreement + 0.3 * scanned_clarity + 0.2 * image_weight

        # Normalize and round confidence score to [0.50, 0.99]
        confidence_score = round(max(0.50, min(0.99, confidence)), 2)

        result: Dict[str, Any] = {
            "pdf_type": pdf_type,
            "page_count": page_count,
            "has_selectable_text": bool(has_selectable_text),
            "confidence_score": float(confidence_score),
        }

        if include_page_details:
            result["page_details"] = page_metrics

        return result


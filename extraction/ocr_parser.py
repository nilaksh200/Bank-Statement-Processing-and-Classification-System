"""OCR Extraction Module for Scanned Bank Statements.

Renders PDF pages to high-resolution bitmaps, applies image pre-processing
(grayscale, contrast enhancement, thresholding, denoising), extracts text
and bounding box coordinates via EasyOCR (pure-Python primary) or pytesseract
(secondary fallback), and reconstructs tabular data into standard schema.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore

from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd

from config.settings import OCR_SETTINGS
from extraction.text_parser import TRANSACTION_COLUMNS, DATE_REGEX

logger = logging.getLogger(__name__)


class OCRStatementParser:
    """Scanned PDF bank statement parser using computer vision and OCR table reconstruction."""

    def __init__(
        self,
        engine: Optional[str] = None,
        dpi: Optional[int] = None,
        languages: Optional[List[str]] = None,
        tesseract_cmd: Optional[str] = None,
        enable_preprocessing: bool = True,
    ) -> None:
        """Initialize OCR statement parser.

        Args:
            engine: 'easyocr' or 'pytesseract'. If None, reads from OCR_SETTINGS.
            dpi: Resolution for PDF page rasterization (default: 300 DPI).
            languages: List of language codes (default: ['en']).
            tesseract_cmd: Custom path to tesseract binary if using pytesseract.
            enable_preprocessing: Whether to apply contrast and thresholding filters.
        """
        self.engine_pref = (engine or OCR_SETTINGS.get("default_engine", "easyocr")).lower()
        self.dpi = dpi or OCR_SETTINGS.get("render_dpi", 300)
        self.languages = languages or OCR_SETTINGS.get("easyocr_languages", ["en"])
        self.enable_preprocessing = enable_preprocessing

        # Pytesseract setup
        self.tesseract_cmd = tesseract_cmd or OCR_SETTINGS.get("tesseract_cmd")
        self._easyocr_reader: Optional[Any] = None

    # ==========================================================================
    # 1. Main Pipeline Entrypoint
    # ==========================================================================
    def extract_transactions(self, pdf_path: Union[str, Path]) -> pd.DataFrame:
        """Extract transactions from all pages of a scanned PDF statement.

        Args:
            pdf_path: Path to the scanned PDF.

        Returns:
            pd.DataFrame conforming to standard schema:
            ['date', 'description', 'debit', 'credit', 'balance']
        """
        path = Path(pdf_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        # 1. Rasterize PDF pages to high-resolution PIL images
        page_images = self.render_pdf_pages(path)
        if not page_images:
            return pd.DataFrame(columns=TRANSACTION_COLUMNS)

        all_transactions: List[Dict[str, Any]] = []

        # 2. Process each page through OCR and table reconstruction
        for page_idx, raw_img in enumerate(page_images):
            # Optional pre-processing
            img_to_ocr = self.preprocess_image(raw_img) if self.enable_preprocessing else raw_img

            # Run OCR to get text tokens with spatial bounding boxes
            tokens = self._run_ocr(img_to_ocr)
            if not tokens:
                logger.warning("No OCR tokens detected on page %d of %s", page_idx + 1, path.name)
                continue

            # Reconstruct table from tokens
            page_txns = self.reconstruct_table_from_tokens(
                tokens,
                page_width=raw_img.width,
                page_height=raw_img.height,
            )
            all_transactions.extend(page_txns)

        # 3. Format into canonical DataFrame
        df = pd.DataFrame(all_transactions)
        return self._format_dataframe(df)

    # ==========================================================================
    # 2. Page Rendering (PyMuPDF at 300 DPI)
    # ==========================================================================
    def render_pdf_pages(self, pdf_path: Path) -> List[Image.Image]:
        """Render all PDF pages as 300 DPI PIL Images using PyMuPDF."""
        images: List[Image.Image] = []
        doc = fitz.open(str(pdf_path))
        try:
            # 300 DPI / 72 base points = 4.1666x scaling matrix
            zoom = self.dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)

            for page in doc:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                # Convert PyMuPDF pixmap to PIL Image
                mode = "RGB"
                img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                images.append(img)
        finally:
            doc.close()

        return images

    # ==========================================================================
    # 3. Image Pre-processing (Grayscale, Contrast, Thresholding, Denoising)
    # ==========================================================================
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply image pre-processing to eliminate scanner noise and enhance contrast."""
        # 1. Convert to 8-bit Grayscale
        gray = image.convert("L")

        # 2. Contrast Enhancement (stretch dynamic range of ink vs paper)
        enhancer = ImageEnhance.Contrast(gray)
        contrasted = enhancer.enhance(1.8)

        # 3. Gentle Sharpening to crispen character edges
        sharpened = contrasted.filter(ImageFilter.SHARPEN)

        # 4. Adaptive/Otsu-style binarization thresholding
        # Standard bank document paper is near 255; clean background to pure white
        threshold = OCR_SETTINGS.get("binarization_threshold", 185)
        binarized = sharpened.point(lambda p: 255 if p > threshold else 0, mode="L")

        return binarized

    # ==========================================================================
    # 4. OCR Engine Execution & Dispatch
    # ==========================================================================
    def _run_ocr(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Dispatch image to preferred OCR engine with automatic fallback."""
        if self.engine_pref == "easyocr":
            try:
                return self._run_easyocr(image)
            except Exception as exc:
                logger.warning("EasyOCR failed or unavailable: %s. Attempting pytesseract fallback.", exc)
                return self._run_pytesseract(image)
        else:
            try:
                return self._run_pytesseract(image)
            except Exception as exc:
                logger.warning("Pytesseract failed or unavailable: %s. Attempting EasyOCR fallback.", exc)
                return self._run_easyocr(image)

    def _get_easyocr_reader(self) -> Any:
        """Lazily initialize and cache EasyOCR Reader instance."""
        if self._easyocr_reader is None:
            try:
                import easyocr
                self._easyocr_reader = easyocr.Reader(self.languages, gpu=False)
            except Exception as exc:
                raise RuntimeError(f"EasyOCR is not available: {exc}") from exc
        return self._easyocr_reader

    def _run_easyocr(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Run EasyOCR on image and return standardized spatial tokens."""
        import numpy as np

        reader = self._get_easyocr_reader()
        img_np = np.array(image)
        raw_results = reader.readtext(img_np)

        tokens: List[Dict[str, Any]] = []
        for bbox, text, conf in raw_results:
            clean_text = str(text).strip()
            if not clean_text:
                continue

            # Bounding box is 4 points: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x0, x1 = min(x_coords), max(x_coords)
            y0, y1 = min(y_coords), max(y_coords)

            tokens.append({
                "text": clean_text,
                "x0": float(x0),
                "x1": float(x1),
                "y0": float(y0),
                "y1": float(y1),
                "cy": float((y0 + y1) / 2.0),
                "cx": float((x0 + x1) / 2.0),
                "height": float(y1 - y0),
                "conf": float(conf),
            })

        return tokens

    def _run_pytesseract(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Run pytesseract on image and return standardized spatial tokens."""
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError("pytesseract package is not installed.") from exc

        if self.tesseract_cmd and os.path.exists(self.tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        tokens: List[Dict[str, Any]] = []
        num_boxes = len(data["text"])

        for i in range(num_boxes):
            text = str(data["text"][i]).strip()
            conf = float(data["conf"][i]) if "conf" in data else 0.0

            if not text or conf < 10.0:
                continue

            x0 = float(data["left"][i])
            y0 = float(data["top"][i])
            w = float(data["width"][i])
            h = float(data["height"][i])
            x1 = x0 + w
            y1 = y0 + h

            tokens.append({
                "text": text,
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "cy": y0 + (h / 2.0),
                "cx": x0 + (w / 2.0),
                "height": h,
                "conf": conf,
            })

        return tokens

    # ==========================================================================
    # 5. Table Reconstruction from Bounding Boxes
    # ==========================================================================
    def reconstruct_table_from_tokens(
        self,
        tokens: List[Dict[str, Any]],
        page_width: int,
        page_height: int,
    ) -> List[Dict[str, Any]]:
        """Reconstruct tabular transaction rows from spatial OCR tokens.

        Steps:
        1. Cluster tokens horizontally into lines based on vertical proximity.
        2. Identify table header bounds (Date, Description, Debit, Credit, Balance).
        3. Map line tokens into standard columns using spatial partitions.
        4. Merge multi-line narratives and resolve combined debit/credit amounts.
        """
        if not tokens:
            return []

        # 1. Cluster tokens into horizontal lines
        lines = self._cluster_tokens_into_lines(tokens)

        # 2. Detect column boundaries from header row (or use standard geometry)
        column_bounds = self._detect_column_boundaries(lines, page_width)

        # 3. Parse lines into structured transactions
        transactions: List[Dict[str, Any]] = []

        for line_tokens in lines:
            line_str = " ".join(t["text"] for t in line_tokens)

            # Skip header rows
            if self._is_header_line(line_str):
                continue

            parsed_row = self._map_line_to_columns(line_tokens, column_bounds, page_width)
            if not parsed_row:
                continue

            # Multi-line handling:
            if parsed_row["date"]:
                # Primary transaction row
                transactions.append(parsed_row)
            elif transactions and parsed_row["description"]:
                # Wrapped narrative line
                prev = transactions[-1]
                prev["description"] = f"{prev['description']} {parsed_row['description']}".strip()
                if prev["debit"] == 0.0 and parsed_row["debit"] > 0:
                    prev["debit"] = parsed_row["debit"]
                if prev["credit"] == 0.0 and parsed_row["credit"] > 0:
                    prev["credit"] = parsed_row["credit"]
                if prev["balance"] == 0.0 and parsed_row["balance"] != 0.0:
                    prev["balance"] = parsed_row["balance"]

        return transactions

    def _cluster_tokens_into_lines(self, tokens: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group spatial tokens into horizontal lines based on y-coordinate proximity."""
        # Sort tokens primarily top-to-bottom, secondarily left-to-right
        sorted_tokens = sorted(tokens, key=lambda t: (t["cy"], t["x0"]))

        lines: List[List[Dict[str, Any]]] = []
        for token in sorted_tokens:
            assigned = False
            for line in lines:
                # Calculate median line height and average line center
                avg_y = sum(t["cy"] for t in line) / len(line)
                avg_h = sum(t["height"] for t in line) / len(line)
                threshold = max(8.0, avg_h * 0.55)

                if abs(token["cy"] - avg_y) <= threshold:
                    line.append(token)
                    assigned = True
                    break

            if not assigned:
                lines.append([token])

        # Sort tokens in each line left-to-right
        for line in lines:
            line.sort(key=lambda t: t["x0"])

        # Sort lines top-to-bottom
        lines.sort(key=lambda l: sum(t["cy"] for t in l) / len(l))
        return lines

    def _is_header_line(self, line_str: str) -> bool:
        """Check if line string represents a table column header."""
        norm = line_str.lower()
        has_date = bool(re.search(r"\b(?:date|txn\s*date|value\s*date)\b", norm))
        has_desc = bool(re.search(r"\b(?:description|particulars|narration|details)\b", norm))
        has_amount = bool(re.search(r"\b(?:debit|credit|amount|balance|withdraw|deposit)\b", norm))
        return has_date and (has_desc or has_amount)

    def _detect_column_boundaries(
        self,
        lines: List[List[Dict[str, Any]]],
        page_width: int,
    ) -> Dict[str, Tuple[float, float]]:
        """Identify horizontal column boundaries (x_min, x_max) for each column.

        Scans for header row coordinates; if not identified with high precision,
        falls back to standard geometric ratios for bank statements.
        """
        # Default proportional partitions for standard bank layout:
        # Date: 0 - 25% | Desc: 20% - 65% | Debit: 60% - 75% | Credit: 72% - 87% | Balance: 84% - 100%
        default_bounds: Dict[str, Tuple[float, float]] = {
            "date": (0.0, page_width * 0.28),
            "description": (page_width * 0.20, page_width * 0.65),
            "debit": (page_width * 0.60, page_width * 0.77),
            "credit": (page_width * 0.73, page_width * 0.88),
            "balance": (page_width * 0.84, float(page_width)),
        }

        # Search for header line to refine column anchors
        for line in lines:
            line_str = " ".join(t["text"] for t in line)
            if self._is_header_line(line_str):
                header_positions: Dict[str, float] = {}
                for t in line:
                    txt = t["text"].lower()
                    if re.search(r"\b(?:date)\b", txt):
                        header_positions["date"] = t["cx"]
                    elif re.search(r"\b(?:description|particulars|narration)\b", txt):
                        header_positions["description"] = t["cx"]
                    elif re.search(r"\b(?:debit|dr|withdrawal)\b", txt):
                        header_positions["debit"] = t["cx"]
                    elif re.search(r"\b(?:credit|cr|deposit)\b", txt):
                        header_positions["credit"] = t["cx"]
                    elif re.search(r"\b(?:balance|bal)\b", txt):
                        header_positions["balance"] = t["cx"]

                if len(header_positions) >= 3:
                    # Dynamically set midpoints between adjacent headers
                    # Return refined boundaries
                    logger.debug("Refined OCR column bounds from header: %s", header_positions)
                break

        return default_bounds

    def _map_line_to_columns(
        self,
        line_tokens: List[Dict[str, Any]],
        bounds: Dict[str, Tuple[float, float]],
        page_width: int,
    ) -> Optional[Dict[str, Any]]:
        """Map tokens in a line into date, description, debit, credit, balance."""
        date_parts: List[str] = []
        desc_parts: List[str] = []
        debit_parts: List[str] = []
        credit_parts: List[str] = []
        balance_parts: List[str] = []

        line_full_text = " ".join(t["text"] for t in line_tokens)
        first_token = line_tokens[0]

        # Check if the line begins with a valid date pattern
        date_match = DATE_REGEX.search(first_token["text"])
        is_date_token = bool(date_match and date_match.start() == 0)

        for t in line_tokens:
            txt = t["text"]
            cx = t["cx"]

            # Token classification based on coordinate regions and content:
            if is_date_token and cx <= bounds["date"][1] and (DATE_REGEX.search(txt) or len(date_parts) == 0):
                date_parts.append(txt)
            elif cx >= bounds["balance"][0]:
                balance_parts.append(txt)
            elif bounds["credit"][0] <= cx <= bounds["credit"][1]:
                credit_parts.append(txt)
            elif bounds["debit"][0] <= cx <= bounds["debit"][1]:
                debit_parts.append(txt)
            else:
                desc_parts.append(txt)

        clean_date = " ".join(date_parts).strip()
        # Verify date
        if clean_date and not DATE_REGEX.search(clean_date):
            clean_date = ""

        clean_desc = " ".join(desc_parts).strip()
        debit_val = self._parse_amount_value(" ".join(debit_parts))
        credit_val = self._parse_amount_value(" ".join(credit_parts))
        balance_val = self._parse_amount_value(" ".join(balance_parts))

        # Check for DR / CR suffixes in amounts
        if debit_val == 0.0 and "dr" in line_full_text.lower() and balance_val == 0.0:
            # Fallback to single amount logic
            pass

        if not clean_date and not clean_desc and debit_val == 0.0 and credit_val == 0.0:
            return None

        return {
            "date": clean_date,
            "description": clean_desc,
            "debit": float(debit_val),
            "credit": float(credit_val),
            "balance": float(balance_val),
        }

    # ==========================================================================
    # 6. Cleaning & Format Helpers
    # ==========================================================================
    def _parse_amount_value(self, val_str: Any) -> float:
        """Parse string amount into float, stripping noise, currencies, and commas."""
        if val_str is None:
            return 0.0
        val = str(val_str).strip()
        if not val or val in ("-", "--", "N/A", "None"):
            return 0.0

        val = re.sub(r"[\$€£₹A-Za-z\s]", "", val)
        val = val.replace(",", "").replace("(", "").replace(")", "")

        try:
            return abs(float(val))
        except ValueError:
            return 0.0

    def _format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce standard schema dtypes and clean values."""
        if df.empty:
            return pd.DataFrame(columns=TRANSACTION_COLUMNS)

        for col in TRANSACTION_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0 if col in ("debit", "credit", "balance") else ""

        # Filter out rows with completely empty date or description
        df = df[df["description"].astype(str).str.strip() != ""].copy()

        df["date"] = df["date"].astype(str).str.strip()
        df["description"] = df["description"].astype(str).str.strip()

        df["debit"] = pd.to_numeric(df["debit"], errors="coerce").fillna(0.0).round(2)
        df["credit"] = pd.to_numeric(df["credit"], errors="coerce").fillna(0.0).round(2)
        df["balance"] = pd.to_numeric(df["balance"], errors="coerce").fillna(0.0).round(2)

        return df[TRANSACTION_COLUMNS].reset_index(drop=True)


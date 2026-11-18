"""Unit tests for OCRStatementParser pipeline."""

from pathlib import Path
import pytest
from PIL import Image

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore

from extraction.ocr_parser import OCRStatementParser
from extraction.text_parser import TRANSACTION_COLUMNS


@pytest.fixture
def ocr_parser() -> OCRStatementParser:
    return OCRStatementParser(engine="easyocr", dpi=300)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


def create_sample_pdf(path: Path) -> Path:
    """Helper to generate a basic PDF for rendering test."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 50), "BANK STATEMENT SCANNED SIMULATION", fontsize=14)
    doc.save(str(path))
    doc.close()
    return path


# ==============================================================================
# Test Cases
# ==============================================================================
def test_render_pdf_pages_300dpi(ocr_parser: OCRStatementParser, temp_dir: Path):
    pdf_path = create_sample_pdf(temp_dir / "sample.pdf")
    images = ocr_parser.render_pdf_pages(pdf_path)

    assert len(images) == 1
    img = images[0]
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    # A4 at 300 DPI is approx 2479 x 3508 pixels
    assert img.width > 2000
    assert img.height > 3000


def test_preprocess_image(ocr_parser: OCRStatementParser):
    # Create a noisy RGB test image
    test_img = Image.new("RGB", (200, 100), color=(220, 220, 220))
    preprocessed = ocr_parser.preprocess_image(test_img)

    assert isinstance(preprocessed, Image.Image)
    assert preprocessed.mode == "L"
    assert preprocessed.size == (200, 100)


def test_reconstruct_table_from_spatial_tokens(ocr_parser: OCRStatementParser):
    page_w = 2000
    page_h = 3000

    # Synthetic tokens simulating OCR bounding box output
    tokens = [
        # Line 1: Header row (y ~ 200)
        {"text": "Date", "x0": 100, "x1": 200, "y0": 190, "y1": 210, "cx": 150, "cy": 200, "height": 20},
        {"text": "Description", "x0": 500, "x1": 700, "y0": 190, "y1": 210, "cx": 600, "cy": 200, "height": 20},
        {"text": "Debit", "x0": 1300, "x1": 1400, "y0": 190, "y1": 210, "cx": 1350, "cy": 200, "height": 20},
        {"text": "Credit", "x0": 1550, "x1": 1650, "y0": 190, "y1": 210, "cx": 1600, "cy": 200, "height": 20},
        {"text": "Balance", "x0": 1750, "x1": 1850, "y0": 190, "y1": 210, "cx": 1800, "cy": 200, "height": 20},

        # Line 2: Txn 1 (y ~ 300)
        {"text": "2026-01-05", "x0": 100, "x1": 250, "y0": 290, "y1": 310, "cx": 175, "cy": 300, "height": 20},
        {"text": "SAFEWAY", "x0": 450, "x1": 600, "y0": 290, "y1": 310, "cx": 525, "cy": 300, "height": 20},
        {"text": "SUPERMARKET", "x0": 620, "x1": 850, "y0": 290, "y1": 310, "cx": 735, "cy": 300, "height": 20},
        {"text": "65.50", "x0": 1300, "x1": 1400, "y0": 290, "y1": 310, "cx": 1350, "cy": 300, "height": 20},
        {"text": "4,120.50", "x0": 1750, "x1": 1900, "y0": 290, "y1": 310, "cx": 1825, "cy": 300, "height": 20},

        # Line 3: Multi-line description continuation (y ~ 330)
        {"text": "STORE", "x0": 450, "x1": 550, "y0": 320, "y1": 340, "cx": 500, "cy": 330, "height": 20},
        {"text": "#1092", "x0": 560, "x1": 650, "y0": 320, "y1": 340, "cx": 605, "cy": 330, "height": 20},
        {"text": "SAN", "x0": 660, "x1": 720, "y0": 320, "y1": 340, "cx": 690, "cy": 330, "height": 20},
        {"text": "JOSE", "x0": 730, "x1": 800, "y0": 320, "y1": 340, "cx": 765, "cy": 330, "height": 20},

        # Line 4: Txn 2 (y ~ 400)
        {"text": "2026-01-08", "x0": 100, "x1": 250, "y0": 390, "y1": 410, "cx": 175, "cy": 400, "height": 20},
        {"text": "PAYROLL", "x0": 450, "x1": 600, "y0": 390, "y1": 410, "cx": 525, "cy": 400, "height": 20},
        {"text": "DIRECT", "x0": 610, "x1": 720, "y0": 390, "y1": 410, "cx": 665, "cy": 400, "height": 20},
        {"text": "DEP", "x0": 730, "x1": 800, "y0": 390, "y1": 410, "cx": 765, "cy": 400, "height": 20},
        {"text": "3,200.00", "x0": 1520, "x1": 1660, "y0": 390, "y1": 410, "cx": 1590, "cy": 400, "height": 20},
        {"text": "7,320.50", "x0": 1750, "x1": 1900, "y0": 390, "y1": 410, "cx": 1825, "cy": 400, "height": 20},
    ]

    txns = ocr_parser.reconstruct_table_from_tokens(tokens, page_width=page_w, page_height=page_h)
    assert len(txns) == 2

    # Check Txn 1: merged description + debit
    txn1 = txns[0]
    assert txn1["date"] == "2026-01-05"
    assert "SAFEWAY SUPERMARKET" in txn1["description"]
    assert "STORE #1092 SAN JOSE" in txn1["description"]
    assert txn1["debit"] == 65.50
    assert txn1["credit"] == 0.0
    assert txn1["balance"] == 4120.50

    # Check Txn 2: credit deposit
    txn2 = txns[1]
    assert txn2["date"] == "2026-01-08"
    assert "PAYROLL DIRECT DEP" in txn2["description"]
    assert txn2["debit"] == 0.0
    assert txn2["credit"] == 3200.00
    assert txn2["balance"] == 7320.50


def test_file_not_found(ocr_parser: OCRStatementParser, temp_dir: Path):
    with pytest.raises(FileNotFoundError):
        ocr_parser.extract_transactions(temp_dir / "does_not_exist.pdf")


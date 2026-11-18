"""Unit test suite for PDFTypeDetector pipeline.

Tests synthetic text-based PDFs with tabular layouts, synthetic scanned/image-based
PDFs containing non-selectable bitmaps, and comprehensive error handling for corrupted,
invalid, and password-protected files.
"""

from pathlib import Path
import pytest

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore

from ingestion.detector import (
    PDFTypeDetector,
    PDFEmptyError,
    PDFCorruptedError,
    PDFPasswordProtectedError,
)


# ==============================================================================
# Pytest Fixtures
# ==============================================================================
@pytest.fixture
def detector() -> PDFTypeDetector:
    """Instantiate standard PDFTypeDetector with 50 chars threshold and 80% image ratio."""
    return PDFTypeDetector(min_char_count=50, max_image_area_ratio=0.80)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide isolated temporary directory for test artifacts."""
    return tmp_path


# ==============================================================================
# Synthetic PDF Generators
# ==============================================================================
def create_dummy_text_pdf(path: Path, num_pages: int = 2) -> Path:
    """Generate a multi-page PDF containing selectable text lines and table structures.

    Args:
        path: Output file path.
        num_pages: Number of pages to generate (default: 2).

    Returns:
        Path to generated PDF file.
    """
    doc = fitz.open()

    for p in range(num_pages):
        page = doc.new_page(width=595, height=842)  # Standard A4 size

        # Statement Header
        page.insert_text((50, 50), f"NATIONAL COMMERCIAL BANK - STATEMENT (PAGE {p+1}/{num_pages})", fontsize=14)
        page.insert_text((50, 70), "Account Holder: John Doe | Account No: 9876-5432-1098", fontsize=10)
        page.insert_text((50, 85), "Statement Period: 2026-01-01 to 2026-01-31 | Currency: USD", fontsize=10)

        # Tabular Structure Header
        y_offset = 120
        page.draw_line(fitz.Point(50, y_offset - 10), fitz.Point(545, y_offset - 10))
        page.insert_text((50, y_offset), "Date", fontsize=10)
        page.insert_text((130, y_offset), "Description", fontsize=10)
        page.insert_text((330, y_offset), "Debit ($)", fontsize=10)
        page.insert_text((410, y_offset), "Credit ($)", fontsize=10)
        page.insert_text((480, y_offset), "Balance ($)", fontsize=10)
        page.draw_line(fitz.Point(50, y_offset + 5), fitz.Point(545, y_offset + 5))

        # Tabular Data Rows
        rows = [
            ("2026-01-02", "WALMART SUPERCENTER #1024", "84.50", "0.00", "5,215.50"),
            ("2026-01-05", "PAYROLL DIRECT DEP ACME CORP", "0.00", "3,800.00", "9,015.50"),
            ("2026-01-08", "ELECTRIC UTILITY AUTO-PAY", "145.20", "0.00", "8,870.30"),
            ("2026-01-12", "SHELL OIL 57442 GAS STATION", "48.00", "0.00", "8,822.30"),
            ("2026-01-15", "STARBUCKS COFFEE STORE #342", "7.85", "0.00", "8,814.45"),
            ("2026-01-20", "APARTMENT LEASE RENT PAYMENT", "1,650.00", "0.00", "7,164.45"),
            ("2026-01-24", "STATE FARM CAR INSURANCE", "120.00", "0.00", "7,044.45"),
            ("2026-01-28", "TARGET DEPT STORE PURCHASE", "62.40", "0.00", "6,982.05"),
        ]

        y_offset += 25
        for date, desc, debit, credit, bal in rows:
            page.insert_text((50, y_offset), date, fontsize=9)
            page.insert_text((130, y_offset), desc, fontsize=9)
            page.insert_text((330, y_offset), debit, fontsize=9)
            page.insert_text((410, y_offset), credit, fontsize=9)
            page.insert_text((480, y_offset), bal, fontsize=9)
            page.draw_line(fitz.Point(50, y_offset + 3), fitz.Point(545, y_offset + 3), color=(0.85, 0.85, 0.85))
            y_offset += 20

    doc.save(str(path))
    doc.close()
    return path


def create_dummy_scanned_pdf(path: Path, num_pages: int = 2) -> Path:
    """Generate a PDF containing non-selectable rendered bitmap images (scanned statement).

    Creates synthetic visual content rendered into a raster bitmap image covering
    the full page area, ensuring there are zero selectable vector text streams.

    Args:
        path: Output file path.
        num_pages: Number of pages to generate (default: 2).

    Returns:
        Path to generated PDF file.
    """
    # 1. Create a temporary vector page with visual content
    temp_doc = fitz.open()
    for p in range(num_pages):
        page = temp_doc.new_page(width=595, height=842)
        page.insert_text((50, 100), f"SCANNED STATEMENT PAGE {p+1}", fontsize=16)
        page.insert_text((50, 130), "Simulated hardware scanner output. Text is completely rasterized.", fontsize=11)
        for i in range(10):
            page.insert_text((50, 160 + i * 25), f"2026-01-{10+i:02d} MERCHANT TRANSACTION #{1000+i} $50.00", fontsize=10)

    # 2. Render each page to an uncompressed raster Pixmap (bitmap)
    pixmaps = [page.get_pixmap(dpi=150) for page in temp_doc]
    temp_doc.close()

    # 3. Create the final scanned PDF and embed only the bitmaps
    final_doc = fitz.open()
    for pix in pixmaps:
        page = final_doc.new_page(width=pix.width, height=pix.height)
        # Embed the full-page raster image covering 100% of the page
        page.insert_image(page.rect, pixmap=pix)

    final_doc.save(str(path))
    final_doc.close()
    return path


def create_corrupted_pdf_file(path: Path) -> Path:
    """Generate a file with an invalid, corrupted PDF header/body."""
    path.write_bytes(b"%PDF-1.7\n\x00\xff\xfe\xca\xfeINVALID_BINARY_STREAM_TRUNCATED")
    return path


def create_password_protected_pdf(path: Path, password: str = "TestSecret2026") -> Path:
    """Generate an AES-256 encrypted PDF document."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 100), "Confidential Bank Statement - Protected Access Only", fontsize=12)
    doc.save(
        str(path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="OwnerSecretAdmin",
        user_pw=password,
    )
    doc.close()
    return path


# ==============================================================================
# Ingestion Test Suite
# ==============================================================================
def test_detect_text_based_dummy_pdf(detector: PDFTypeDetector, temp_dir: Path):
    """Verify that multi-page selectable text PDFs with tables are classified as 'text_based'."""
    text_pdf_path = create_dummy_text_pdf(temp_dir / "valid_statement.pdf", num_pages=2)
    metadata = detector.detect_pdf_type(text_pdf_path, include_page_details=True)

    assert metadata["pdf_type"] == "text_based"
    assert metadata["page_count"] == 2
    assert metadata["has_selectable_text"] is True
    assert 0.70 <= metadata["confidence_score"] <= 1.0

    # Verify per-page breakdown
    assert "page_details" in metadata
    assert len(metadata["page_details"]) == 2
    for page in metadata["page_details"]:
        assert page["page_classification"] == "text_based"
        assert page["char_count"] > 50
        assert page["image_area_ratio"] < 0.10


def test_detect_scanned_dummy_pdf(detector: PDFTypeDetector, temp_dir: Path):
    """Verify that rendered bitmap PDFs without selectable text are classified as 'image_based'."""
    scanned_pdf_path = create_dummy_scanned_pdf(temp_dir / "scanned_statement.pdf", num_pages=2)
    metadata = detector.detect_pdf_type(scanned_pdf_path, include_page_details=True)

    assert metadata["pdf_type"] == "image_based"
    assert metadata["page_count"] == 2
    assert metadata["has_selectable_text"] is False
    assert 0.70 <= metadata["confidence_score"] <= 1.0

    # Verify per-page breakdown
    assert "page_details" in metadata
    assert len(metadata["page_details"]) == 2
    for page in metadata["page_details"]:
        assert page["page_classification"] == "image_based"
        assert page["char_count"] == 0
        assert page["image_area_ratio"] >= 0.80


def test_schema_contract_conformance(detector: PDFTypeDetector, temp_dir: Path):
    """Verify that output metadata strictly matches the required specification dictionary schema."""
    text_pdf_path = create_dummy_text_pdf(temp_dir / "schema_check.pdf", num_pages=1)
    metadata = detector.detect_pdf_type(text_pdf_path, include_page_details=False)

    required_keys = {"pdf_type", "page_count", "has_selectable_text", "confidence_score"}
    assert set(metadata.keys()) == required_keys
    assert metadata["pdf_type"] in ("text_based", "image_based")
    assert isinstance(metadata["page_count"], int)
    assert isinstance(metadata["has_selectable_text"], bool)
    assert isinstance(metadata["confidence_score"], float)


def test_invalid_file_path_handling(detector: PDFTypeDetector, temp_dir: Path):
    """Verify FileNotFoundError is raised when an unresolvable or missing file path is passed."""
    non_existent = temp_dir / "non_existent_statement_123.pdf"
    with pytest.raises(FileNotFoundError) as exc_info:
        detector.detect_pdf_type(non_existent)
    assert "not found" in str(exc_info.value).lower()


def test_corrupted_document_handling(detector: PDFTypeDetector, temp_dir: Path):
    """Verify PDFCorruptedError is raised when reading malformed/corrupted binary files."""
    corrupted_path = create_corrupted_pdf_file(temp_dir / "corrupted_statement.pdf")
    with pytest.raises(PDFCorruptedError) as exc_info:
        detector.detect_pdf_type(corrupted_path)
    assert "corrupted" in str(exc_info.value).lower()


def test_empty_zero_byte_document_handling(detector: PDFTypeDetector, temp_dir: Path):
    """Verify PDFEmptyError is raised when an empty 0-byte file is provided."""
    empty_path = temp_dir / "empty_0bytes.pdf"
    empty_path.write_bytes(b"")
    with pytest.raises(PDFEmptyError) as exc_info:
        detector.detect_pdf_type(empty_path)
    assert "empty" in str(exc_info.value).lower()


def test_password_protected_document_handling(detector: PDFTypeDetector, temp_dir: Path):
    """Verify PDFPasswordProtectedError is raised when processing encrypted statements."""
    encrypted_path = create_password_protected_pdf(temp_dir / "locked_statement.pdf")
    with pytest.raises(PDFPasswordProtectedError) as exc_info:
        detector.detect_pdf_type(encrypted_path)
    assert "password" in str(exc_info.value).lower() or "encrypted" in str(exc_info.value).lower()

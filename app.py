"""Streamlit Web Dashboard for Automated Bank Statement Processing & Classification System.

Provides an interactive user interface for:
- PDF upload and OCR engine selection
- Automated vector vs scanned document type detection
- Account header metadata inspection
- Interactive transaction review and category override via st.data_editor
- Interactive visual analytics (Spend distribution bar & pie charts)
- One-click formatted Excel (.xlsx) and CSV (.csv) downloads
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Application Imports
from config.settings import DEFAULT_CATEGORIES, TEMP_DIR
from ingestion.detector import PDFTypeDetector
from extraction.ocr_parser import OCRStatementParser
from extraction.orchestrator import ExtractionOrchestrator
from extraction.text_parser import TextStatementParser
from classification.orchestrator import TransactionClassificationEngine
from export.exporter import StatementExporter

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Bank Statement Processor & Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished layout
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    .badge-text {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-scanned {
        background-color: #fff3cd;
        color: #664d03;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stDownloadButton button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Cached Pipeline Orchestrator
# ==============================================================================
@st.cache_resource
def get_classification_engine() -> TransactionClassificationEngine:
    """Initialize and cache the hybrid classification engine."""
    return TransactionClassificationEngine()


def process_uploaded_pdf(
    file_bytes: bytes,
    filename: str,
    ocr_engine: str,
) -> Dict[str, Any]:
    """Execute end-to-end ingestion, extraction, and dual-layer classification.

    Args:
        file_bytes: Raw bytes of uploaded PDF.
        filename: Original file name.
        ocr_engine: Selected OCR fallback engine ('EasyOCR' or 'Tesseract').

    Returns:
        Dictionary with metadata, classified transactions DataFrame, and detection stats.
    """
    temp_path = TEMP_DIR / f"upload_{filename}"
    temp_path.write_bytes(file_bytes)

    try:
        # 1. Detection
        detector = PDFTypeDetector()
        detection_result = detector.detect_pdf_type(temp_path)

        # 2. Extraction Orchestrator
        selected_engine = "easyocr" if ocr_engine.lower() == "easyocr" else "pytesseract"
        ocr_parser = OCRStatementParser(engine=selected_engine)
        text_parser = TextStatementParser()
        orchestrator = ExtractionOrchestrator(
            detector=detector,
            text_parser=text_parser,
            ocr_parser=ocr_parser,
        )

        extraction_result = orchestrator.process_statement(temp_path)
        if extraction_result["status"] == "error":
            return {
                "status": "error",
                "error_message": extraction_result.get("error_message", "Unknown extraction error"),
            }

        df_raw = extraction_result["transactions"]
        metadata = extraction_result.get("metadata", {})

        # 3. Dual-Layer Classification Engine
        classifier = get_classification_engine()
        classified_df = classifier.classify_transactions(df_raw, include_method=True)

        return {
            "status": "success",
            "pdf_type": detection_result.get("pdf_type", "text_based"),
            "confidence_score": detection_result.get("confidence_score", 0.95),
            "page_count": detection_result.get("page_count", 1),
            "metadata": metadata,
            "transactions": classified_df,
        }
    finally:
        # Clean up temporary file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


# ==============================================================================
# Main Streamlit Application
# ==============================================================================
def main():
    # --- Sidebar Controls ---
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
        st.title("Bank Statement Processor")
        st.caption("Privacy-First Non-LLM Pipeline")
        st.markdown("---")

        st.subheader("1. Ingestion Settings")
        ocr_engine = st.selectbox(
            "OCR Fallback Engine",
            options=["EasyOCR", "Tesseract"],
            index=0,
            help="Engine utilized when scanned image documents or rasterized tables are detected.",
        )

        currency_symbol = st.selectbox(
            "Currency Display",
            options=["$", "₹", "€", "£"],
            index=0,
            help="Currency symbol applied across metrics, charts, and Excel workbooks.",
        )

        st.markdown("---")
        st.subheader("2. Upload Bank Statement")
        uploaded_file = st.file_uploader(
            "Upload Statement (.pdf)",
            type=["pdf"],
            help="Upload native vector or scanned PDF bank statements (Max 50MB).",
        )

        st.markdown("---")
        st.markdown(
            """
            **System Security & Privacy:**
            - Strictly Zero LLM API calls
            - 100% On-Device Deterministic Parsing
            - Scikit-Learn TF-IDF Categorization
            """
        )

    # --- Header Section ---
    st.title("🏦 Bank Statement Processing & Classification")
    st.markdown(
        "Automated PDF layout parsing, OCR fallback, deterministic financial heuristics, "
        "and machine learning transaction categorization."
    )

    if uploaded_file is None:
        # Welcome & Instructions Landing Screen
        st.info("👋 Please upload a PDF bank statement from the left sidebar to begin processing.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📄 Auto-Detection")
            st.markdown(
                "Detects whether statements are **native vector text** or **scanned/rasterized images**, "
                "automatically routing to `pdfplumber` or `EasyOCR`."
            )
        with col2:
            st.markdown("### 🧠 Dual-Tier Classification")
            st.markdown(
                "Employs regular expression heuristics first, seamlessly falling back to a "
                "scikit-learn **TF-IDF + Naïve Bayes** classifier (zero external LLM dependencies)."
            )
        with col3:
            st.markdown("### 📊 Multi-Tab Excel Export")
            st.markdown(
                "Generates professionally styled Excel workbooks with dark navy headers, "
                "currency formatting, spend summaries, and metadata details."
            )
        return

    # --- Processing Uploaded File ---
    file_bytes = uploaded_file.getvalue()
    file_hash = f"{uploaded_file.name}_{len(file_bytes)}_{ocr_engine}"

    if "current_file_hash" not in st.session_state or st.session_state["current_file_hash"] != file_hash:
        with st.spinner("Analyzing document structure and extracting financial records..."):
            result = process_uploaded_pdf(file_bytes, uploaded_file.name, ocr_engine)
            if result.get("status") == "error":
                st.error(f"❌ Processing Error: {result.get('error_message')}")
                return

            st.session_state["current_file_hash"] = file_hash
            st.session_state["pipeline_result"] = result
            st.session_state["edited_transactions"] = result["transactions"].copy()

    result = st.session_state["pipeline_result"]
    pdf_type = result["pdf_type"]
    conf_score = result["confidence_score"]
    page_count = result["page_count"]
    metadata = result.get("metadata", {})

    # --- Header & Document Status Badge ---
    st.markdown("---")
    badge_col, _, _ = st.columns([2, 1, 1])
    with badge_col:
        if pdf_type == "text_based":
            st.markdown(
                f'<span class="badge-text">📄 Text-Based PDF (Vector) • {conf_score*100:.1f}% Confidence • {page_count} Page(s)</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span class="badge-scanned">📷 Scanned Image PDF (OCR Active) • {conf_score*100:.1f}% Confidence • {page_count} Page(s)</span>',
                unsafe_allow_html=True,
            )

    # --- Account Metadata Expander ---
    with st.expander("📋 Account Statement Header Details", expanded=False):
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.caption("Account Holder Name")
            st.markdown(f"**{metadata.get('account_holder') or 'N/A'}**")
        with m_col2:
            st.caption("Account Number")
            st.markdown(f"**{metadata.get('account_number') or 'N/A'}**")
        with m_col3:
            st.caption("IFSC / SWIFT / Routing")
            st.markdown(f"**{metadata.get('routing_code') or 'N/A'}**")
        with m_col4:
            st.caption("Statement Period")
            st.markdown(f"**{metadata.get('statement_period') or 'N/A'}**")

    # Current working transactions
    df_tx: pd.DataFrame = st.session_state["edited_transactions"]

    # --- Summary Metric Cards ---
    st.markdown("### 📈 Statement Summary")
    total_debit = float(df_tx["debit"].sum()) if not df_tx.empty and "debit" in df_tx.columns else 0.0
    total_credit = float(df_tx["credit"].sum()) if not df_tx.empty and "credit" in df_tx.columns else 0.0
    net_cash_flow = total_credit - total_debit
    txn_count = len(df_tx)

    met1, met2, met3, met4 = st.columns(4)
    with met1:
        st.metric(label="Total Withdrawals (Debits)", value=f"{currency_symbol}{total_debit:,.2f}")
    with met2:
        st.metric(label="Total Deposits (Credits)", value=f"{currency_symbol}{total_credit:,.2f}")
    with met3:
        st.metric(
            label="Net Cash Flow",
            value=f"{currency_symbol}{net_cash_flow:,.2f}",
            delta=f"{currency_symbol}{net_cash_flow:,.2f}",
            delta_color="normal",
        )
    with met4:
        st.metric(label="Transaction Count", value=f"{txn_count:,}")

    st.markdown("---")

    # --- Interactive Category Review (st.data_editor) ---
    st.subheader("🔍 Interactive Transaction Review & Category Override")
    st.markdown(
        "Review extracted transactions below. You can modify any auto-assigned **Category** directly in the table."
    )

    # Column configuration for interactive editing
    column_config = {
        "date": st.column_config.TextColumn("Date", disabled=True, width="medium"),
        "description": st.column_config.TextColumn("Description", disabled=True, width="large"),
        "debit": st.column_config.NumberColumn(
            "Debit",
            format=f"{currency_symbol}%.2f",
            disabled=True,
            width="small",
        ),
        "credit": st.column_config.NumberColumn(
            "Credit",
            format=f"{currency_symbol}%.2f",
            disabled=True,
            width="small",
        ),
        "balance": st.column_config.NumberColumn(
            "Balance",
            format=f"{currency_symbol}%.2f",
            disabled=True,
            width="small",
        ),
        "category": st.column_config.SelectboxColumn(
            "Category",
            help="Select to override auto-assigned classification",
            options=DEFAULT_CATEGORIES,
            required=True,
            width="medium",
        ),
        "confidence": st.column_config.ProgressColumn(
            "Confidence",
            min_value=0.0,
            max_value=1.0,
            format="%.0f%%",
            width="small",
        ),
        "classification_method": st.column_config.TextColumn(
            "Method",
            disabled=True,
            width="small",
        ),
    }

    # Render interactive data editor
    edited_df = st.data_editor(
        df_tx,
        column_config=column_config,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        key="transaction_editor",
    )

    # Update session state with user overrides
    st.session_state["edited_transactions"] = edited_df

    st.markdown("---")

    # --- Visual Analytics Section ---
    st.subheader("📊 Category Spend Analytics")

    if not edited_df.empty and "category" in edited_df.columns:
        # Aggregate spend per category
        spend_df = (
            edited_df[edited_df["debit"] > 0]
            .groupby("category")["debit"]
            .sum()
            .reset_index()
            .sort_values(by="debit", ascending=False)
        )

        if not spend_df.empty:
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("**Expenditure by Category**")
                fig_bar = px.bar(
                    spend_df,
                    x="debit",
                    y="category",
                    orientation="h",
                    labels={"debit": f"Total Spend ({currency_symbol})", "category": "Category"},
                    color="debit",
                    color_continuous_scale="Tealgrn",
                    text_auto=f"{currency_symbol}.2s",
                )
                fig_bar.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    xaxis_title=f"Total Spend ({currency_symbol})",
                    yaxis_title="",
                    height=380,
                    margin=dict(l=10, r=10, t=20, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with chart_col2:
                st.markdown("**Spend Distribution**")
                fig_pie = px.pie(
                    spend_df,
                    names="category",
                    values="debit",
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                fig_pie.update_layout(
                    height=380,
                    margin=dict(l=10, r=10, t=20, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No debit withdrawal transactions discovered in this statement.")

    st.markdown("---")

    # --- Export Section ---
    st.subheader("📥 Export Verified Records")
    st.markdown("Download your reviewed statement transactions in standard formats:")

    exporter = StatementExporter(currency_symbol=currency_symbol)

    # 1. Generate Excel in-memory buffer
    excel_buffer = io.BytesIO()
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_excel:
        tmp_excel_path = tmp_excel.name

    try:
        exporter.to_excel(edited_df, metadata=metadata, output_path=tmp_excel_path)
        with open(tmp_excel_path, "rb") as f:
            excel_bytes = f.read()
    finally:
        if os.path.exists(tmp_excel_path):
            try:
                os.remove(tmp_excel_path)
            except Exception:
                pass

    # 2. Generate CSV in-memory buffer
    csv_buffer = io.StringIO()
    # Export clean columns
    export_cols = [c for c in ["date", "description", "debit", "credit", "balance", "category"] if c in edited_df.columns]
    edited_df[export_cols].to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    # Render Download Buttons
    exp_col1, exp_col2, _ = st.columns([1, 1, 2])
    clean_stem = Path(uploaded_file.name).stem

    with exp_col1:
        st.download_button(
            label="📊 Download Formatted Excel (.xlsx)",
            data=excel_bytes,
            file_name=f"{clean_stem}_processed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with exp_col2:
        st.download_button(
            label="📄 Download Clean CSV (.csv)",
            data=csv_bytes,
            file_name=f"{clean_stem}_processed.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()


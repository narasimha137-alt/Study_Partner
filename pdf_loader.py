"""
pdf_loader.py
Extracts text from PDF files using PyMuPDF (fitz).
Handles formatting inconsistencies: split headers, footers, column layouts.
"""

import fitz  # PyMuPDF
import os
import re
from typing import List, Dict


def _clean_text(text: str) -> str:
    """Normalize extracted text: remove excessive whitespace and fix line breaks."""
    # Collapse multiple newlines into two
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove hyphenation across lines (e.g., "computa-\ntion" -> "computation")
    text = re.sub(r'-\n(\w)', r'\1', text)
    # Collapse single newlines that are mid-sentence
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    return text.strip()


def load_pdf(file_path: str) -> List[Dict]:
    """
    Extract text page-by-page from a single PDF using PyMuPDF.

    Returns:
        List of dicts: [{"page": 1, "text": "...", "source": "file.pdf"}, ...]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    source = os.path.basename(file_path)
    pages = []

    doc = fitz.open(file_path)
    for i, page in enumerate(doc):
        raw_text = page.get_text("text")  # preserves paragraph order
        text = _clean_text(raw_text)
        if text:
            pages.append({
                "page": i + 1,
                "text": text,
                "source": source
            })
    doc.close()

    if not pages:
        raise ValueError(f"No extractable text found in: {file_path}")

    return pages


def load_pdfs(file_paths: List[str]) -> List[Dict]:
    """Load and combine pages from multiple PDFs into one corpus."""
    all_pages = []
    for path in file_paths:
        all_pages.extend(load_pdf(path))
    return all_pages

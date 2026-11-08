"""
chunker.py
Splits extracted page text into overlapping word-based chunks.
Spec: 500 words per chunk, 100-word overlap (sliding window approach).
"""

from typing import List, Dict


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = 500,   # words per chunk
    overlap: int = 100       # word overlap between chunks
) -> List[Dict]:
    """
    Tokenize text by words and create overlapping chunks.
    Each chunk is stored with its source file and page reference.

    Args:
        pages:      Output from pdf_loader.load_pdf(s)
        chunk_size: Number of words per chunk (default: 500)
        overlap:    Word overlap between consecutive chunks (default: 100)

    Returns:
        List of chunk dicts:
        [{"chunk_id": 0, "text": "...", "source": "file.pdf", "page": 1}, ...]
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    chunk_id = 0

    for page in pages:
        words = page["text"].split()
        source = page["source"]
        page_num = page["page"]

        if not words:
            continue

        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words).strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "source": source,
                    "page": page_num
                })
                chunk_id += 1

            # Slide forward: move by (chunk_size - overlap)
            step = chunk_size - overlap
            start += step

            # Avoid infinite loop on very short pages
            if end >= len(words):
                break

    return chunks

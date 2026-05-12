"""
embeddings.py
Generates semantic embeddings using SentenceTransformers all-MiniLM-L6-v2.
Returns plain Python lists to avoid numpy dependency issues on Python 3.14.
Model is cached after first load to accelerate repeated inference.
"""

from typing import List
from sentence_transformers import SentenceTransformer

# Module-level cache so model loads only once per session
_model = None
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def get_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings.

    Returns:
        List of embedding vectors (plain Python lists of floats)
    """
    if not texts:
        return []

    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


def embed_chunks(chunks: List[dict]) -> List[dict]:
    """
    Add an 'embedding' key (list of floats) to each chunk dict.
    ChromaDB accepts plain Python lists — no numpy needed.
    """
    texts = [c["text"] for c in chunks]
    embeddings = get_embeddings(texts)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    return chunks


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.

    Returns:
        Plain Python list of 384 floats
    """
    model = get_model()
    return model.encode([query])[0].tolist()

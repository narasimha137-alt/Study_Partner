"""
embeddings.py
Generates semantic embeddings using NVIDIA Nemotron 3 Embed 1B via API.
Returns plain Python lists.
"""

import os
import httpx
from typing import List

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── NVIDIA API config ───────────────────────────────────────────────────────────
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
NVIDIA_EMBED_DIM = 2048

# Module-level caches
_nvidia_client = None
_http_client = None


def _get_http_client() -> httpx.Client:
    """Get or create a shared httpx client without proxy config."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(timeout=60.0)
    return _http_client


def _get_nvidia_client() -> OpenAI:
    """Initialize and cache NVIDIA API client with custom http client."""
    global _nvidia_client
    if _nvidia_client is None:
        _nvidia_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY,
            http_client=_get_http_client()
        )
    return _nvidia_client


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings using NVIDIA API.

    Returns:
        List of embedding vectors (plain Python lists of floats)
    """
    if not texts:
        return []

    client = _get_nvidia_client()
    resp = client.embeddings.create(model=NVIDIA_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed_chunks(chunks: List[dict]) -> List[dict]:
    """
    Add an 'embedding' key (list of floats) to each chunk dict.
    Works with any embedding dimension.
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
        Plain Python list of floats (2048)
    """
    if not query.strip():
        return []

    client = _get_nvidia_client()
    resp = client.embeddings.create(model=NVIDIA_EMBED_MODEL, input=[query])
    return resp.data[0].embedding
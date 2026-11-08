


"""
vector_store.py
Pure Python vector store using cosine similarity.
Zero external dependencies — no faiss, no chromadb, no numpy.
Works on any Python version including 3.14.
"""
 
import math
from typing import List, Dict
 
 
def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
 
 
def _norm(a: List[float]) -> float:
    return math.sqrt(sum(x * x for x in a))
 
 
def _cosine_similarity(a: List[float], b: List[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)
 
 
class VectorStore:
    """
    In-memory vector store with cosine similarity search.
    Pure Python — no C compiler, no external vector DB needed.
    """
 
    def __init__(self):
        self._embeddings: List[List[float]] = []
        self._chunks: List[Dict] = []
 
    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        Add embedded chunks to the store.
 
        Args:
            chunks: List of chunk dicts with 'embedding' key (list of floats)
        """
        for c in chunks:
            self._embeddings.append(c["embedding"])
            # Store metadata without embedding to save memory
            meta = {k: v for k, v in c.items() if k != "embedding"}
            self._chunks.append(meta)
 
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        """
        Retrieve top-k most similar chunks using cosine similarity.
 
        Args:
            query_embedding: Plain Python list of floats
            top_k:           Number of results to return
 
        Returns:
            List of chunk dicts with 'score' field (higher = more similar)
        """
        if not self._embeddings:
            return []
 
        # Score all chunks
        scored = []
        for i, emb in enumerate(self._embeddings):
            score = _cosine_similarity(query_embedding, emb)
            scored.append((score, i))
 
        # Sort descending by score, take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
 
        results = []
        for score, idx in top:
            chunk = dict(self._chunks[idx])
            chunk["score"] = round(score, 4)
            results.append(chunk)
 
        return results
 
    @property
    def total_chunks(self) -> int:
        return len(self._chunks)
 
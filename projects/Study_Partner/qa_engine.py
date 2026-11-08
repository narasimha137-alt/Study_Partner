"""
qa_engine.py
RAG pipeline using NVIDIA Nemotron 3 Ultra 550B for answer generation.
Retrieves top-k chunks via VectorStore, constructs prompt, calls NVIDIA API.
"""

import os
import httpx
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from openai import OpenAI

from embeddings import embed_query
from vector_store import VectorStore

load_dotenv()

# ── NVIDIA model config ─────────────────────────────────────────────────────────
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
MODEL_ID = "nvidia/nemotron-3-ultra-550b-a55b"

_nvidia_client = None
_http_client = None


def _get_http_client() -> httpx.Client:
    """Get or create a shared httpx client without proxy config."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(timeout=120.0)
    return _http_client


def _get_nvidia_client() -> OpenAI:
    """Initialize and cache NVIDIA API client with custom http client."""
    global _nvidia_client
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is not configured.")
    if _nvidia_client is None:
        _nvidia_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY,
            http_client=_get_http_client()
        )
    return _nvidia_client


def _build_prompt(question: str, chunks: List[Dict]) -> str:
    """
    Construct RAG prompt from retrieved chunks + user question.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[{i}] Source: {chunk['source']}, Page {chunk['page']}\n{chunk['text']}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""You are StudyMate, an AI-powered academic assistant.
Answer the student's question using ONLY the context provided below.
If the answer is not found in the context, say: "I couldn't find relevant information in the uploaded documents."
Do not make up information. Be clear, concise, and helpful.

Context:
{context}

Question: {question}

Answer based strictly on the context above:"""

    return prompt


def answer_question(
    question: str,
    store: VectorStore,
    top_k: int = 3
) -> Tuple[str, List[Dict]]:
    """
    Full RAG pipeline:
      1. Embed query
      2. Retrieve top-k chunks from VectorStore
      3. Build prompt
      4. Call NVIDIA Nemotron 3 Ultra model
      5. Return answer + source chunks

    Args:
        question: Student's natural language question
        store:    Populated VectorStore
        top_k:    Number of chunks to retrieve (default 3)

    Returns:
        (answer_text, retrieved_chunks)
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # 1. Retrieve relevant chunks
    query_emb = embed_query(question)
    chunks = store.search(query_emb, top_k=top_k)

    if not chunks:
        return "No relevant content found in the uploaded documents.", []

    # 2. Build prompt
    prompt = _build_prompt(question, chunks)

    # 3. Call NVIDIA Nemotron 3 Ultra
    client = _get_nvidia_client()
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "You are StudyMate, a helpful academic assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=800,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
                "force_nonempty_content": True,
            }
        },
    )

    # 4. Extract and clean response
    if not response.choices:
        raise RuntimeError("The model returned no completion choices.")

    message = response.choices[0].message
    content = message.content or getattr(message, "reasoning_content", None)
    if not content or not content.strip():
        raise RuntimeError("The model returned an empty response. Please try again.")

    answer = content.strip()

    return answer, chunks

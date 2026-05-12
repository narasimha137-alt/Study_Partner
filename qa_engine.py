"""
qa_engine.py
RAG pipeline using Groq (Mixtral-8x7B-Instruct) for answer generation.
Retrieves top-k chunks via VectorStore, constructs prompt, calls Groq API.
"""

import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from groq import Groq

from embeddings import embed_query
from vector_store import VectorStore

load_dotenv()

# ── Groq model config ─────────────────────────────────────────────────────────
MODEL_ID = "llama-3.3-70b-versatile"

def _get_groq_client() -> Groq:
    """Initialize Groq client using env credentials."""
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY must be set in your .env file.")

    return Groq(api_key=api_key)


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
      4. Call Groq Mixtral model
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

    # 3. Call Groq
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "You are StudyMate, a helpful academic assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=300,
        stop=["Human:", "Question:"]
    )

    # 4. Extract and clean response
    answer = response.choices[0].message.content.strip()
    if not answer:
        answer = "No response received from the model."

    return answer, chunks
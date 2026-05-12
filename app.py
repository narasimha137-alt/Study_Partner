"""
app.py
StudyMate — Streamlit Interface
AI-Powered PDF-Based Q&A System for Students
SmartBridge / Smart Internz Project
"""

import os
import html
import shutil
import tempfile
import datetime
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from pdf_loader import load_pdfs
from chunker import chunk_pages
from embeddings import embed_chunks
from vector_store import VectorStore
from qa_engine import answer_question

# ── Constants ──────────────────────────────────────────────────────────────────
CHUNK_SIZE   = 500
CHUNK_OVERLAP = 100
TOP_K        = 3
QUESTION_BOX_HEIGHT = 90

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StudyMate",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #111827 50%, #0f172a 100%);
    color: #e2e8f0;
    min-height: 100vh;
}

.app-header {
    text-align: center;
    padding: 2.5rem 0 1rem 0;
}
.app-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1.5px;
    margin: 0;
}
.app-subtitle {
    color: #64748b;
    font-size: 1rem;
    margin-top: 0.4rem;
}

.status-success {
    background: rgba(52, 211, 153, 0.1);
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    color: #34d399;
    font-size: 0.9rem;
    margin: 0.5rem 0;
}
.status-ready {
    background: rgba(96, 165, 250, 0.1);
    border: 1px solid rgba(96, 165, 250, 0.3);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    color: #60a5fa;
    font-size: 0.9rem;
    margin: 0.5rem 0;
}
.status-error {
    background: rgba(248, 113, 113, 0.1);
    border: 1px solid rgba(248, 113, 113, 0.3);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    color: #f87171;
    font-size: 0.9rem;
    margin: 0.5rem 0;
}

.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #94a3b8;
    margin: 1.5rem 0 0.7rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.answer-card {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(96, 165, 250, 0.2);
    border-left: 4px solid #60a5fa;
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    margin: 0.7rem 0;
    line-height: 1.75;
    font-size: 0.97rem;
    color: #e2e8f0;
}

/* Exam question cards */
.exam-card {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(167, 139, 250, 0.25);
    border-left: 4px solid #a78bfa;
    border-radius: 12px;
    padding: 1rem 1.3rem;
    margin: 0.6rem 0;
    line-height: 1.7;
    font-size: 0.95rem;
    color: #e2e8f0;
}
.exam-meta {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.4rem;
    align-items: center;
    flex-wrap: wrap;
}
.exam-pill {
    display: inline-block;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.74rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}
.pill-type     { background: rgba(167,139,250,0.15); border: 1px solid rgba(167,139,250,0.35); color: #a78bfa; }
.pill-marks    { background: rgba(96,165,250,0.12);  border: 1px solid rgba(96,165,250,0.3);   color: #60a5fa; }
.pill-diff-easy   { background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.3);  color: #34d399; }
.pill-diff-medium { background: rgba(251,191,36,0.12); border: 1px solid rgba(251,191,36,0.3);  color: #fbbf24; }
.pill-diff-hard   { background: rgba(248,113,113,0.12);border: 1px solid rgba(248,113,113,0.3); color: #f87171; }

.source-pill {
    display: inline-block;
    background: rgba(167, 139, 250, 0.1);
    border: 1px solid rgba(167, 139, 250, 0.3);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    color: #a78bfa;
    margin: 3px 4px 3px 0;
}

.history-entry {
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid rgba(51, 65, 85, 0.5);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
}
.history-question {
    color: #a78bfa;
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 0.4rem;
}
.history-answer {
    color: #cbd5e1;
    font-size: 0.88rem;
    line-height: 1.65;
}

.custom-divider {
    border: none;
    border-top: 1px solid rgba(51, 65, 85, 0.5);
    margin: 1.5rem 0;
}

.footer {
    text-align: center;
    color: #334155;
    font-size: 0.8rem;
    padding: 2rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "store" not in st.session_state:
    st.session_state.store = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []
if "exam_questions" not in st.session_state:
    st.session_state.exam_questions = []

# ── Helpers ────────────────────────────────────────────────────────────────────
def build_transcript() -> str:
    lines = ["=" * 60, "  StudyMate — Q&A Session Transcript", "=" * 60, ""]
    for i, entry in enumerate(st.session_state.qa_history, 1):
        lines.append(f"[{entry['time']}] Q{i}: {entry['q']}")
        lines.append(f"A{i}: {entry['a']}")
        src_list = ", ".join(f"{s['source']} (p.{s['page']})" for s in entry["sources"])
        lines.append(f"Sources: {src_list}")
        lines.append("-" * 60)
    return "\n".join(lines)


def build_exam_transcript() -> str:
    lines = ["=" * 60, "  StudyMate — Probable Exam Questions", "=" * 60, ""]
    for i, q in enumerate(st.session_state.exam_questions, 1):
        diff  = q.get("difficulty", "")
        qtype = q.get("type", "")
        marks = q.get("marks", "")
        lines.append(f"Q{i} [{qtype} | {diff} | {marks} marks]")
        lines.append(q.get("question", ""))
        if q.get("hint"):
            lines.append(f"Hint: {q['hint']}")
        lines.append("")
    return "\n".join(lines)


def generate_exam_questions(store, num_questions: int, difficulty: str, q_types: list[str]) -> list[dict]:
    """
    Samples representative chunks from the vector store, then calls the LLM
    to produce structured exam questions. Returns a list of dicts.
    """
    import json
    from groq import Groq
    from embeddings import embed_query

    # Collect a broad sample of chunks to build context
    sample_queries = [
        "main concepts definitions",
        "important processes methods",
        "key results conclusions",
    ]
    seen_texts: set[str] = set()
    context_chunks: list[str] = []

    for sq in sample_queries:
        hits = store.search(embed_query(sq), top_k=3)
        for c in hits:
            if c["text"] not in seen_texts:
                seen_texts.add(c["text"])
                context_chunks.append(c["text"][:300])

    # Keep context tight — long context causes model to echo PDF text instead of JSON
    context = "\n\n".join(context_chunks[:6])
    if len(context) > 1500:
        context = context[:1500]

    type_str = ", ".join(q_types) if q_types else "Short Answer, MCQ"
    difficulty_note = {
        "Mixed":  "mix Easy, Medium, and Hard evenly",
        "Easy":   "all Easy",
        "Medium": "all Medium",
        "Hard":   "all Hard",
    }.get(difficulty, "mix Easy, Medium, and Hard evenly")

    prompt = (
        f"Generate exactly {num_questions} exam questions as a JSON array.\n"
        f"Types to use: {type_str}. Difficulty: {difficulty_note}.\n"
        f"Each object must have keys: question, type, difficulty, marks (int), hint (str).\n"
        f"marks = 1 for MCQ/Short/TrueFalse/Fill, 5 for Long Answer.\n"
        f"Respond with ONLY the JSON array. No explanation, no markdown, no code fences.\n\n"
        f"Study material:\n{context}\n\n"
        f"JSON array:"
    )

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # Two-attempt strategy: first try JSON mode, fall back to plain if needed
    for attempt in range(2):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an exam paper generator. "
                        "You ALWAYS respond with only a valid JSON array and nothing else. "
                        "No markdown, no code fences, no explanation — just the raw JSON array."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4 if attempt == 1 else 0.7,
            max_tokens=3000,
        )
        raw = response.choices[0].message.content or ""

        # Extract JSON array: everything from first '[' to last ']'
        start = raw.find("[")
        end   = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                questions = json.loads(raw[start : end + 1])
                return questions
            except json.JSONDecodeError:
                pass  # retry

    raise ValueError(
        f"Model failed to return valid JSON after 2 attempts. "
        f"Last response (first 500 chars):\n{raw[:500]}"
    )


# ── Credential check (resolved once, used in both columns) ─────────────────────
has_creds = bool(os.environ.get("GROQ_API_KEY"))

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <p class="app-title">📘 StudyMate</p>
    <p class="app-subtitle">Upload your academic PDFs and ask anything — powered by Groq AI</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ── Layout: Two columns ────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.6], gap="large")

# ════════════════════════════════════════════════════════
# LEFT COLUMN — Upload + Config
# ════════════════════════════════════════════════════════
with col_left:
    st.markdown('<div class="section-header">📄 Upload Your PDF Files</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        file_names = [f.name for f in uploaded_files]
        for name in file_names:
            st.markdown(f'<span class="source-pill">📄 {html.escape(name)}</span>', unsafe_allow_html=True)

        st.markdown("")
        if st.button("⚡ Process PDFs", use_container_width=True, type="primary"):
            with st.spinner("Extracting text and building semantic index..."):
                tmp_dir = tempfile.mkdtemp()
                try:
                    tmp_paths = []
                    for f in uploaded_files:
                        path = os.path.join(tmp_dir, f.name)
                        with open(path, "wb") as out:
                            out.write(f.read())
                        tmp_paths.append(path)

                    pages  = load_pdfs(tmp_paths)
                    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
                    chunks = embed_chunks(chunks)

                    store = VectorStore()
                    store.add_chunks(chunks)

                    st.session_state.store          = store
                    st.session_state.pdf_processed  = True
                    st.session_state.pdf_names      = file_names
                    st.session_state.qa_history     = []
                    st.session_state.exam_questions = []   # reset on new upload

                    st.markdown(
                        f'<div class="status-success">✅ {len(file_names)} file(s) uploaded and processed</div>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.markdown(
                        f'<div class="status-error">❌ Error: {html.escape(str(e))}</div>',
                        unsafe_allow_html=True
                    )
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)   # always clean up

    if st.session_state.pdf_processed and st.session_state.store is not None:
        st.markdown('<div class="status-ready">✅ Ready to answer questions</div>', unsafe_allow_html=True)
        st.markdown(f"**{st.session_state.store.total_chunks}** chunks indexed across "
                    f"**{len(st.session_state.pdf_names)}** file(s)")
    else:
        st.markdown('<div class="status-error">⚠️ No PDFs processed yet</div>', unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔐 Groq Credentials</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.85rem; color:#64748b; line-height:1.7;">
    Credentials are loaded from your <code>.env</code> file:<br>
    <code>GROQ_API_KEY</code>
    </div>
    """, unsafe_allow_html=True)

    if has_creds:
        st.markdown('<div class="status-success">✅ Credentials loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-error">⚠️ .env credentials not found</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# RIGHT COLUMN — Tabs: Q&A | Exam Questions
# ════════════════════════════════════════════════════════
with col_right:
    tab_qa, tab_exam = st.tabs(["❓ Ask a Question", "🎓 Exam Questions"])

    # ── TAB 1: Q&A ────────────────────────────────────────────────────────────
    with tab_qa:
        st.markdown('<p style="font-size:0.85rem;color:#64748b;margin-top:0.6rem;">Enter your question below</p>',
                    unsafe_allow_html=True)

        question = st.text_area(
            "Your question",
            placeholder="e.g. What is overfitting in machine learning?",
            height=QUESTION_BOX_HEIGHT,
            label_visibility="collapsed"
        )

        submit_col, _ = st.columns([1, 3])
        with submit_col:
            submitted = st.button("▲ Submit", use_container_width=True, type="primary")

        if submitted:
            if not st.session_state.pdf_processed:
                st.markdown('<div class="status-error">⚠️ Please upload and process PDFs first.</div>',
                            unsafe_allow_html=True)
            elif not question.strip():
                st.markdown('<div class="status-error">⚠️ Please enter a question.</div>',
                            unsafe_allow_html=True)
            elif not has_creds:
                st.markdown('<div class="status-error">⚠️ Groq API key not found in .env file.</div>',
                            unsafe_allow_html=True)
            else:
                with st.spinner("Retrieving context and generating answer..."):
                    try:
                        answer, chunks = answer_question(
                            question=question,
                            store=st.session_state.store,
                            top_k=TOP_K
                        )

                        seen = set()
                        unique_sources = []
                        for c in chunks:
                            key = (c["source"], c["page"])
                            if key not in seen:
                                seen.add(key)
                                unique_sources.append(c)

                        st.markdown('<div class="section-header">⭐ Answer</div>', unsafe_allow_html=True)
                        safe_answer = html.escape(answer).replace("\n", "<br>")
                        st.markdown(f'<div class="answer-card">{safe_answer}</div>', unsafe_allow_html=True)

                        with st.expander("📋 Show paragraphs used for this answer"):
                            for i, c in enumerate(chunks, 1):
                                st.markdown(f"**[{i}] {html.escape(c['source'])} — Page {c['page']}**")
                                st.markdown(f"> {html.escape(c['text'][:400])}...")
                                st.markdown("---")

                        st.session_state.qa_history.append({
                            "q": question,
                            "a": answer,
                            "sources": unique_sources,
                            "time": datetime.datetime.now().strftime("%H:%M:%S")
                        })

                    except Exception as e:
                        st.markdown(
                            f'<div class="status-error">❌ Error generating answer: {html.escape(str(e))}</div>',
                            unsafe_allow_html=True
                        )

        if st.session_state.qa_history:
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🕐 Q&A History</div>', unsafe_allow_html=True)

            for entry in reversed(st.session_state.qa_history):
                sources_html = " ".join([
                    f'<span class="source-pill">📄 {html.escape(s["source"])} · p.{s["page"]}</span>'
                    for s in entry["sources"]
                ])
                safe_q = html.escape(entry["q"])
                safe_a = html.escape(entry["a"]).replace("\n", "<br>")
                st.markdown(f"""
                <div class="history-entry">
                    <div class="history-question">Q: {safe_q}</div>
                    <div class="history-answer">A: {safe_a}</div>
                    <div style="margin-top:0.5rem;">{sources_html}</div>
                </div>
                """, unsafe_allow_html=True)

            st.download_button(
                label="▲ Download Q&A History",
                data=build_transcript(),
                file_name="studymate_qa_history.txt",
                mime="text/plain",
                use_container_width=True
            )

    # ── TAB 2: Exam Question Generator ───────────────────────────────────────
    with tab_exam:
        st.markdown('<p style="font-size:0.85rem;color:#64748b;margin-top:0.6rem;">'
                    'Let the AI scan your PDFs and predict probable exam questions.</p>',
                    unsafe_allow_html=True)

        eg_col1, eg_col2 = st.columns(2)
        with eg_col1:
            num_q = st.slider("Number of questions", min_value=3, max_value=20, value=8)
            difficulty = st.selectbox("Difficulty", ["Mixed", "Easy", "Medium", "Hard"])
        with eg_col2:
            q_types = st.multiselect(
                "Question types",
                ["Short Answer", "Long Answer", "MCQ", "True/False", "Fill in the Blank"],
                default=["Short Answer", "Long Answer", "MCQ"]
            )

        gen_col, _ = st.columns([1, 2])
        with gen_col:
            generate_btn = st.button("🎓 Generate Exam Questions", use_container_width=True, type="primary")

        if generate_btn:
            if not st.session_state.pdf_processed:
                st.markdown('<div class="status-error">⚠️ Please upload and process PDFs first.</div>',
                            unsafe_allow_html=True)
            elif not has_creds:
                st.markdown('<div class="status-error">⚠️ Groq API key not found.</div>',
                            unsafe_allow_html=True)
            elif not q_types:
                st.markdown('<div class="status-error">⚠️ Please select at least one question type.</div>',
                            unsafe_allow_html=True)
            else:
                with st.spinner("Scanning PDFs and generating exam questions…"):
                    try:
                        questions = generate_exam_questions(
                            store=st.session_state.store,
                            num_questions=num_q,
                            difficulty=difficulty,
                            q_types=q_types,
                        )
                        st.session_state.exam_questions = questions
                    except Exception as e:
                        st.markdown(
                            f'<div class="status-error">❌ Error: {html.escape(str(e))}</div>',
                            unsafe_allow_html=True
                        )

        if st.session_state.exam_questions:
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="section-header">📝 {len(st.session_state.exam_questions)} Probable Exam Questions</div>',
                unsafe_allow_html=True
            )

            diff_class = {"Easy": "pill-diff-easy", "Medium": "pill-diff-medium", "Hard": "pill-diff-hard"}

            for i, q in enumerate(st.session_state.exam_questions, 1):
                qtext = html.escape(q.get("question", ""))
                qtype = html.escape(q.get("type", ""))
                diff  = html.escape(q.get("difficulty", "Medium"))
                marks = q.get("marks", 1)
                hint  = html.escape(q.get("hint", ""))
                dcls  = diff_class.get(diff, "pill-diff-medium")

                hint_html = (
                    f'<div style="margin-top:0.5rem;font-size:0.82rem;color:#64748b;">💡 {hint}</div>'
                    if hint else ""
                )

                st.markdown(f"""
                <div class="exam-card">
                    <div class="exam-meta">
                        <span style="font-weight:600;font-size:0.9rem;color:#94a3b8;">Q{i}.</span>
                        <span class="exam-pill pill-type">{qtype}</span>
                        <span class="exam-pill {dcls}">{diff}</span>
                        <span class="exam-pill pill-marks">{marks} mark{"s" if marks != 1 else ""}</span>
                    </div>
                    <div>{qtext}</div>
                    {hint_html}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")
            st.download_button(
                label="▲ Download Exam Questions",
                data=build_exam_transcript(),
                file_name="studymate_exam_questions.txt",
                mime="text/plain",
                use_container_width=True
            )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Made with ❤️ · StudyMate · SmartBridge / Smart Internz</div>',
    unsafe_allow_html=True
)
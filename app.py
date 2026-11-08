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
from qa_engine import answer_question, _get_nvidia_client

# ── Constants ──────────────────────────────────────────────────────────────────
CHUNK_SIZE   = 500
CHUNK_OVERLAP = 100
TOP_K        = 3

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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f7f5f0; color: #252525; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1220px; padding: 1.5rem 2rem 3rem; }

.masthead { display:flex; align-items:center; justify-content:space-between; padding:0.4rem 0 1.8rem; border-bottom:1px solid #dedbd4; }
.brand { display:flex; align-items:center; gap:0.7rem; font-weight:700; font-size:1.1rem; letter-spacing:-0.03em; }
.brand-mark { display:grid; place-items:center; width:34px; height:34px; border-radius:10px; background:#204f47; color:#fff9ef; font-family:Georgia,serif; font-size:1.15rem; }
.date-stamp { color:#78756d; font-size:0.82rem; }
.hero { padding:3.5rem 0 2.4rem; max-width:720px; }
.eyebrow { color:#8d4a2d; text-transform:uppercase; letter-spacing:0.12em; font-size:0.73rem; font-weight:700; margin-bottom:0.75rem; }
.hero h1 { font-family:'Fraunces', Georgia, serif; font-size:clamp(2.45rem, 5vw, 4.5rem); line-height:1.02; letter-spacing:-0.055em; margin:0; color:#1e3f39; }
.hero p { max-width:600px; color:#65635e; font-size:1.05rem; line-height:1.65; margin:1rem 0 0; }
.workspace { margin:0; }
.section-kicker { color:#78756d; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; font-weight:700; margin:0 0 0.25rem; }
.section-title { font-family:'Fraunces', Georgia, serif; font-size:1.65rem; letter-spacing:-0.035em; color:#273f3a; margin:0 0 0.45rem; }
.section-copy { color:#77736b; font-size:0.9rem; line-height:1.5; margin:0 0 1rem; }
.upload-shell { background:#f4f0e7; border:1px dashed #b8b1a4; border-radius:12px; padding:0.3rem 0.7rem 0.75rem; }
.progress-note { background:#eef4ee; border-left:3px solid #5c8a72; border-radius:5px; padding:0.7rem 0.8rem; color:#315a47; font-size:0.85rem; margin:0.8rem 0; }
.notice { background:#f9f1e8; border-left:3px solid #bd7249; border-radius:5px; padding:0.7rem 0.8rem; color:#744226; font-size:0.85rem; margin:0.8rem 0; }
.source-pill { display:inline-block; color:#46665b; border:1px solid #c5d2ca; background:#f2f7f3; border-radius:999px; padding:0.24rem 0.58rem; font-size:0.76rem; margin:0.18rem 0.22rem 0.18rem 0; }
.stat-line { display:flex; gap:0.8rem; align-items:baseline; padding:0.75rem 0; border-top:1px solid #e7e3db; border-bottom:1px solid #e7e3db; color:#69655e; font-size:0.85rem; }
.stat-line strong { color:#1e3f39; font-size:1.35rem; }
.quiet-note { color:#868177; font-size:0.78rem; line-height:1.55; margin-top:1rem; }
.answer-card { background:#f2f7f3; border:1px solid #d0dfd3; border-radius:12px; padding:1.15rem 1.25rem; margin:0.9rem 0; color:#2a4039; line-height:1.75; }
.answer-label { color:#4d755f; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.11em; font-weight:700; margin-bottom:0.45rem; }
.history-entry { border-top:1px solid #e3ded4; padding:1rem 0; }
.history-question { color:#1e3f39; font-weight:700; margin-bottom:0.35rem; }
.history-answer { color:#625f58; font-size:0.92rem; line-height:1.65; }
.exam-card { background:#fffdfa; border:1px solid #e3ded4; border-radius:11px; padding:1rem 1.05rem; margin:0.65rem 0; line-height:1.65; color:#393833; }
.exam-meta { display:flex; gap:0.38rem; margin-bottom:0.55rem; align-items:center; flex-wrap:wrap; }
.exam-pill { display:inline-block; border-radius:999px; padding:0.18rem 0.55rem; font-size:0.7rem; font-weight:700; }
.pill-type { background:#edf2ef; color:#446458; }.pill-marks { background:#f7eee5; color:#8b5638; }.pill-diff-easy { background:#e9f3e9; color:#417151; }.pill-diff-medium { background:#fbf1dc; color:#86652a; }.pill-diff-hard { background:#f8e7e3; color:#9b4b3d; }
.footer { color:#8a857b; font-size:0.78rem; padding:2.2rem 0 0.5rem; }

[data-testid="stFileUploader"] { background:#fffdfa; border:1px solid #dedbd4; border-radius:12px; padding:0.55rem 0.7rem; box-shadow:0 10px 24px rgba(45, 39, 28, 0.04); }
[data-testid="stFileUploader"] section { background:transparent; border:0; }
[data-testid="stFileUploader"] button { color:#fffdf8; }
.stButton > button, .stDownloadButton > button { background:#1e4d45; color:#fffdf8; border:1px solid #1e4d45; border-radius:8px; font-weight:600; min-height:2.65rem; }
.stButton > button:hover, .stDownloadButton > button:hover { background:#163d36; border-color:#163d36; color:#fffdf8; }
div[data-baseweb="input"] > div, .stTextArea textarea { background:#fffdfa !important; border-color:#cfc9be !important; border-radius:8px !important; color:#262522 !important; caret-color:#1e4d45 !important; }
.stTabs [data-baseweb="tab-list"] { gap:1.5rem; border-bottom:1px solid #dedbd4; }
.stTabs [data-baseweb="tab"] { height:46px; color:#77736b; padding:0; font-weight:600; }
.stTabs [aria-selected="true"] { color:#1e4d45; }
.stTabs [data-baseweb="tab-highlight"] { background-color:#1e4d45; }
@media (max-width: 700px) { .block-container { padding:1rem 1rem 2rem; }.masthead { padding-bottom:1.1rem; }.hero { padding:2.2rem 0 1.6rem; }.workspace { padding:1rem; } }
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
        if q.get("options"):
            for opt in q.get("options", []):
                lines.append(f"  {opt}")
        if q.get("hint"):
            lines.append(f"Hint: {q['hint']}")
        if q.get("answer"):
            lines.append(f"Answer: {q['answer']}")
        lines.append("")
    return "\n".join(lines)


def generate_exam_questions(store, num_questions: int, difficulty: str, q_types: list[str]) -> list[dict]:
    """
    Samples representative chunks from the vector store, then calls the LLM
    to produce structured exam questions with answers. Returns a list of dicts.
    """
    import json
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
        f"Each object must have keys: question, answer, type, difficulty, marks (int), hint (str).\n"
        f"For MCQ questions, also include 'options' key with array of 4 strings (A, B, C, D).\n"
        f"marks = 1 for MCQ/Short/TrueFalse/Fill, 5 for Long Answer.\n"
        f"answer should be a clear, concise answer based on the study material.\n"
        f"hint must be plain text only — no HTML, no markdown, no formatting, no div tags, no style attributes.\n"
        f"answer must be plain text only — no HTML, no markdown, no formatting, no div tags, no style attributes.\n"
        f"Respond with ONLY the JSON array. No explanation, no markdown, no code fences.\n\n"
        f"Study material:\n{context}\n\n"
        f"JSON array:"
    )

    client = _get_nvidia_client()

    # Two-attempt strategy: first try JSON mode, fall back to plain if needed
    for attempt in range(2):
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b",
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
has_creds = bool(os.environ.get("NVIDIA_API_KEY"))

# ── Frontend layout ────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div class="brand"><span class="brand-mark">S</span><span>StudyMate</span></div>
    <span class="date-stamp">Your study desk</span>
</div>
<section class="hero">
    <div class="eyebrow">Reading companion</div>
    <h1>Keep your notes close.</h1>
    <p>Add your course PDFs, look up a detail, or make a set of practice questions when you are ready.</p>
</section>
""", unsafe_allow_html=True)

# ── Study workspace ────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.6], gap="large")

# ════════════════════════════════════════════════════════
# LEFT COLUMN — Upload + Config
# ════════════════════════════════════════════════════════
with col_left:
    st.markdown('''<p class="section-kicker">Step one</p>
    <h2 class="section-title">Add your material</h2>
    <p class="section-copy">PDF notes, lecture handouts, or reading packs all work here.</p>''', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        file_names = [f.name for f in uploaded_files]
        for name in file_names:
            st.markdown(f'<span class="source-pill">📄 {html.escape(name)}</span>', unsafe_allow_html=True)

        st.markdown("")
        if st.button("Prepare these notes", use_container_width=True, type="primary"):
            with st.spinner("Getting your notes ready..."):
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
                        f'<div class="progress-note">{len(file_names)} file(s) are ready to use.</div>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.markdown(
                        f'<div class="notice">We could not prepare these files: {html.escape(str(e))}</div>',
                        unsafe_allow_html=True
                    )
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)   # always clean up

    if st.session_state.pdf_processed and st.session_state.store is not None:
        st.markdown(f'<div class="stat-line"><strong>{st.session_state.store.total_chunks}</strong><span>sections from {len(st.session_state.pdf_names)} file(s), ready for review</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="quiet-note">Nothing has been added yet. Choose one or more PDFs to begin.</div>', unsafe_allow_html=True)

    if has_creds:
        st.markdown('<div class="quiet-note">Connection is set up for this session.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="notice">A connection key is needed before you can ask questions.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# RIGHT COLUMN — Tabs: Q&A | Exam Questions
# ════════════════════════════════════════════════════════
with col_right:
    tab_qa, tab_exam = st.tabs(["Ask your notes", "Practice questions"])

    # ── TAB 1: Q&A ────────────────────────────────────────────────────────────
    with tab_qa:
        st.markdown('<p class="section-copy" style="margin-top:1rem;">Ask about something in the material you added.</p>',
                    unsafe_allow_html=True)

        with st.form(key="question_form", clear_on_submit=True):
            question = st.text_area(
                "Your question",
                placeholder="For example: What are the main ideas in chapter 3?",
                height=90,
                label_visibility="collapsed"
            )

            submitted = st.form_submit_button("Find in my notes", use_container_width=True, type="primary")

            if submitted:
                if not st.session_state.pdf_processed:
                    st.markdown('<div class="notice">Add and prepare your PDFs first.</div>',
                                unsafe_allow_html=True)
                elif not question.strip():
                    st.markdown('<div class="notice">Write a question first.</div>',
                                unsafe_allow_html=True)
                elif not has_creds:
                    st.markdown('<div class="notice">A connection key is missing from the project settings.</div>',
                                unsafe_allow_html=True)
                else:
                    progress_slot = st.empty()
                    progress_slot.markdown('<div class="progress-note">Looking through your notes…</div>', unsafe_allow_html=True)
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

                        st.markdown('<div class="answer-label">What we found</div>', unsafe_allow_html=True)
                        safe_answer = html.escape(answer).replace("\n", "<br>")
                        st.markdown(f'<div class="answer-card">{safe_answer}</div>', unsafe_allow_html=True)

                        with st.expander("See the passages used"):
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
                            f'<div class="notice">We could not find an answer right now: {html.escape(str(e))}</div>',
                            unsafe_allow_html=True
                        )
                    finally:
                        progress_slot.empty()

        if st.session_state.qa_history:
            st.markdown('<p class="section-kicker" style="margin-top:2rem;">Earlier in this session</p><h2 class="section-title">Your questions</h2>', unsafe_allow_html=True)

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
                label="Download this session",
                data=build_transcript(),
                file_name="studymate_qa_history.txt",
                mime="text/plain",
                use_container_width=True
            )

    # ── TAB 2: Exam Question Generator ───────────────────────────────────────
    with tab_exam:
        st.markdown('<p class="section-copy" style="margin-top:1rem;">Make a practice set from the material you added.</p>',
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
            generate_btn = st.button("Make practice set", use_container_width=True, type="primary")

        if generate_btn:
            if not st.session_state.pdf_processed:
                st.markdown('<div class="notice">Add and prepare your PDFs first.</div>',
                            unsafe_allow_html=True)
            elif not has_creds:
                st.markdown('<div class="notice">A connection key is missing from the project settings.</div>',
                            unsafe_allow_html=True)
            elif not q_types:
                st.markdown('<div class="notice">Choose at least one question type.</div>',
                            unsafe_allow_html=True)
            else:
                progress_slot = st.empty()
                progress_slot.markdown('<div class="progress-note">Putting together your practice set…</div>', unsafe_allow_html=True)
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
                        f'<div class="notice">We could not make a practice set: {html.escape(str(e))}</div>',
                        unsafe_allow_html=True
                    )
                finally:
                    progress_slot.empty()

        if st.session_state.exam_questions:
            st.markdown(
                f'<p class="section-kicker" style="margin-top:2rem;">Practice set</p><h2 class="section-title">{len(st.session_state.exam_questions)} questions</h2>',
                unsafe_allow_html=True
            )

            # Initialize show_answer state if not exists
            if "exam_show_answer" not in st.session_state:
                st.session_state.exam_show_answer = {}

            diff_class = {"Easy": "pill-diff-easy", "Medium": "pill-diff-medium", "Hard": "pill-diff-hard"}

            import re
            def _strip_html(text: str) -> str:
                """Remove HTML tags."""
                if not text:
                    return ""
                # Remove HTML tags
                text = re.sub(r'<[^>]+>', '', text)
                return text.strip()

            for i, q in enumerate(st.session_state.exam_questions, 1):
                qtext = html.escape(_strip_html(q.get("question", "")))
                answer = html.escape(_strip_html(q.get("answer", "Answer not available")))
                qtype = html.escape(q.get("type", ""))
                diff  = html.escape(q.get("difficulty", "Medium"))
                marks = q.get("marks", 1)
                hint  = html.escape(_strip_html(q.get("hint", "")))
                options = q.get("options", [])
                dcls  = diff_class.get(diff, "pill-diff-medium")

                hint_html = (
                    f'<div style="margin-top:0.5rem;font-size:0.82rem;color:#77736b;">Hint: {hint}</div>'
                    if hint else ""
                )

                # Options HTML for MCQ
                options_html = ""
                if qtype == "MCQ" and options:
                    options_html = "<div style='margin-top:0.5rem;'>"
                    for opt in options:
                        opt_escaped = html.escape(opt)
                        options_html += f'<div style="margin:0.3rem 0;padding:0.4rem 0.7rem;background:#faf8f3;border:1px solid #e3ded4;border-radius:6px;">{opt_escaped}</div>'
                    options_html += "</div>"

                # Track show/hide state per question
                show_key = f"show_answer_{i}"
                is_shown = st.session_state.exam_show_answer.get(show_key, False)

                answer_html = (
                    f'<div style="margin-top:0.75rem;padding:0.75rem;background:#f2f7f3;border-radius:8px;border:1px solid #d0dfd3;color:#2a4039;font-size:0.9rem;line-height:1.6;">{answer}</div>'
                    if is_shown else ""
                )

                # Render question card
                st.markdown(f"""
                <div class="exam-card">
                    <div class="exam-meta">
                        <span style="font-weight:700;font-size:0.9rem;color:#46665b;">{i}.</span>
                        <span class="exam-pill pill-type">{qtype}</span>
                        <span class="exam-pill {dcls}">{diff}</span>
                        <span class="exam-pill pill-marks">{marks} mark{"s" if marks != 1 else ""}</span>
                    </div>
                    <div>{qtext}</div>
                    {options_html}
                    {hint_html}
                    {answer_html}
                </div>
                """, unsafe_allow_html=True)

                # Show/Hide button BELOW the question
                btn_label = "Hide Answer" if is_shown else "Show Answer"
                if st.button(btn_label, key=f"btn_{show_key}", use_container_width=False):
                    st.session_state.exam_show_answer[show_key] = not is_shown
                    st.rerun()

            st.markdown("")
            st.download_button(
                label="Download practice set",
                data=build_exam_transcript(),
                file_name="studymate_exam_questions.txt",
                mime="text/plain",
                use_container_width=True
            )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">StudyMate · a quiet place to work through your reading</div>',
    unsafe_allow_html=True
)

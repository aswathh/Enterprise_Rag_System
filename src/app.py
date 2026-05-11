# Streamlit UI

import os
import shutil
import time
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

from rag_pipeline import (
    load_pdf,
    split_documents,
    create_vectorstore,
    build_rag_chain,
    ask_question,
    VECTORSTORE_DIR
)

# ── Constants ─────────────────────────────────
DATA_DIR = Path("./data")

# ── Page Config ───────────────────────────────
st.set_page_config(
    page_title="Document Intelligence System",
    page_icon="📄",
    layout="wide"
)

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.title("📂 Upload Documents")

    uploaded = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:
        DATA_DIR.mkdir(exist_ok=True)
        for f in uploaded:
            (DATA_DIR / f.name).write_bytes(f.getbuffer())
        st.success(f"{len(uploaded)} file(s) uploaded!")

    st.divider()

    if st.button("⚡ Index Documents", type="primary", use_container_width=True):
        if not uploaded:
            st.error("Upload a PDF first!")
            st.stop()
        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY missing in .env!")
            st.stop()

        # Clear old vectorstore
        if Path(VECTORSTORE_DIR).exists():
            shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)
            time.sleep(1)  # Windows file lock fix

        with st.spinner("Indexing documents..."):
            all_docs = []
            for f in uploaded:
                all_docs.extend(load_pdf(str(DATA_DIR / f.name)))
            chunks      = split_documents(all_docs)
            vectorstore = create_vectorstore(chunks)
            chain       = build_rag_chain(vectorstore)
            st.session_state["chain"] = chain

        st.success("Ready! Ask questions below ✅")

    st.divider()

    # API Key status
    if os.getenv("GROQ_API_KEY"):
        st.success("API Key found ✅")
    else:
        st.error("GROQ_API_KEY missing!")
        st.code("Add to .env:\nGROQ_API_KEY=gsk_your_key")

    st.divider()
    st.markdown("**Pipeline:**")
    st.code("PDF → Chunks → Embeddings\n→ ChromaDB → Groq LLM → Answer")

# ── Main ──────────────────────────────────────
st.title("📄 Document Intelligence System")
st.caption("Upload PDFs → Ask questions → Get cited answers")

# Chat history init
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
CONFIDENCE_COLOR = {"High": "green", "Medium": "orange", "Low": "red"}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            for c in msg.get("citations", []):
                st.caption(f"📄 {c['file']} — Page {c['page']}")
            label = msg.get("label", "Low")
            score = msg.get("score", 0)
            color = CONFIDENCE_COLOR[label]
            st.markdown(f"Confidence: :{color}[{label} ({score}%)]")

# Chat input
if question := st.chat_input("Ask about your documents..."):
    if "chain" not in st.session_state:
        st.error("Index documents first!")
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            start   = time.time()
            result  = ask_question(st.session_state["chain"], question)
            elapsed = round(time.time() - start, 1)

        st.markdown(result["answer"])

        for c in result["citations"]:
            st.caption(f"📄 {c['file']} — Page {c['page']}")

        label = result["confidence_label"]
        score = result["confidence_score"]
        color = CONFIDENCE_COLOR[label]
        st.markdown(
            f"Confidence: :{color}[{label} ({score}%)] | {elapsed}s"
        )

    st.session_state.messages.append({
        "role":      "assistant",
        "content":   result["answer"],
        "citations": result["citations"],
        "label":     result["confidence_label"],
        "score":     result["confidence_score"]
    })

# Empty state
if not st.session_state.messages:
    st.info("Upload PDFs in the sidebar and click Index Documents to start!")
    st.markdown("""
**Example questions:**
- How many leave days do employees get?
- What is the work from home policy?
- What is the notice period for resignation?
    """)
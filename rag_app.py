import os
import shutil
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Load environment
load_dotenv()

# ── Page Config ──────────────────────────────
st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Intelligence System")
st.caption("Upload PDFs and ask questions with source citations")

# ── Helpers ──────────────────────────────────
DATA_DIR   = Path("data")
CHROMA_DIR = Path("chroma_db")

def build_chain(files):
    DATA_DIR.mkdir(exist_ok=True)
    
    # Force delete old chroma_db
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        import time
        time.sleep(1)
    
    for f in files:
        (DATA_DIR / f.name).write_bytes(f.getbuffer())

    docs = []
    for f in files:
        docs.extend(PyPDFLoader(str(DATA_DIR / f.name)).load())

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    ).split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    llm = ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1024
    )

    prompt = PromptTemplate(
        template="""Use ONLY the context below to answer.
If not found, say: I do not have enough information.
Do NOT make up any information.

Context: {context}
Question: {question}
Answer:""",
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return RunnableParallel(
        answer=(
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        ),
        source_documents=retriever
    )


# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.header("📂 Upload PDF")
    files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("⚡ Index Documents", type="primary", use_container_width=True):
        if not files:
            st.error("Upload a PDF first!")
            st.stop()
        with st.spinner("Processing... please wait..."):
            st.session_state["chain"] = build_chain(files)
        st.success(f"{len(files)} file(s) indexed! Ask questions below ✅")


# ── Chat Interface ────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask about your documents..."):
    if "chain" not in st.session_state:
        st.error("Please index documents first!")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            result = st.session_state["chain"].invoke(question)

        st.markdown(result["answer"])

        # Show sources
        seen = set()
        for doc in result.get("source_documents", []):
            page = int(doc.metadata.get("page", 0)) + 1
            src  = Path(doc.metadata.get("source", "doc")).name
            key  = f"{src}_{page}"
            if key not in seen:
                st.caption(f"📄 {src} — Page {page}")
                seen.add(key)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })
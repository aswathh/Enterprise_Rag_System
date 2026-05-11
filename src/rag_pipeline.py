# ============================================
# src/rag_pipeline.py — Core RAG Logic
# ============================================

import os
import shutil
import time
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

load_dotenv()

# ── Constants ─────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
VECTORSTORE_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"


# ── Helpers ───────────────────────────────────
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def _format_docs(docs) -> str:
    """Format retrieved docs into single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def _safe_delete(path: str):
    """Force delete folder — handles Windows file locks."""
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
        time.sleep(1)


# ── Core Functions ────────────────────────────
def load_pdf(pdf_path: str) -> list:
    """Load a single PDF and return pages."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    return PyPDFLoader(str(path)).load()


def load_pdfs_from_folder(folder: str = "./data") -> list:
    """Load all PDFs from a folder."""
    folder_path = Path(folder)
    folder_path.mkdir(exist_ok=True)

    pdf_files = list(folder_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {folder}")

    all_docs = []
    for pdf in pdf_files:
        all_docs.extend(load_pdf(str(pdf)))
    return all_docs


def split_documents(documents: list) -> list:
    """Split documents into chunks."""
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    ).split_documents(documents)


def create_vectorstore(chunks: list) -> Chroma:
    """Create embeddings and store in ChromaDB."""
    _safe_delete(VECTORSTORE_DIR)
    return Chroma.from_documents(
        documents=chunks,
        embedding=_get_embeddings(),
        persist_directory=VECTORSTORE_DIR
    )


def load_vectorstore() -> Chroma:
    """Load existing ChromaDB from disk."""
    if not os.path.exists(VECTORSTORE_DIR):
        raise FileNotFoundError("No ChromaDB found — index documents first!")
    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=_get_embeddings()
    )


def build_rag_chain(vectorstore: Chroma):
    """Build modern LCEL RAG chain."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in .env!")

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.1,
        max_tokens=1024
    )

    prompt = PromptTemplate(
        template="""You are a helpful document assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say: I do not have enough information in the documents.
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

    return RunnableParallel(
        answer=(
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        ),
        source_documents=retriever
    )


def ask_question(rag_chain, question: str) -> dict:
    """Ask a question and return answer with citations."""
    result  = rag_chain.invoke(question)
    sources = result.get("source_documents", [])

    citations = []
    seen = set()
    for doc in sources:
        src  = doc.metadata.get("source", "Unknown")
        page = int(doc.metadata.get("page", 0)) + 1
        key  = f"{src}_{page}"
        if key not in seen:
            citations.append({
                "file":    Path(src).name,
                "page":    page,
                "snippet": doc.page_content[:150]
            })
            seen.add(key)

    score = min(len(citations) * 25, 100)
    label = "High" if score >= 75 else "Medium" if score >= 50 else "Low"

    return {
        "answer":           result["answer"],
        "citations":        citations,
        "confidence_score": score,
        "confidence_label": label
    }


def initialize_system(pdf_path: str = None, data_folder: str = "./data"):
    """Full setup — load PDFs, create vectorstore, build chain."""
    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
        vectorstore = load_vectorstore()
    else:
        docs        = load_pdf(pdf_path) if pdf_path else load_pdfs_from_folder(data_folder)
        chunks      = split_documents(docs)
        vectorstore = create_vectorstore(chunks)
    return build_rag_chain(vectorstore)


# ── Quick Test ────────────────────────────────
if __name__ == "__main__":
    chain  = initialize_system(data_folder="./data")
    result = ask_question(chain, "What is this document about?")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence_label']} ({result['confidence_score']}%)")
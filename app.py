"""CodeSage - Streamlit application (Phase 1I).

AI-based codebase understanding system using Retrieval-Augmented Generation (RAG).
Provides a web interface to:
1. Upload and index Python source files or ZIP archives.
2. Ask questions about the codebase grounded in retrieved code chunks.
3. Inspect AI explanations alongside the exact code sources and line numbers.
"""

from __future__ import annotations

import io
import os
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from config import settings
from src.chunker import CodeChunker
from src.embedder import Embedder
from src.llm import LLM, LLMConnectionError, LLMError, LLMResponseError
from src.parser import CodeParser, ParsedFile
from src.repository_graph import RepositoryGraph, build_repository_graph
from src.graph_retrieval import build_graph_chunk_lookup
from src.rag_pipeline import (
    NO_CONTEXT_MESSAGE,
    RAGInputError,
    RAGPipeline,
    RAGPipelineError,
    RAGResult,
)
from src.retriever import RetrievalResult, Retriever
from src.utils import ensure_directories, get_logger, safe_filename, setup_logging
from src.vectordb import VectorStore

logger = get_logger("app")

SECTIONS = ["Upload Code", "Ask Code", "Documentation"]
ALLOWED_TYPES = ["py", "zip"]


# --------------------------------------------------------------------------- #
# Helpers & Business Logic (testable outside Streamlit runtime)
# --------------------------------------------------------------------------- #
def extract_safe_zip(zip_bytes: bytes, target_dir: Path) -> list[Path]:
    """Extract .py files from a ZIP archive safely preventing path traversal."""
    extracted_paths: list[Path] = []
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue

            path_obj = Path(member.filename)
            if path_obj.suffix.lower() != ".py":
                continue

            # Prevent Zip Slip / path traversal attacks
            dest = (target_dir / member.filename).resolve()
            if not str(dest).startswith(str(target_dir)):
                logger.warning("Path traversal attempt detected in ZIP: %s", member.filename)
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)

            extracted_paths.append(dest)

    return extracted_paths


def save_uploaded_files(uploaded_files: list[Any], target_dir: Path) -> list[Path]:
    """Save uploaded files (.py and .zip) as inert data to target_dir.

    Never executes or imports uploaded code.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    stored_paths: list[Path] = []

    for f in uploaded_files:
        name = getattr(f, "name", "uploaded_file")
        content: bytes = f.getvalue() if hasattr(f, "getvalue") else f.read()

        if name.lower().endswith(".zip"):
            zip_extract_dir = target_dir / safe_filename(name).removesuffix(".zip")
            extracted = extract_safe_zip(content, zip_extract_dir)
            stored_paths.extend(extracted)
        elif name.lower().endswith(".py"):
            dest = target_dir / safe_filename(name)
            dest.write_bytes(content)
            stored_paths.append(dest)

    return stored_paths


def index_codebase_files(
    file_paths: list[Path],
    embedder: Embedder,
    vector_store: VectorStore,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    graph_context_callback: Optional[Callable[[RepositoryGraph, dict[str, RetrievalResult]], None]] = None,
) -> tuple[int, int]:
    """Parse, chunk, embed, and store code files in ChromaDB."""
    py_files = [p for p in file_paths if p.is_file() and p.suffix.lower() == ".py"]
    if not py_files:
        raise ValueError("No valid Python (.py) source files found to index.")

    parser = CodeParser()
    chunker = CodeChunker()

    if progress_callback:
        progress_callback("Parsing codebase files...", 0.1)

    all_chunks = []
    parsed_sources: list[ParsedFile] = []
    parsed_files = 0
    for p in py_files:
        try:
            parsed = parser.parse_file(p)
            chunks = chunker.chunk(parsed)
            all_chunks.extend(chunks)
            parsed_sources.append(parsed)
            parsed_files += 1
        except Exception as exc:
            logger.warning("Could not parse file %s: %s", p, exc)

    if not all_chunks:
        raise ValueError("No code chunks could be extracted from the provided files.")

    if progress_callback:
        progress_callback(f"Generating embeddings for {len(all_chunks)} chunks...", 0.4)

    embeddings = embedder.embed_chunks(all_chunks)

    if progress_callback:
        progress_callback("Storing vectors in ChromaDB...", 0.8)

    vector_store.clear()
    vector_store.add_chunks(all_chunks, embeddings)

    if graph_context_callback is not None:
        try:
            common_root = Path(os.path.commonpath([str(Path(item.path).resolve().parent) for item in parsed_sources]))
            graph = build_repository_graph(parsed_sources, root=common_root)
            chunk_lookup = build_graph_chunk_lookup(graph, all_chunks)
            graph_context_callback(graph, chunk_lookup)
        except Exception as exc:
            # Vector retrieval remains available if optional graph preparation fails.
            logger.warning("Could not prepare graph retrieval context: %s", exc)

    if progress_callback:
        progress_callback("Indexing complete!", 1.0)

    return parsed_files, len(all_chunks)


def format_source_label(source: RetrievalResult, index: int) -> str:
    """Format a clean summary label for a retrieved source expander."""
    file_label = source.file_path or "unknown_file.py"
    type_name = f"{source.chunk_type}: {source.name}" if source.name else source.chunk_type or "code"
    lines_label = f"Lines {source.start_line}-{source.end_line}" if source.start_line and source.end_line else ""
    parts = [f"Source {index}: {file_label}", type_name]
    if lines_label:
        parts.append(lines_label)
    return " — ".join(parts)


# --------------------------------------------------------------------------- #
# Cached Backend Singletons
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_embedder() -> Embedder:
    """Load the SentenceTransformer embedder once across Streamlit reruns."""
    return Embedder()


@st.cache_resource
def get_vector_store() -> VectorStore:
    """Initialize persistent ChromaDB vector store once across Streamlit reruns."""
    return VectorStore()


@st.cache_resource
def get_llm() -> LLM:
    """Initialize local Ollama LLM interface once across Streamlit reruns."""
    return LLM()


def get_pipeline() -> RAGPipeline:
    """Construct RAGPipeline with the cached backend resources."""
    embedder = get_embedder()
    store = get_vector_store()
    llm = get_llm()
    graph = st.session_state.get("repository_graph")
    graph_chunk_lookup = st.session_state.get("graph_chunk_lookup")
    retriever_kwargs: dict[str, Any] = {}
    if graph is not None and graph_chunk_lookup:
        retriever_kwargs["repository_graph"] = graph
        retriever_kwargs["graph_chunk_lookup"] = graph_chunk_lookup
    retriever = Retriever(embedder=embedder, vector_store=store, **retriever_kwargs)
    return RAGPipeline(retriever=retriever, llm=llm)


# --------------------------------------------------------------------------- #
# UI Render Sections
# --------------------------------------------------------------------------- #
def render_upload() -> None:
    """Upload and index codebase section."""
    st.subheader("Upload & Index Codebase")
    st.write(
        "Upload one or more Python files (`.py`) or a ZIP archive containing Python source code. "
        "Uploaded files are treated purely as inert data and are never executed."
    )

    uploaded = st.file_uploader(
        "Select Python source files or ZIP archive",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
        help="Upload individual .py files or a .zip archive.",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        index_clicked = st.button("Index Codebase", type="primary")

    if index_clicked:
        if not uploaded:
            st.warning("Please upload at least one Python file or ZIP archive before indexing.")
            return

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(msg: str, frac: float) -> None:
            status_text.text(msg)
            progress_bar.progress(frac)

        try:
            update_progress("Saving uploaded files...", 0.05)
            upload_dir = settings.uploads_dir
            saved_paths = save_uploaded_files(uploaded, upload_dir)

            embedder = get_embedder()
            store = get_vector_store()

            def save_graph_context(graph: RepositoryGraph, chunk_lookup: dict[str, RetrievalResult]) -> None:
                st.session_state["repository_graph"] = graph
                st.session_state["graph_chunk_lookup"] = chunk_lookup

            num_files, num_chunks = index_codebase_files(
                saved_paths,
                embedder,
                store,
                progress_callback=update_progress,
                graph_context_callback=save_graph_context,
            )

            st.session_state["indexed"] = True
            st.session_state["num_files"] = num_files
            st.session_state["num_chunks"] = num_chunks
            st.session_state["uploaded_file_names"] = [f.name for f in uploaded]

            status_text.empty()
            progress_bar.empty()
            st.success(f"Indexed {num_files} files into {num_chunks} code chunks successfully!")
            st.info("You can now navigate to **Ask Code** to query your codebase.")
        except Exception as exc:
            status_text.empty()
            progress_bar.empty()
            logger.error("Indexing failed: %s", exc)
            st.error(f"Indexing error: {exc}")

    # Display current index status if available
    if st.session_state.get("indexed"):
        st.markdown("---")
        st.write(
            f"**Current Index:** {st.session_state.get('num_files', 0)} files, "
            f"{st.session_state.get('num_chunks', 0)} chunks ready."
        )


def render_ask() -> None:
    """Ask Code section: natural language query input, answer, and sources display."""
    st.subheader("Ask Codebase")

    if not st.session_state.get("indexed"):
        st.info("No codebase indexed yet. Please upload and index your Python code first in **Upload Code**.")

    question = st.text_input(
        "Ask a question about your codebase",
        placeholder="e.g. How is discount calculated? Or: Where is authentication handled?",
        key="rag_question_input",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        ask_clicked = st.button("Ask CodeSage", type="primary")

    if ask_clicked:
        if not question or not question.strip():
            st.warning("Please enter a question about your codebase.")
            return

        if not st.session_state.get("indexed"):
            st.warning("Please upload and index a codebase before asking questions.")
            return

        with st.spinner("Retrieving relevant code and generating explanation..."):
            try:
                pipeline = get_pipeline()
                result = pipeline.answer(question.strip())
                st.session_state["latest_result"] = result
            except (LLMConnectionError, ConnectionError) as exc:
                logger.error("Ollama connection failed: %s", exc)
                st.error(
                    "Unable to connect to the local Ollama service. "
                    "Please make sure Ollama is running (`http://localhost:11434`) and the configured model is available."
                )
                return
            except LLMResponseError as exc:
                logger.error("Ollama response error: %s", exc)
                st.error(f"Ollama server returned an error: {exc}")
                return
            except RAGInputError as exc:
                st.warning(str(exc))
                return
            except (RAGPipelineError, LLMError) as exc:
                logger.error("Pipeline failure: %s", exc)
                st.error(f"Pipeline error: {exc}")
                return
            except Exception as exc:
                logger.error("Unexpected error in ask: %s", exc)
                st.error(f"An unexpected error occurred: {exc}")
                return

    # Render latest answer & sources from session state if available
    latest_result: Optional[RAGResult] = st.session_state.get("latest_result")
    if latest_result:
        st.markdown("### Answer")
        st.markdown(latest_result.answer)

        st.markdown("### Sources")
        if latest_result.sources:
            for idx, source in enumerate(latest_result.sources, start=1):
                label = format_source_label(source, idx)
                with st.expander(label, expanded=(idx == 1)):
                    meta_col1, meta_col2 = st.columns(2)
                    with meta_col1:
                        st.markdown(f"**File:** `{source.file_path or 'unknown'}`")
                        st.markdown(f"**Type:** `{source.chunk_type or 'code'}`")
                    with meta_col2:
                        name_display = source.qualified_name or source.name or "N/A"
                        st.markdown(f"**Name:** `{name_display}`")
                        if source.distance is not None:
                            st.markdown(f"**Distance:** `{source.distance:.4f}`")
                    st.code(source.content, language="python")
        else:
            st.info("No code sources were retrieved for this query.")


def render_docs() -> None:
    """Documentation and system overview section."""
    st.subheader("Documentation")
    st.markdown(
        "**CodeSage** is an AI-based codebase understanding system using "
        "Retrieval-Augmented Generation (RAG)."
    )
    st.markdown(
        "**Official Workflow:**  \n"
        "1. **Code Upload:** Accepts Python files or ZIP archives safely without executing code.  \n"
        "2. **AST Parsing:** Parses Python source into functions, classes, and methods via Python AST.  \n"
        "3. **Code Chunking:** Creates deterministic, structure-aware code chunks.  \n"
        "4. **Embedding:** Generates vector representations using Sentence Transformers (`all-MiniLM-L6-v2`).  \n"
        "5. **Vector Storage:** Stores vectors and metadata in ChromaDB.  \n"
        "6. **Retrieval:** Performs cosine distance similarity retrieval to find relevant chunks.  \n"
        "7. **RAG Explanation:** Grounds prompts with retrieved context and asks local Ollama to explain.  \n"
        "8. **Result Display:** Displays the answer and precise source citations (file, lines, symbol)."
    )
    st.markdown(
        f"**Active Configuration:**  \n"
        f"- Embedding Model: `{settings.embedding_model}`  \n"
        f"- ChromaDB Path: `{settings.chroma_path}`  \n"
        f"- Local Ollama Host: `{settings.ollama_host}`  \n"
        f"- Ollama Model: `{settings.ollama_model}`"
    )
    st.markdown(
        "**Security Guarantee:** Uploaded code is strictly treated as text data. "
        "No uploaded scripts or generated responses are ever executed (`no eval/exec/subprocess`)."
    )


def main() -> None:
    """Application entry point."""
    st.set_page_config(page_title="CodeSage", page_icon="🧠", layout="wide")
    ensure_directories()
    setup_logging()

    st.title("CodeSage")
    st.caption("AI-Based Codebase Understanding System using Retrieval-Augmented Generation (RAG)")

    # Sidebar Navigation & Status
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Select Section", SECTIONS, index=0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Index Status")
    if st.session_state.get("indexed"):
        st.sidebar.success(
            f"Indexed: {st.session_state.get('num_files', 0)} files, "
            f"{st.session_state.get('num_chunks', 0)} chunks"
        )
    else:
        st.sidebar.info("No codebase indexed yet")

    if section == "Upload Code":
        render_upload()
    elif section == "Ask Code":
        render_ask()
    elif section == "Documentation":
        render_docs()


if __name__ == "__main__":
    main()

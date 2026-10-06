"""CodeSage - Streamlit application foundation.

The RAG pipeline is NOT implemented yet. This UI is a structural shell only.
"""

from __future__ import annotations

import streamlit as st

from config import settings
from src.utils import ensure_directories, save_uploaded_bytes, setup_logging

SECTIONS = ["Upload Code", "Ask Code", "Documentation"]
ALLOWED_TYPES = ["py", "java", "js", "ts", "c", "cpp", "h", "cs", "go", "rb", "php", "txt", "md"]


def render_status_banner() -> None:
    """Show that the RAG pipeline is not implemented yet."""
    st.warning("The RAG pipeline is not implemented yet. This is the project foundation only.")


def render_upload() -> None:
    """Upload section: files are saved as inert data and never executed."""
    st.subheader("Upload Code")
    files = st.file_uploader("Select source code files", type=ALLOWED_TYPES, accept_multiple_files=True)
    if files and st.button("Save uploaded files"):
        for f in files:
            path = save_uploaded_bytes(f.name, f.getvalue())
            st.write(f"Saved `{path.name}`")
        st.info("Files are stored only. Parsing, chunking and indexing are not implemented yet.")


def render_ask() -> None:
    """Ask section placeholder."""
    st.subheader("Ask Code")
    st.text_input("Your question about the code", disabled=True, placeholder="Available once the RAG pipeline is built")
    st.button("Ask", disabled=True)
    st.caption("Retrieval and LLM explanation are not implemented yet.")


def render_docs() -> None:
    """Documentation section."""
    st.subheader("Documentation")
    st.markdown(
        "**Workflow:** Code Upload → Parsing & Chunking → Embedding Generation → "
        "Vector Database Storage → Query Input → Retrieval → LLM Explanation → Result Display"
    )
    st.markdown(
        f"**Configuration:** embedding model `{settings.embedding_model}`, "
        f"Ollama `{settings.ollama_model}` at `{settings.ollama_host}`"
    )
    st.markdown("**Status:** foundation only; see README.md for details.")


def main() -> None:
    """Application entry point."""
    st.set_page_config(page_title="CodeSage", page_icon="🧠", layout="wide")
    ensure_directories()
    setup_logging()
    st.title("CodeSage")
    st.write("AI-based code understanding system using Retrieval-Augmented Generation (RAG).")
    render_status_banner()
    section = st.sidebar.radio("Navigation", SECTIONS)
    {"Upload Code": render_upload, "Ask Code": render_ask, "Documentation": render_docs}[section]()


if __name__ == "__main__":
    main()

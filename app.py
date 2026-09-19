"""Streamlit front end for the AI Legal Document Assistant."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from services.exceptions import DocumentAssistantError
from services.rag import IngestionResult, RAGPipeline
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="AI Legal Document Assistant",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading models…")
def load_pipeline() -> RAGPipeline:
    """Build the RAG pipeline once per server process."""
    pipeline = RAGPipeline()
    pipeline.load_existing_index()
    return pipeline


def init_session_state(pipeline: RAGPipeline) -> None:
    """Seed session state, reusing any index restored from disk."""
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("document_name", pipeline.document_name)
    st.session_state.setdefault("ingestion", None)


def ingest_upload(pipeline: RAGPipeline, uploaded_file: Any) -> IngestionResult:
    """Write an upload to a temporary file and index it.

    The temporary file is always removed, including when indexing fails.
    """
    suffix = Path(uploaded_file.name).suffix
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = Path(tmp.name)

        return pipeline.ingest_document(temp_path, document_name=uploaded_file.name)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def render_sidebar(pipeline: RAGPipeline) -> None:
    """Show configuration and index status."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.caption("Set through environment variables or a .env file.")

        st.markdown(
            f"""
            - **Answer model:** `{settings.ollama_model}`
            - **Embedding model:** `{settings.embedding_model}`
            - **Chunk size / overlap:** `{settings.chunk_size}` / `{settings.chunk_overlap}`
            - **Chunks retrieved:** `{settings.top_k}`
            - **Max upload size:** `{settings.max_upload_mb} MB`
            """
        )

        st.divider()
        st.header("📊 Index status")

        if pipeline.is_ready:
            st.success(f"{pipeline.chunk_count} chunks indexed")
            if pipeline.document_name:
                st.caption(f"Document: {pipeline.document_name}")
        else:
            st.info("No document indexed yet.")

        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Render retrieved chunks inside a collapsed expander."""
    if not sources:
        return

    with st.expander(f"📚 Retrieved sources ({len(sources)})"):
        for source in sources:
            st.markdown(f"**Chunk {source['id']}** — similarity `{source['score']:.4f}`")
            st.write(source["chunk"])
            st.divider()


def render_history() -> None:
    """Replay the conversation so far."""
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            render_sources(turn["sources"])


def handle_question(pipeline: RAGPipeline, question: str) -> None:
    """Answer a question and append the exchange to the history."""
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the document…"):
                stream, sources = pipeline.ask_stream(question)

            # st.write_stream renders tokens as they arrive and returns the
            # complete text, so the stored answer keeps its markdown formatting.
            answer = st.write_stream(stream)
        except DocumentAssistantError as exc:
            st.error(str(exc))
            return
        except Exception:
            logger.exception("Unexpected failure while answering a question")
            st.error("Something went wrong while answering. Check the server logs for details.")
            return

        render_sources(sources)

    st.session_state.chat_history.append(
        {"question": question, "answer": answer, "sources": sources}
    )


def main() -> None:
    pipeline = load_pipeline()
    init_session_state(pipeline)

    st.title("📄 AI Legal Document Assistant")
    st.caption("Upload a PDF or DOCX file and ask questions about its contents.")

    render_sidebar(pipeline)

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx"],
        help=f"PDF or DOCX, up to {settings.max_upload_mb} MB.",
    )

    if uploaded_file is not None:
        if uploaded_file.size > settings.max_upload_bytes:
            st.error(
                f"{uploaded_file.name} is {uploaded_file.size / 1_048_576:.1f} MB, "
                f"which exceeds the {settings.max_upload_mb} MB limit."
            )
        elif uploaded_file.name != st.session_state.document_name:
            # Re-index only when the file actually changed; Streamlit reruns the
            # whole script on every interaction.
            try:
                with st.spinner(f"Indexing {uploaded_file.name}…"):
                    result = ingest_upload(pipeline, uploaded_file)
            except DocumentAssistantError as exc:
                st.error(str(exc))
            except Exception:
                logger.exception("Unexpected failure while indexing %s", uploaded_file.name)
                st.error("Could not index that document. Check the server logs for details.")
            else:
                st.session_state.document_name = result.document_name
                st.session_state.ingestion = result
                st.session_state.chat_history = []
                st.rerun()

    if pipeline.is_ready and st.session_state.document_name:
        st.success(f"✅ {st.session_state.document_name} is ready — ask a question below.")

    render_history()

    question = st.chat_input(
        "Ask a question about the document…",
        disabled=not pipeline.is_ready,
    )

    if question:
        handle_question(pipeline, question)


if __name__ == "__main__":
    main()

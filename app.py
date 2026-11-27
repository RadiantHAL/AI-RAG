"""Streamlit UI for the AI & Machine Learning Knowledge Assistant (RAG Chatbot).

Visitors enter their own Gemini API key in the sidebar (stored only in
`st.session_state`), optionally upload PDF/TXT/MD files (or use the 3 default
AI/ML papers in `knowledge_base/`), and ask questions grounded strictly in
those documents with inline citations.
"""

from __future__ import annotations

import streamlit as st

from rag_core import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DocumentLoadError,
    InMemoryVectorStore,
    InvalidAPIKeyError,
    RAGError,
    RetrievedChunk,
    chunk_pages,
    create_genai_client,
    generate_grounded_answer,
    load_sample_documents,
    load_uploaded_files,
)

# 4 suggested questions tailored to the default AI & ML knowledge base.
SUGGESTED_QUESTIONS = [
    "What is the main advantage of the Transformer architecture over recurrent models?",
    "How does Retrieval-Augmented Generation (RAG) combine parametric and non-parametric memory?",
    "What are the two pre-training tasks used in BERT?",
    "How does scaled dot-product attention work and why is it scaled?",
]


def _init_session_state() -> None:
    """Initialize required keys in Streamlit session state."""
    defaults = {
        "api_key": "",
        "messages": [],
        "vector_store": None,
        "index_signature": None,
        "indexed_doc_names": [],
        "indexed_chunk_count": 0,
        "pending_question": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _build_index_signature(
    uploaded_files: list,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple:
    """Create a lightweight signature to detect when the index needs rebuilding."""
    if uploaded_files:
        file_sig = tuple((f.name, f.size) for f in uploaded_files)
        mode = "uploaded"
    else:
        file_sig = ("sample_knowledge_base",)
        mode = "sample"
    return (mode, file_sig, chunk_size, chunk_overlap)


def _format_sources_for_ui(retrieved: list[RetrievedChunk]) -> list[dict]:
    """Convert RetrievedChunk objects into serializable dicts for session state."""
    return [
        {
            "num": idx,
            "source": item.chunk.source,
            "page": item.chunk.page,
            "score": round(item.score, 3),
            "excerpt": item.chunk.text,
        }
        for idx, item in enumerate(retrieved, start=1)
    ]


def _render_sources(sources: list[dict]) -> None:
    """Display source citations and retrieved excerpts below an assistant answer."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)} retrieved chunks)"):
        for src in sources:
            st.markdown(
                f"**[{src['num']}] `{src['source']}` — Page {src['page']}** "
                f"(similarity: `{src['score']:.3f}`)"
            )
            st.caption(src["excerpt"][:400] + ("..." if len(src["excerpt"]) > 400 else ""))


def _ensure_vector_store(
    api_key: str,
    uploaded_files: list,
    chunk_size: int,
    chunk_overlap: int,
) -> InMemoryVectorStore | None:
    """Build or reuse the in-memory vector store for the current documents."""
    signature = _build_index_signature(uploaded_files, chunk_size, chunk_overlap)

    if (
        st.session_state.vector_store is not None
        and st.session_state.index_signature == signature
    ):
        return st.session_state.vector_store

    client = create_genai_client(api_key)

    if uploaded_files:
        file_tuples = [(f.name, f.getvalue()) for f in uploaded_files]
        pages, warnings = load_uploaded_files(file_tuples)
    else:
        pages, warnings = load_sample_documents()

    for warning in warnings:
        st.warning(warning)

    if not pages:
        raise DocumentLoadError(
            "No readable text could be extracted from the selected documents."
        )

    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise DocumentLoadError("Document chunking produced zero chunks.")

    status_placeholder = st.empty()
    vector_store = InMemoryVectorStore.from_chunks(
        client=client,
        chunks=chunks,
        embed_model=DEFAULT_EMBED_MODEL,
        status_callback=lambda msg: status_placeholder.info(msg),
    )
    status_placeholder.empty()

    st.session_state.vector_store = vector_store
    st.session_state.index_signature = signature
    st.session_state.indexed_doc_names = sorted({p.source for p in pages})
    st.session_state.indexed_chunk_count = len(chunks)
    return vector_store


def _handle_user_question(
    question: str,
    api_key: str,
    uploaded_files: list,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
) -> None:
    """Run retrieval + grounded generation for a user question and update chat history."""
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and generating grounded answer..."):
                vector_store = _ensure_vector_store(
                    api_key=api_key,
                    uploaded_files=uploaded_files,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                client = create_genai_client(api_key)
                retrieved = vector_store.search(
                    client=client,
                    query=question,
                    top_k=top_k,
                    embed_model=DEFAULT_EMBED_MODEL,
                )
                answer = generate_grounded_answer(
                    client=client,
                    question=question,
                    retrieved=retrieved,
                    chat_history=st.session_state.messages[:-1],
                    model=DEFAULT_CHAT_MODEL,
                )
                sources = _format_sources_for_ui(retrieved)

            st.markdown(answer)
            _render_sources(sources)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        except (InvalidAPIKeyError, DocumentLoadError, RAGError) as exc:
            error_msg = str(exc)
            st.error(error_msg)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"⚠️ {error_msg}",
                    "sources": [],
                }
            )


def main() -> None:
    st.set_page_config(
        page_title="AI & ML Knowledge Assistant",
        page_icon="🤖",
        layout="wide",
    )
    _init_session_state()

    st.title("🤖 AI & Machine Learning Knowledge Assistant")
    st.markdown(
        "Ask questions grounded **strictly** in your uploaded documents or the built-in "
        "AI/ML research papers. Every answer includes inline citations (`[1]`, `[2]`) and "
        "refuses to guess when information is unavailable."
    )

    # --- Sidebar: API Key, Document Upload, Chunking & Retrieval Settings ---
    with st.sidebar:
        st.header("🔑 Gemini API Key")
        api_key_input = st.text_input(
            "Enter your Gemini API key",
            value=st.session_state.api_key,
            type="password",
            placeholder="AIza...",
            help="Stored in your browser session only. Never saved or logged.",
        )
        st.session_state.api_key = api_key_input
        st.caption(
            "Get a free key from [Google AI Studio](https://aistudio.google.com/apikey)."
        )

        st.divider()
        st.header("📄 Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload your own documents (optional)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="If no files are uploaded, the 3 sample AI/ML papers are used.",
        )
        if uploaded_files:
            st.info(f"Using **{len(uploaded_files)}** uploaded file(s) (in-memory only).")
        else:
            st.info("Using **3 sample AI/ML PDFs** from `knowledge_base/`.")

        st.divider()
        st.header("⚙️ Chunking & Retrieval")
        chunk_size = st.slider(
            "Chunk size (characters)",
            min_value=400,
            max_value=2000,
            value=1000,
            step=100,
        )
        max_overlap = min(400, chunk_size - 100)
        chunk_overlap = st.slider(
            "Chunk overlap (characters)",
            min_value=0,
            max_value=max_overlap,
            value=min(200, max_overlap),
            step=50,
        )
        top_k = st.slider(
            "Top-K retrieved chunks",
            min_value=2,
            max_value=8,
            value=5,
            step=1,
        )

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Rebuild Index", use_container_width=True):
                st.session_state.vector_store = None
                st.session_state.index_signature = None
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        if st.session_state.indexed_doc_names:
            st.caption(
                f"Indexed **{st.session_state.indexed_chunk_count}** chunks from: "
                + ", ".join(st.session_state.indexed_doc_names)
            )

    # --- Extra Feature: 4 Clickable Suggested Questions ---
    st.subheader("💡 Suggested Questions")
    q_cols = st.columns(2)
    for idx, suggestion in enumerate(SUGGESTED_QUESTIONS):
        with q_cols[idx % 2]:
            if st.button(suggestion, key=f"suggested_{idx}", use_container_width=True):
                st.session_state.pending_question = suggestion

    st.divider()

    # --- Conversation History ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    # --- Chat Input (or Suggested Question Submission) ---
    user_input = st.chat_input("Ask a question about the documents...")
    question_to_run = st.session_state.pending_question or user_input
    st.session_state.pending_question = None

    if question_to_run:
        _handle_user_question(
            question=question_to_run,
            api_key=st.session_state.api_key,
            uploaded_files=uploaded_files or [],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
        )


if __name__ == "__main__":
    main()

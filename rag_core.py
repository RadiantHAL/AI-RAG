"""Core RAG (Retrieval-Augmented Generation) pipeline for the AI & ML Knowledge Assistant.

This module is intentionally UI-agnostic so it can be imported by Streamlit (`app.py`)
or tested directly from Python scripts. It handles:
1. Document loading (PDF, TXT, Markdown) with filename and page metadata
2. Configurable text chunking with overlap
3. Gemini embeddings (`gemini-embedding-001`) via the official `google-genai` SDK
4. In-memory cosine similarity vector search using NumPy
5. Grounded answer generation (`gemini-3.8-flash`) with inline citations and strict fallback
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NoReturn

import numpy as np
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pypdf import PdfReader

# Resolve paths relative to this file so they work both locally and on Streamlit Cloud.
KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent / "knowledge_base"

# Current supported Gemini API models (non-deprecated).
DEFAULT_CHAT_MODEL = "gemini-3.8-flash"
DEFAULT_EMBED_MODEL = "gemini-embedding-001"

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
FALLBACK_RESPONSE = "I don’t know based on the provided documents."


class RAGError(Exception):
    """Base exception for user-facing RAG pipeline errors."""


class DocumentLoadError(RAGError):
    """Raised when no valid readable text can be loaded from documents."""


class InvalidAPIKeyError(RAGError):
    """Raised when the provided Gemini API key is missing or rejected."""


@dataclass(frozen=True)
class DocumentPage:
    """Represents extracted text from a single page (or file for TXT/MD)."""

    source: str
    page: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    """Represents a chunk of text with preserved source metadata."""

    chunk_id: int
    source: str
    page: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    """Represents a retrieved chunk and its cosine similarity score."""

    chunk: TextChunk
    score: float


def _clean_text(text: str) -> str:
    """Normalize extracted text while preserving paragraph breaks."""
    text = text.replace("\x00", "")
    # Collapse horizontal whitespace on each line and limit consecutive blank lines.
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def load_pdf_bytes(filename: str, data: bytes) -> list[DocumentPage]:
    """Extract text page-by-page from PDF bytes using PyPDF."""
    if not data:
        raise DocumentLoadError(f"'{filename}' is empty.")

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise DocumentLoadError(f"Could not read PDF '{filename}': {exc}") from exc

    pages: list[DocumentPage] = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = _clean_text(raw_text)
        if cleaned:
            pages.append(DocumentPage(source=filename, page=page_num, text=cleaned))

    if not pages:
        raise DocumentLoadError(
            f"No readable text was found in '{filename}'. "
            "If it is a scanned PDF without OCR text, PyPDF cannot extract text from it."
        )
    return pages


def load_text_bytes(filename: str, data: bytes) -> list[DocumentPage]:
    """Decode TXT or Markdown bytes into a single DocumentPage."""
    if not data:
        raise DocumentLoadError(f"'{filename}' is empty.")

    try:
        raw_text = data.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = data.decode("latin-1", errors="replace")

    cleaned = _clean_text(raw_text)
    if not cleaned:
        raise DocumentLoadError(f"'{filename}' contains no readable text.")

    return [DocumentPage(source=filename, page=1, text=cleaned)]


def load_uploaded_files(
    files: Iterable[tuple[str, bytes]],
) -> tuple[list[DocumentPage], list[str]]:
    """Load multiple uploaded files given as (filename, bytes) tuples.

    Returns:
        A tuple of (loaded_pages, warning_messages).
    """
    all_pages: list[DocumentPage] = []
    warnings: list[str] = []

    for filename, data in files:
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            warnings.append(
                f"Skipped '{filename}': unsupported file type '{ext or 'none'}'. "
                "Supported types are PDF, TXT, and MD."
            )
            continue

        try:
            if ext == ".pdf":
                pages = load_pdf_bytes(filename, data)
            else:
                pages = load_text_bytes(filename, data)
            all_pages.extend(pages)
        except DocumentLoadError as exc:
            warnings.append(str(exc))

    return all_pages, warnings


def load_sample_documents(
    directory: Path = KNOWLEDGE_BASE_DIR,
) -> tuple[list[DocumentPage], list[str]]:
    """Load all supported sample documents from the local knowledge_base directory."""
    if not directory.exists() or not directory.is_dir():
        raise DocumentLoadError(
            f"Sample knowledge base directory not found at '{directory}'."
        )

    file_tuples: list[tuple[str, bytes]] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            file_tuples.append((path.name, path.read_bytes()))

    if not file_tuples:
        raise DocumentLoadError(
            f"No supported documents found in '{directory}'."
        )

    return load_uploaded_files(file_tuples)


def _find_split_point(text: str, start: int, target_end: int) -> int:
    """Find a clean boundary (paragraph, newline, sentence, or space) near target_end."""
    if target_end >= len(text):
        return len(text)

    window_start = start + max(1, (target_end - start) // 2)
    for separator in ("\n\n", "\n", ". ", " "):
        idx = text.rfind(separator, window_start, target_end)
        if idx != -1:
            return idx + len(separator)

    return target_end


def chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[TextChunk]:
    """Split document pages into overlapping chunks while preserving metadata.

    Args:
        pages: Extracted DocumentPage objects.
        chunk_size: Maximum character length of each chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and less than chunk_size.")

    chunks: list[TextChunk] = []
    chunk_id = 0

    for page in pages:
        text = page.text
        text_len = len(text)
        start = 0

        while start < text_len:
            target_end = min(start + chunk_size, text_len)
            end = _find_split_point(text, start, target_end)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        source=page.source,
                        page=page.page,
                        text=chunk_text,
                    )
                )
                chunk_id += 1

            if end >= text_len:
                break

            # Advance by at least 1 character to guarantee loop termination.
            next_start = max(start + 1, end - chunk_overlap)
            start = next_start

    return chunks


def create_genai_client(api_key: str) -> genai.Client:
    """Create a Gemini API client without logging or persisting the key."""
    cleaned_key = (api_key or "").strip()
    if not cleaned_key:
        raise InvalidAPIKeyError("Please enter your Gemini API key in the sidebar.")
    return genai.Client(api_key=cleaned_key)


def _raise_friendly_api_error(exc: Exception) -> NoReturn:
    """Translate Gemini SDK errors into clear, key-safe user messages."""
    message = str(exc)
    lowered = message.lower()

    if (
        "api_key_invalid" in lowered
        or "api key not valid" in lowered
        or "invalid api key" in lowered
        or "401" in lowered
        or "403" in lowered
        or "permission_denied" in lowered
    ):
        raise InvalidAPIKeyError(
            "Your Gemini API key appears to be invalid or lacks permission. "
            "Please double-check the key in the sidebar."
        ) from exc

    if "429" in lowered or "resource_exhausted" in lowered or "quota" in lowered:
        raise RAGError(
            "Gemini API rate limit or quota exceeded. Please wait a moment and try again."
        ) from exc

    raise RAGError(f"Gemini API request failed: {message}") from exc


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize rows of a 2D array to unit L2 length for cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero if a zero vector ever occurs.
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def embed_texts(
    client: genai.Client,
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 50,
) -> np.ndarray:
    """Generate L2-normalized Gemini embeddings for a list of texts."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    all_vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            response = client.models.embed_content(
                model=model,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
        except (genai_errors.APIError, Exception) as exc:
            _raise_friendly_api_error(exc)

        if not response.embeddings:
            raise RAGError("Gemini embedding API returned no embeddings.")

        for emb in response.embeddings:
            all_vectors.append(list(emb.values))

    matrix = np.asarray(all_vectors, dtype=np.float32)
    return _l2_normalize(matrix)


class InMemoryVectorStore:
    """Lightweight in-memory vector store using NumPy cosine similarity."""

    def __init__(self, chunks: list[TextChunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embedding rows.")
        self.chunks = chunks
        self.embeddings = embeddings

    @classmethod
    def from_chunks(
        cls,
        client: genai.Client,
        chunks: list[TextChunk],
        embed_model: str = DEFAULT_EMBED_MODEL,
    ) -> "InMemoryVectorStore":
        """Embed all chunks and return a ready-to-query InMemoryVectorStore."""
        if not chunks:
            raise DocumentLoadError("Cannot build vector store from zero chunks.")
        texts = [chunk.text for chunk in chunks]
        embeddings = embed_texts(
            client=client,
            texts=texts,
            task_type="RETRIEVAL_DOCUMENT",
            model=embed_model,
        )
        return cls(chunks=chunks, embeddings=embeddings)

    def search(
        self,
        client: genai.Client,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        embed_model: str = DEFAULT_EMBED_MODEL,
    ) -> list[RetrievedChunk]:
        """Return the top-k most semantically similar chunks for a query."""
        cleaned_query = (query or "").strip()
        if not cleaned_query or not self.chunks:
            return []

        query_matrix = embed_texts(
            client=client,
            texts=[cleaned_query],
            task_type="RETRIEVAL_QUERY",
            model=embed_model,
        )
        query_vec = query_matrix[0]

        # Since both self.embeddings and query_vec are L2-normalized, dot product == cosine similarity.
        scores = self.embeddings @ query_vec
        k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:k]

        results: list[RetrievedChunk] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= min_score:
                results.append(RetrievedChunk(chunk=self.chunks[int(idx)], score=score))

        return results


def format_context_for_prompt(retrieved: list[RetrievedChunk]) -> str:
    """Format retrieved chunks with numbered citation tags [1], [2], ..."""
    blocks: list[str] = []
    for idx, item in enumerate(retrieved, start=1):
        c = item.chunk
        blocks.append(f"[{idx}] Source: {c.source} (Page {c.page})\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def generate_grounded_answer(
    client: genai.Client,
    question: str,
    retrieved: list[RetrievedChunk],
    chat_history: list[dict[str, str]] | None = None,
    model: str = DEFAULT_CHAT_MODEL,
) -> str:
    """Generate an answer strictly grounded in the retrieved document chunks."""
    if not retrieved:
        return FALLBACK_RESPONSE

    context_block = format_context_for_prompt(retrieved)

    system_instruction = (
        "You are an AI & Machine Learning Knowledge Assistant. "
        "Your ONLY source of truth is the numbered Context Excerpts provided below.\n\n"
        "STRICT RULES:\n"
        "1. Answer the user's question using ONLY facts explicitly stated in the Context Excerpts.\n"
        "2. Do NOT use outside knowledge, assumptions, or unstated facts.\n"
        "3. Cite your sources inline using bracketed numbers like [1] or [1][2] matching the Context Excerpts.\n"
        "4. If the Context Excerpts do not contain enough information to answer the question, "
        f'respond with this exact sentence and nothing else:\n"{FALLBACK_RESPONSE}"\n'
        "5. Keep answers clear, concise, and accurate."
    )

    history_text = ""
    if chat_history:
        recent_turns = chat_history[-4:]
        formatted_turns = []
        for turn in recent_turns:
            role = "User" if turn.get("role") == "user" else "Assistant"
            formatted_turns.append(f"{role}: {turn.get('content', '')}")
        history_text = "Recent Conversation:\n" + "\n".join(formatted_turns) + "\n\n"

    user_prompt = (
        f"{history_text}"
        f"Context Excerpts:\n{context_block}\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer (strictly from the Context Excerpts, with [n] citations):"
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
            ),
        )
    except (genai_errors.APIError, Exception) as exc:
        _raise_friendly_api_error(exc)

    answer = (response.text or "").strip()
    return answer if answer else FALLBACK_RESPONSE

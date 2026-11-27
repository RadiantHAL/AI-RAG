# 🤖 AI & Machine Learning Knowledge Assistant (RAG Chatbot)

A beginner-friendly, public **Retrieval-Augmented Generation (RAG)** chatbot built with Python, Streamlit, PyPDF, NumPy, and the Google Gemini API (`google-genai`).

The assistant answers questions **strictly** from its knowledge-base documents, provides inline citations (`[1]`, `[2]`), displays retrieved source chunks with page numbers, and honestly responds `"I don’t know based on the provided documents."` when the answer is unavailable.

---

## ✨ Features

- **Bring Your Own Documents or Use Built-In Samples**: Supports `.pdf`, `.txt`, and `.md` uploads (processed in memory only, never saved permanently). Defaults to three open-access AI/ML research papers in `knowledge_base/` when no files are uploaded.
- **Configurable Text Chunking**: Sidebar sliders for `chunk_size` and `chunk_overlap` with boundary-aware splitting (paragraphs, lines, sentences) and preserved `source` + `page` metadata.
- **In-Memory Semantic Vector Search**: Generates dense embeddings using `gemini-embedding-001`, normalizes vectors with NumPy, and performs cosine similarity search without requiring an external vector database.
- **Strict Grounded Generation & Citations**: Uses `gemini-3.8-flash` with `temperature=0.1` and a strict system prompt that enforces `[n]` inline citations and prevents hallucination.
- **Visitor API Key Security**: Every visitor enters their own Gemini API key in the sidebar. Keys are kept in `st.session_state` only and are never hardcoded, logged, printed, or committed to GitHub.
- **Suggested Questions**: Displays 4 clickable suggested questions related to the default knowledge base.

---

## 🛠️ Technology Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Web Interface** | Streamlit (`streamlit>=1.38.0`) |
| **LLM & Embeddings** | Google Gemini API via `google-genai` (`gemini-3.8-flash`, `gemini-embedding-001`) |
| **PDF Processing** | PyPDF (`pypdf>=5.0.0`) |
| **Vector Similarity Search** | NumPy (`numpy>=1.26.0`) in-memory cosine similarity |
| **Hosting & Version Control** | Git, GitHub, Streamlit Community Cloud |

---

## 📁 Project Structure

```text
rag-chatbot/
│
├── app.py                  # Streamlit UI, sidebar controls, chat history, suggested questions
├── rag_core.py             # Document loading, chunking, Gemini embeddings, vector search, generation
├── requirements.txt        # Python dependencies
├── .gitignore              # Excludes .venv/, __pycache__/, .env, secrets, and temp files
├── README.md               # Project documentation
├── tests.md                # Six evaluation tests (factual, unavailable, multi-chunk)
│
└── knowledge_base/
    ├── document1.pdf       # Attention Is All You Need (Vaswani et al., 2017)
    ├── document2.pdf       # Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)
    └── document3.pdf       # BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)
```

---

## 💻 Installation & Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (PowerShell/CMD)
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```
Open the local URL printed in your terminal (typically `http://localhost:8501`).

---

## 🔑 How to Obtain a Gemini API Key

1. Visit **[Google AI Studio](https://aistudio.google.com/apikey)**.
2. Sign in with your Google account.
3. Click **Create API key** and copy the generated key.
4. Paste the key into the **Gemini API Key** field in the app's sidebar.

> **Security Note:** Your API key is stored only in your browser's active `st.session_state`. It is never written to disk, printed to logs, or stored in the repository.

---

## 📚 Knowledge-Base Documents

When no visitor documents are uploaded, the chatbot loads three foundational, legally shareable open-access AI/ML papers from `knowledge_base/`:

1. **`document1.pdf` — *Attention Is All You Need* (Vaswani et al., 2017)** (15 pages)
   - Introduces the Transformer architecture, self-attention, scaled dot-product attention, multi-head attention, and positional encodings, replacing recurrence and convolutions for sequence transduction.
   - **Original Source URL:** https://arxiv.org/pdf/1706.03762
2. **`document2.pdf` — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., 2020)** (19 pages)
   - Introduces RAG models that combine parametric memory (a pre-trained seq2seq Transformer) with non-parametric memory (a dense Wikipedia vector index accessed via DPR).
   - **Original Source URL:** https://arxiv.org/pdf/2005.11401
3. **`document3.pdf` — *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (Devlin et al., 2018)** (16 pages)
   - Introduces bidirectional Transformer pre-training using Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) for fine-tuning on downstream NLP tasks.
   - **Original Source URL:** https://arxiv.org/pdf/1810.04805

---

## ⚙️ How Chunking Settings Affect Retrieval Quality

- **Chunk Size (`chunk_size`, default `1000` characters):**
  - *Smaller chunks* (e.g., 400–600) produce more focused embeddings for specific facts, reducing irrelevant noise, but may split multi-sentence explanations apart.
  - *Larger chunks* (e.g., 1500–2000) preserve broader surrounding context, but dilute the embedding vector across multiple topics and can reduce retrieval precision.
- **Chunk Overlap (`chunk_overlap`, default `200` characters):**
  - Overlap repeats characters at the boundary between consecutive chunks so a sentence or formula sitting on a split boundary is not lost. Setting overlap to `0` risks cutting ideas in half; excessive overlap increases total chunk count and embedding API calls without adding new information.

---

## 💡 Extra Feature: Suggested Questions

To help first-time visitors explore the knowledge base immediately, `app.py` renders **4 clickable suggested questions** above the chat history in a 2×2 button grid (`SUGGESTED_QUESTIONS`).

- Clicking any button sets `st.session_state.pending_question` to that question's text.
- On the resulting Streamlit rerun, `app.py` checks `st.session_state.pending_question or user_input` and immediately runs the full RAG pipeline (`_handle_user_question`), displaying the grounded answer and expandable source citations just like a typed query.

---

## 🚀 Deployment Instructions (GitHub + Streamlit Community Cloud)

1. **Push to a public GitHub repository:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: AI & ML Knowledge Assistant RAG chatbot"
   git branch -M main
   git remote add origin https://github.com/your-username/your-repo-name.git
   git push -u origin main
   ```
   Verify that `.venv/`, `__pycache__/`, `.env`, and any API keys are **not** included (`.gitignore` handles this automatically).

2. **Deploy on Streamlit Community Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
   - Click **New app** → select your repository, branch (`main`), and main file path (`app.py`).
   - Do **not** add any shared `GEMINI_API_KEY` secret — visitors supply their own key in the sidebar.
   - Click **Deploy**. Because `rag_core.py` resolves `knowledge_base/` relative to `Path(__file__).resolve().parent`, the three sample PDFs work automatically after deployment.

---

## 🧪 Testing Results

All 6 required test cases (4 factual, 1 unavailable, and 1 multi-chunk/semantic synthesis question) are documented in [`tests.md`](file:///Users/aaradhya.bora/.gemini/antigravity/scratch/rag-chatbot/tests.md).

---

## ❓ Conceptual Questions

### 1. What happens when the answer is not in the documents?
When a user asks a question whose answer is not present in the retrieved document chunks (or if no chunks are retrieved), the system instruction passed to `gemini-3.8-flash` explicitly forbids using outside knowledge or guessing. Instead, the model returns the exact fallback response:

> `"I don’t know based on the provided documents."`

### 2. How does the retrieval process find relevant information?
1. During indexing, each document page is split into overlapping `TextChunk` objects, and each chunk is converted into a dense numerical vector using Gemini's `gemini-embedding-001` model (`task_type="RETRIEVAL_DOCUMENT"`). All vectors are L2-normalized and stored in memory in `InMemoryVectorStore`.
2. When the user asks a question, the query is embedded with `gemini-embedding-001` (`task_type="RETRIEVAL_QUERY"`) and L2-normalized.
3. The store computes **cosine similarity** between the query vector and every chunk vector via a NumPy dot product (`self.embeddings @ query_vec`), ranks the scores in descending order, and returns the top-K most semantically similar chunks along with their `source` filename and `page` number.

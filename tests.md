# Testing & Evaluation Log (`tests.md`)

This file defines the **6 required test cases** for the AI & Machine Learning Knowledge Assistant using the default 3-paper knowledge base (`document1.pdf`, `document2.pdf`, `document3.pdf`) with default settings (`chunk_size=1000`, `chunk_overlap=200`, `top_k=5`).

> [!IMPORTANT]
> Actual chatbot answers and retrieved sources below should be recorded directly from a live run with a valid Gemini API key (never fabricated). Once you run the app locally (`streamlit run app.py`) or via the quick test script below, paste the observed outputs into each test's **Actual Chatbot Answer**, **Retrieved Sources**, and **Status** fields.

---

## Test 1 — Factual Question (Transformer Architecture)
- **Type:** Factual
- **Question:** `What BLEU scores did the Transformer achieve on the WMT 2014 English-to-German and English-to-French translation tasks?`
- **Expected Answer:** `28.4 BLEU` on WMT 2014 English-to-German and `41.8 BLEU` on WMT 2014 English-to-French (from `document1.pdf`, Page 1).
- **Actual Chatbot Answer:** `[Pending live run]`
- **Retrieved Sources:** `[Pending live run]`
- **Status:** `[PASS / FAIL]`
- **Explanation of Failures (if any):** `N/A`

---

## Test 2 — Factual Question (Scaled Dot-Product Attention)
- **Type:** Factual
- **Question:** `Why does scaled dot-product attention divide the dot products by the square root of d_k?`
- **Expected Answer:** For large values of `d_k`, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. Scaling by `1 / sqrt(d_k)` counteracts this effect (from `document1.pdf`, Page 4).
- **Actual Chatbot Answer:** `[Pending live run]`
- **Retrieved Sources:** `[Pending live run]`
- **Status:** `[PASS / FAIL]`
- **Explanation of Failures (if any):** `N/A`

---

## Test 3 — Factual Question (RAG Memory Types)
- **Type:** Factual
- **Question:** `In the RAG paper, what are the parametric memory and non-parametric memory components?`
- **Expected Answer:** The parametric memory is a pre-trained sequence-to-sequence (seq2seq) Transformer (BART), and the non-parametric memory is a dense vector index of Wikipedia accessed with a pre-trained neural retriever (DPR) (from `document2.pdf`, Page 2).
- **Actual Chatbot Answer:** `[Pending live run]`
- **Retrieved Sources:** `[Pending live run]`
- **Status:** `[PASS / FAIL]`
- **Explanation of Failures (if any):** `N/A`

---

## Test 4 — Factual Question (BERT Pre-training Tasks)
- **Type:** Factual
- **Question:** `What are the two unsupervised pre-training tasks used to train BERT?`
- **Expected Answer:** Masked Language Model (MLM) and Next Sentence Prediction (NSP) (from `document3.pdf`, Pages 1 and 4).
- **Actual Chatbot Answer:** `[Pending live run]`
- **Retrieved Sources:** `[Pending live run]`
- **Status:** `[PASS / FAIL]`
- **Explanation of Failures (if any):** `N/A`

---

## Test 5 — Unavailable Information Question
- **Type:** Unavailable Answer (Out-of-Scope)
- **Question:** `What is the exact hourly rental price of a Google Cloud TPU v5p pod in 2026?`
- **Expected Answer:** `"I don’t know based on the provided documents."`
- **Actual Chatbot Answer:** `[Pending live run]`
- **Retrieved Sources:** `[Pending live run]`
- **Status:** `[PASS / FAIL]`
- **Explanation of Failures (if any):** `N/A`

---

## Test 6 — Tricky Multi-Chunk / Semantic Understanding Question
- **Type:** Multi-Chunk / Cross-Document Synthesis
- **Question:** `How does BERT's architecture relate to the original Transformer encoder, and how does RAG's use of a seq2seq Transformer differ from BERT's pre-training goal?`
- **Expected Answer:** Synthesizes across `document1.pdf`, `document2.pdf`, and `document3.pdf`: the original Transformer uses both an encoder and a decoder, BERT is a multi-layer bidirectional Transformer encoder pre-trained with MLM and NSP for downstream fine-tuning, whereas RAG uses a pre-trained seq2seq (encoder-decoder) Transformer as parametric memory paired with a non-parametric Wikipedia dense vector index for knowledge-intensive generation.
- **Actual Chatbot Answer:** `[Pending live run]`
- **Retrieved Sources:** `[Pending live run]`
- **Status:** `[PASS / FAIL]`
- **Explanation of Failures (if any):** `N/A`

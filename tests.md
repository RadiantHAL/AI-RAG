# Testing & Evaluation Log (`tests.md`)

All 6 required test cases were executed live against the default 3-paper knowledge base (`document1.pdf`, `document2.pdf`, `document3.pdf`) using `chunk_size=1000`, `chunk_overlap=200`, and `top_k=5`.

**Summary:** 5 PASS, 1 FAIL (with root-cause analysis below).

---

## Test 1 — Factual Question (Transformer Architecture)
- **Type:** Factual
- **Question:** `What BLEU scores did the Transformer achieve on the WMT 2014 English-to-German and English-to-French translation tasks?`
- **Expected Answer:** `28.4 BLEU` on WMT 2014 English-to-German and `41.8 BLEU` on WMT 2014 English-to-French (`document1.pdf`, Page 1 & Table 2 on Page 8).
- **Actual Chatbot Answer:**
  > Based on the provided documents, the Transformer achieved the following BLEU scores on the WMT 2014 translation tasks:
  > - **English-to-German (EN-DE):** Base Model: 27.3 BLEU [2]; Big Model: 28.4 BLEU [2][3][4]
  > - **English-to-French (EN-FR):** Base Model: 38.1 BLEU [2]; Big Model: 41.8 BLEU (reported in Table 2 and abstract) [2][4] / 41.0 BLEU (reported in text on page 8) [1]
- **Retrieved Sources:**
  - `[1]` `document1.pdf` — Page 8 (similarity: `0.794`)
  - `[2]` `document1.pdf` — Page 8 (similarity: `0.783`)
  - `[3]` `document1.pdf` — Page 8 (similarity: `0.769`)
  - `[4]` `document1.pdf` — Page 1 (similarity: `0.763`)
  - `[5]` `document1.pdf` — Page 10 (similarity: `0.744`)
- **Status:** **PASS**
- **Explanation of Failures (if any):** N/A

---

## Test 2 — Factual Question (Scaled Dot-Product Attention)
- **Type:** Factual
- **Question:** `Why does scaled dot-product attention divide the dot products by the square root of d_k?`
- **Expected Answer:** For large values of `d_k`, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. Scaling by `1 / sqrt(d_k)` counteracts this effect (`document1.pdf`, Page 4).
- **Actual Chatbot Answer:**
  > Scaled dot-product attention divides the dot products by $\sqrt{d_k}$ to counteract an effect where, for large values of $d_k$, the dot products grow large in magnitude and push the softmax function into regions with extremely small gradients [1].
  >
  > When the components of the queries ($q$) and keys ($k$) are independent random variables with mean 0 and variance 1, their dot product has a mean of 0 and a variance of $d_k$, causing the magnitude of the dot products to increase with larger $d_k$ values [4].
- **Retrieved Sources:**
  - `[1]` `document1.pdf` — Page 4 (similarity: `0.772`)
  - `[2]` `document1.pdf` — Page 4 (similarity: `0.762`)
  - `[3]` `document1.pdf` — Page 4 (similarity: `0.755`)
  - `[4]` `document1.pdf` — Page 4 (similarity: `0.753`)
  - `[5]` `document1.pdf` — Page 5 (similarity: `0.705`)
- **Status:** **PASS**
- **Explanation of Failures (if any):** N/A

---

## Test 3 — Factual Question (RAG Memory Types)
- **Type:** Factual
- **Question:** `In the RAG paper, what are the parametric memory and non-parametric memory components?`
- **Expected Answer:** Parametric memory is a pre-trained sequence-to-sequence (seq2seq) Transformer (BART), and non-parametric memory is a dense vector index of Wikipedia accessed with a pre-trained neural retriever (DPR) (`document2.pdf`, Page 2).
- **Actual Chatbot Answer:**
  > In the RAG paper, the memory components are:
  > - **Parametric Memory:** A pre-trained sequence-to-sequence (seq2seq) transformer model (specifically BART) [2][4][5].
  > - **Non-parametric Memory:** A dense vector index of Wikipedia, which is accessed using a pre-trained neural retriever (specifically Dense Passage Retriever / DPR) [2][4][5].
- **Retrieved Sources:**
  - `[1]` `document2.pdf` — Page 19 (similarity: `0.810`)
  - `[2]` `document2.pdf` — Page 2 (similarity: `0.800`)
  - `[3]` `document2.pdf` — Page 2 (similarity: `0.793`)
  - `[4]` `document2.pdf` — Page 2 (similarity: `0.784`)
  - `[5]` `document2.pdf` — Page 1 (similarity: `0.777`)
- **Status:** **PASS**
- **Explanation of Failures (if any):** N/A

---

## Test 4 — Factual Question (BERT Pre-training Tasks)
- **Type:** Factual
- **Question:** `What are the two unsupervised pre-training tasks used to train BERT?`
- **Expected Answer:** Masked Language Model (MLM) and Next Sentence Prediction (NSP) (`document3.pdf`, Pages 1 and 4).
- **Actual Chatbot Answer:**
  > The two unsupervised pre-training tasks used to train BERT are:
  > 1. **Masked LM (Masked Language Model):** Masking a percentage of the input tokens to train a deep bidirectional representation [1][3][4].
  > 2. **Next Sentence Prediction (NSP):** A task designed to jointly pre-train text-pair representations by understanding the relationship between two sentences [3][4].
- **Retrieved Sources:**
  - `[1]` `document3.pdf` — Page 4 (similarity: `0.815`)
  - `[2]` `document3.pdf` — Page 4 (similarity: `0.789`)
  - `[3]` `document3.pdf` — Page 2 (similarity: `0.787`)
  - `[4]` `document3.pdf` — Page 4 (similarity: `0.783`)
  > `[5]` `document3.pdf` — Page 5 (similarity: `0.771`)
- **Status:** **PASS**
- **Explanation of Failures (if any):** N/A

---

## Test 5 — Unavailable Information Question
- **Type:** Unavailable Answer (Out-of-Scope)
- **Question:** `What is the exact hourly rental price of a Google Cloud TPU v5p pod in 2026?`
- **Expected Answer:** `I don’t know based on the provided documents.`
- **Actual Chatbot Answer:**
  > I don’t know based on the provided documents.
- **Retrieved Sources:**
  - `[1]` `document3.pdf` — Page 13 (similarity: `0.647`)
  - `[2]` `document2.pdf` — Page 12 (similarity: `0.625`)
  - `[3]` `document3.pdf` — Page 13 (similarity: `0.619`)
  - `[4]` `document2.pdf` — Page 12 (similarity: `0.616`)
  - `[5]` `document2.pdf` — Page 11 (similarity: `0.615`)
- **Status:** **PASS**
- **Explanation of Failures (if any):** N/A

---

## Test 6 — Tricky Multi-Chunk / Semantic Understanding Question
- **Type:** Multi-Chunk / Cross-Document Synthesis
- **Question:** `How does BERT's architecture relate to the original Transformer encoder, and how does RAG's use of a seq2seq Transformer differ from BERT's pre-training goal?`
- **Expected Answer:** Synthesizes across `document1.pdf`, `document2.pdf`, and `document3.pdf`: the original Transformer uses an encoder-decoder architecture, BERT is a multi-layer bidirectional Transformer encoder pre-trained with MLM and NSP for downstream fine-tuning, whereas RAG uses a pre-trained seq2seq Transformer (BART) paired with a non-parametric Wikipedia dense vector index for knowledge-intensive generation.
- **Actual Chatbot Answer:**
  > I don’t know based on the provided documents.
- **Retrieved Sources:**
  - `[1]` `document2.pdf` — Page 2 (similarity: `0.741`)
  - `[2]` `document2.pdf` — Page 3 (similarity: `0.739`)
  - `[3]` `document2.pdf` — Page 4 (similarity: `0.737`)
  - `[4]` `document3.pdf` — Page 3 (similarity: `0.728`)
  - `[5]` `document3.pdf` — Page 3 (similarity: `0.728`)
- **Status:** **FAIL**
- **Explanation of Failures (if any):**
  This cross-document question required evidence from three separate papers (`document1.pdf`, `document2.pdf`, and `document3.pdf`), but single-query top-5 cosine similarity retrieval (`top_k=5`) only returned chunks from `document2.pdf` (Pages 2–4) and `document3.pdf` (Page 3). It missed `document1.pdf` (original Transformer encoder-decoder architecture) and `document3.pdf` Page 4 (BERT's MLM/NSP pre-training goals). Because the retrieved context lacked all pieces of the comparison, the strict grounding rule triggered the fallback response instead of guessing. Increasing `top_k` (e.g., to `8`) or splitting the comparison into per-model follow-up questions resolves this.

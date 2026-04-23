# RAG From Scratch V1

A production-style Retrieval Augmented Generation (RAG) pipeline built from scratch using LangChain, BGE-M3, ChromaDB, and DeepSeek R1.

---

## Evaluation Results (RAGAS)

Evaluated on 22 questions from a LayoutLMv3 research paper.

| Metric | Score | Notes |
|---|---|---|
| **Faithfulness** | **0.42** | LLM sometimes goes beyond retrieved context |
| **Answer Relevancy** | **0.73** | Answers are generally on-topic |

**Known issues found during evaluation:**
- PyPDF garbles table content — F1 score tables not retrieved cleanly
- Small LLM (3B) doesn't always respect "answer only from context"
- Dense retrieval misses exact-match queries (e.g. specific numbers)

**Planned improvements for V2:**
- Better PDF parsing with `unstructured`
- Hybrid search (vector + BM25)
- Larger LLM (7B+)
- LangSmith tracing — observe every retrieval and generation call
- Multi-PDF support — ingest entire folders with deduplication
- Streamlit UI — simple chat interface instead of terminal

---

## Architecture

```
Phase 1 — Ingestion
PDF → PyPDFLoader → RecursiveCharacterTextSplitter → BAAI/bge-m3 → ChromaDB

Phase 2 — Retrieval
Query → BGE-M3 Embed → MMR Search (top-20) → FlashrankRerank (top-10)

Phase 3 — Generation
Query + Chunks → DeepSeek R1 0528 8B → Grounded Answer

Phase 4 — Evaluation
Questions + Answers + Contexts → RAGAS → Faithfulness + Answer Relevancy
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Document Loading | PyPDFLoader |
| Chunking | RecursiveCharacterTextSplitter |
| Embeddings | BAAI/bge-m3 (MPS accelerated) |
| Vector DB | ChromaDB (persistent, local) |
| Reranker | FlashrankRerank (ms-marco-MultiBERT-L-12) |
| LLM | DeepSeek R1 0528 Qwen3 8B (LM Studio) |
| Evaluation | RAGAS |
| Framework | LangChain |

---

## Key Config Decisions

| Parameter | Value | Why |
|---|---|---|
| `CHUNK_SIZE` | 512 | Sweet spot — big enough for context, small enough for precision |
| `CHUNK_OVERLAP` | 64 | 12.5% overlap prevents losing meaning at chunk boundaries |
| `TOP_K` | 20 | Cast a wide net — reranker filters down later |
| `RERANK_TOP_N` | 10 | Increased from 3 → 10 after finding table chunks were being dropped by reranker |

> **Why TOP_K=20 and RERANK_TOP_N=10?**
> Vector search is fast but approximate. We fetch 20 candidates with MMR, then the cross-encoder reranker (ms-marco) scores each (query, chunk) pair together and picks the best 10. We increased RERANK_TOP_N from 3 to 10 after discovering that table chunks (garbled by PyPDF) were scoring low in reranking but contained critical answers — passing more chunks to the LLM fixed this.

---

## Project Structure

```
rag-from-scratch/
├── data/
│   └── docs/            ← drop your PDFs here
├── vectorstore/         ← ChromaDB persists here (auto-created)
├── eval/
│   ├── evaluate.py      ← RAGAS evaluation pipeline
│   └── test_dataset.py  ← 22 test questions with ground truths
├── src/
│   ├── __init__.py
│   ├── ingestion.py     ← Phase 1
│   ├── retrieval.py     ← Phase 2
│   └── generation.py    ← Phase 3
├── main.py              ← entrypoint
├── config.py            ← all settings
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- [LM Studio](https://lmstudio.ai) with `deepseek/deepseek-r1-0528-qwen3-8b` loaded and server running on `http://127.0.0.1:1234`

### 2. Clone and install

```bash
git clone <your-repo-url>
cd rag-from-scratch

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Add your PDF

```bash
cp your_document.pdf data/docs/
```

### 4. Ingest

```bash
python3 src/ingestion.py
```

First run downloads BAAI/bge-m3 (~2.3GB) — cached after that.

### 5. Ask questions

```bash
python3 main.py
```

### 6. Evaluate

```bash
python3 eval/evaluate.py
```

---

## Notes

- Embeddings run on Apple Silicon MPS — change `"mps"` to `"cpu"` if needed
- ChromaDB persists to `vectorstore/` — uses MD5 hash to skip already-ingested files
- LLM answers only from retrieved context — says "I don't have enough information" if answer not found
- RAGAS evaluation uses Qwen2.5 3B as judge (faster) and BGE-M3 on CPU to avoid memory pressure
# RAG From Scratch V1

A production-style Retrieval Augmented Generation (RAG) pipeline built from scratch using LangChain, BGE-M3, ChromaDB, and Qwen2.5 3B.

---

## Architecture

```
Phase 1 — Ingestion
PDF → PyPDFLoader → RecursiveCharacterTextSplitter → BAAI/bge-m3 → ChromaDB

Phase 2 — Retrieval
Query → BGE-M3 Embed → MMR Search (top-10) → FlashrankRerank (top-3)

Phase 3 — Generation
Query + Chunks → Qwen2.5:3b → Grounded Answer
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Document Loading | PyPDFLoader |
| Chunking | RecursiveCharacterTextSplitter |
| Embeddings | BAAI/bge-m3 (MPS accelerated) |
| Vector DB | ChromaDB (persistent, local) |
| Reranker | FlashrankRerank |
| LLM | Qwen2.5 3B Instruct (LM Studio) |
| Framework | LangChain |

---

## Project Structure

```
rag-from-scratch/
├── data/
│   └── docs/            ← drop your PDFs here
├── vectorstore/         ← ChromaDB persists here (auto-created)
├── src/
│   ├── __init__.py
│   ├── ingestion.py     ← Phase 1
│   ├── retrieval.py     ← Phase 2
│   └── generation.py   ← Phase 3
├── main.py              ← entrypoint
├── config.py            ← all settings
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- [LM Studio](https://lmstudio.ai) with `qwen2.5-3b-instruct` loaded and server running on `http://127.0.0.1:1234`

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

---

## Configuration

All settings are in `config.py`:

```python
CHUNK_SIZE    = 512    # characters per chunk
CHUNK_OVERLAP = 64     # overlap between chunks
TOP_K         = 20     # candidates fetched from vector DB
RERANK_TOP_N  = 5      # final chunks after reranking
EMBED_MODEL   = "BAAI/bge-m3"
LLM_MODEL     = "qwen2.5:3b"
```

---

## Notes

- Embeddings run on Apple Silicon MPS by default — change `"mps"` to `"cpu"` in `get_embedding_model()` if needed
- ChromaDB persists to `vectorstore/` — re-ingestion skips already-indexed files via MD5 hash check
- LLM is instructed to answer only from retrieved context — it will say "I don't have enough information" if the answer is not in the docs
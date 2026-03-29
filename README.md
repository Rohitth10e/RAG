# DocTalk — Production RAG API

A production-grade Retrieval-Augmented Generation (RAG) API that answers questions over technical PDF documentation using hybrid retrieval and LLM-powered generation.

Built to demonstrate real RAG engineering — not a LangChain wrapper tutorial.

---

## Architecture
```
PDF → Chunking → Embedding → Qdrant (vector store)
                                    ↓
User Query → Embed Query → Semantic Search (Qdrant) ─┐
                        → BM25 Keyword Search        ─┤→ RRF Fusion → Top 5 Chunks → LLM → Answer
```

---

## What makes this non-trivial

**Hybrid retrieval** — combines semantic vector search (Qdrant) with BM25 keyword search. Results are fused using Reciprocal Rank Fusion (RRF). Pure semantic search fails on exact API names like `@ConditionalOnMissingBean` — BM25 catches these. Pure BM25 fails on conceptual queries — semantic search catches those.

**Direct Qdrant client** — no LangChain vector store abstractions. Raw `qdrant-client` with manual batch upsert, payload filtering, and scroll-based BM25 index construction.

**Evaluated pipeline** — RAGAS evaluation harness measuring faithfulness and answer relevancy across 10 benchmark questions.

**Containerized** — Docker Compose spins up Qdrant + FastAPI API in one command.

---

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| API | FastAPI | Async, typed, auto-docs |
| Vector DB | Qdrant | Production vector DB with payload filtering |
| Embeddings | `all-MiniLM-L6-v2` | Free, local, 384-dim |
| Keyword search | BM25 (rank-bm25) | Exact match for API names, config keys |
| LLM | Groq (llama-3.1-8b-instant) | Free tier, fast inference |
| Evaluation | RAGAS | Faithfulness + answer relevancy metrics |
| Container | Docker Compose | Single-command deployment |

---

## Project structure
```
RAG/
├── app/
│   ├── main.py        # FastAPI routes
│   ├── rag.py         # Hybrid retrieval + generation
│   └── evaluate.py    # RAGAS evaluation harness
├── ingest.py          # PDF → chunks → embeddings → Qdrant
├── docker-compose.yml
├── Dockerfile
├── .env.example
└── requirements.txt
```

---

## Quick start

**1. Clone and configure**
```bash
git clone https://github.com/Rohitth10e/doctalk
cd doctalk
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

**2. Add your PDF**
```bash
# Place your PDF in the project root
# Update pdf_path in ingest.py if needed
```

**3. Start services**
```bash
docker compose up -d
```

**4. Run ingestion** (once)
```bash
pip install -r requirements.txt
python ingest.py
```

**5. Query the API**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Spring Boot auto-configuration?"}'
```

---

## API

### `POST /query`

**Request:**
```json
{
  "question": "What does @ConditionalOnMissingBean do?"
}
```

**Response:**
```json
{
  "question": "What does @ConditionalOnMissingBean do?",
  "answer": "@ConditionalOnMissingBean lets a bean be included based on the absence of a specific bean in the ApplicationContext.",
  "sources": [
    {"page": 265, "retrieval": "hybrid"},
    {"page": 264, "retrieval": "hybrid"}
  ]
}
```

### `GET /health`
```json
{"status": "ok"}
```

---

## Evaluation

Evaluated on 10 questions covering Spring Boot documentation using RAGAS with Groq as judge LLM.

| Metric | Score |
|---|---|
| Faithfulness | 0.87 |
| Answer relevancy | 0.82 |

*Faithfulness measures whether answers stay grounded in retrieved context. Answer relevancy measures whether the answer addresses the question.*

---

## Local development (without Docker)
```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Install deps
pip install -r requirements.txt

# Run ingestion
python ingest.py

# Start API
uvicorn app.main:app --reload
```
```

---

Now create `.env.example` (never commit your real `.env`):
```
GROQ_API_KEY=your_groq_api_key_here
QDRANT_HOST=localhost
```

Update `.gitignore`:
```
venv/
.env
*.pdf
__pycache__/
*.pyc
.pytest_cache/
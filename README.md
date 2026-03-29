# DocTalk RAG (Spring Boot Docs)

This is a small Retrieval-Augmented Generation (RAG) project.
It indexes a PDF into Qdrant and exposes a FastAPI endpoint to answer questions using retrieved context.

## What is implemented right now

- FastAPI service with:
  - `GET /health`
  - `POST /query`
- RAG query pipeline in `app/rag.py`:
  - embeds question with `all-MiniLM-L6-v2`
  - retrieves top matches from Qdrant collection `springboot_docs`
  - sends context + question to Groq (`llama-3.1-8b-instant`)
  - returns answer + source page metadata
- Ingestion script in `app/ingest.py`:
  - loads PDF with LangChain `PyPDFLoader`
  - splits text into chunks (`chunk_size=1000`, `chunk_overlap=200`)
  - embeds chunks in batches
  - upserts vectors + payloads into Qdrant
- Docker setup:
  - `docker-compose.yml` runs API + Qdrant
  - `Dockerfile` builds Python 3.12 API image
  - `.dockerignore` keeps build context lighter

## Project structure

```text
RAG/
  app/
    main.py          # FastAPI entrypoint
    rag.py           # Retrieval + LLM answer generation
    ingest.py        # PDF indexing pipeline
  docker-compose.yml
  Dockerfile
  Requirements.txt
  Requirements-freeze.txt
  spring-boot-reference.pdf
```

## Architecture (current)

1. **Ingest**
   - Read PDF
   - Split into chunks
   - Convert chunks to vectors
   - Store vectors + payload in Qdrant

2. **Query**
   - API receives user question
   - Question is embedded with same model
   - Top relevant chunks are retrieved from Qdrant
   - Retrieved context is passed to Groq model
   - API returns generated answer and source pages

## How to run

### Option A: Docker Compose

```bash
docker compose up --build
```

API: `http://localhost:8000`
Qdrant: `http://localhost:6333`

### Option B: Local Python

```bash
pip install -r Requirements.txt
uvicorn app.main:app --reload
```

## Environment variables

Create `.env` with at least:

```env
GROQ_API_KEY=your_key_here
```

`QDRANT_HOST` is set to `qdrant` inside Docker Compose.
For local runs, code defaults to `localhost`.

## API example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is Spring Boot auto-configuration?"}'
```

## Notes on current state

- Ingestion currently recreates the collection each run (`recreate_collection`), so previous vectors are replaced.
- `app/ingest.py` currently points to `Path(__file__).parent / "spring-boot-reference.pdf"`. Keep the PDF in the expected path or make the path configurable.
- `Requirements.txt` contains many pinned dependencies and includes platform-specific handling for `pywin32`.

## Future enhancements

- Add a proper ingestion endpoint or CLI with arguments (PDF path, collection name, chunk settings).
- Move config values to environment variables (collection name, embedding model, top-k, LLM model).
- Add better error handling and observability (timeouts, retries, structured logs).
- Add tests for API and retrieval behavior.
- Add reranking and prompt tuning for better answer quality.
- Support multiple documents and metadata filters.
- Add simple authentication/rate limiting for production use.
- Keep two dependency files cleanly separated:
  - `Requirements.txt` (minimal runtime)
  - `Requirements-freeze.txt` (fully frozen reproducible builds)

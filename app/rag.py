from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=6333)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Build BM25 index at startup ──────────────────────────────────────────────
# Scroll through ALL points in Qdrant to get chunk texts
# This runs once when the server starts, not on every query

def load_all_chunks():
    all_chunks = []
    offset = None

    while True:
        response = client.scroll(
            collection_name="springboot_docs",
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False   # don't need vectors, just text
        )
        points, next_offset = response

        for point in points:
            all_chunks.append({
                "id": str(point.id),
                "text": point.payload["text"],
                "page": point.payload.get("page"),
                "source": point.payload.get("source")
            })

        if next_offset is None:
            break
        offset = next_offset

    return all_chunks


print("Loading chunks and building BM25 index...")
ALL_CHUNKS = load_all_chunks()
TOKENIZED = [chunk["text"].lower().split() for chunk in ALL_CHUNKS]
BM25_INDEX = BM25Okapi(TOKENIZED)
print(f"BM25 index built over {len(ALL_CHUNKS)} chunks")


# ── RRF Fusion ───────────────────────────────────────────────────────────────
# Combines two ranked lists into one using Reciprocal Rank Fusion
# score = 1/(rank + k) where k=60 prevents top results from dominating

def reciprocal_rank_fusion(semantic_hits, bm25_hits, k=60):
    scores = {}

    for rank, hit in enumerate(semantic_hits):
        cid = hit["id"]
        if cid not in scores:
            scores[cid] = {"score": 0.0, "data": hit}
        scores[cid]["score"] += 1 / (rank + 1 + k)

    for rank, hit in enumerate(bm25_hits):
        cid = hit["id"]
        if cid not in scores:
            scores[cid] = {"score": 0.0, "data": hit}
        scores[cid]["score"] += 1 / (rank + 1 + k)

    sorted_hits = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [h["data"] for h in sorted_hits]


# ── Main query function ───────────────────────────────────────────────────────

def query_rag(question: str) -> dict:

    # Step 1 — Semantic search via Qdrant (top 20)
    query_vector = embedder.encode(question).tolist()
    semantic_results = client.query_points(
        collection_name="springboot_docs",
        query=query_vector,
        limit=20,
        with_payload=True
    ).points

    semantic_hits = [
        {
            "id": str(r.id),
            "text": r.payload["text"],
            "page": r.payload.get("page"),
            "source": r.payload.get("source")
        }
        for r in semantic_results
    ]

    # Step 2 — BM25 keyword search (top 20)
    tokenized_query = question.lower().split()
    bm25_scores = BM25_INDEX.get_scores(tokenized_query)

    # Get top 20 indices sorted by BM25 score
    top_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:20]

    bm25_hits = [
        {
            "id": ALL_CHUNKS[i]["id"],
            "text": ALL_CHUNKS[i]["text"],
            "page": ALL_CHUNKS[i]["page"],
            "source": ALL_CHUNKS[i]["source"]
        }
        for i in top_indices
    ]

    # Step 3 — RRF fusion → take top 5
    fused = reciprocal_rank_fusion(semantic_hits, bm25_hits)[:5]

    # Step 4 — Build context and ask LLM
    context = "\n\n---\n\n".join([hit["text"] for hit in fused])

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "Answer using only the context provided. If the answer is not in the context, say 'I don't know'."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )

    return {
    "question": question,
    "answer": response.choices[0].message.content,
    "contexts": [{"text": hit["text"], "page": hit["page"]} for hit in fused],
    "sources": [
        {"page": hit["page"], "retrieval": "hybrid"}
        for hit in fused
        ]
    }
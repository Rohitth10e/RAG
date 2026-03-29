from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import os
import dotenv

dotenv.load_dotenv()  # Load environment variables from .env file

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=6333)
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def query_rag(question: str) -> dict:
    # step 1- embed the question using same model used during ingestion
    query_vector = embedder.encode(question).tolist()
    
    # step 2- search qdrant for top 5 most similar chunks
    result = client.query_points(
        collection_name="springboot_docs",
        query=query_vector,
        limit=5,
        with_payload=True
    ).points
    
    # step 3- extract text from the returned chunks
    context_chunks = [r.payload["text"] for r in result]
    context = "\n\n---\n\n".join(context_chunks)
    
    # step 4- send context + question to Groq LLM
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": '''Answer the user's question using only the context provided.
                If the answer is not in the context, say 'I don't know'.'''
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    
    # Step 5 — print answer + which pages it came from
    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": [
            {"page": r.payload["page"], "score": round(r.score, 3)}
            for r in result
        ]
    }
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import os
import dotenv

dotenv.load_dotenv()  # Load environment variables from .env file

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def query_rag(question: str):
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
    
    answer = response.choices[0].message.content
    
    # Step 5 — print answer + which pages it came from
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{answer}")
    print("\nSources:")
    for r in result:
        print(f"  Page {r.payload['page']}  (similarity score: {r.score:.3f})")
    
query_rag("What is Spring Boot auto-configuration?")
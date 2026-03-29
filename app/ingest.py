from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
import uuid



pdf_path = Path(__file__).resolve().parent.parent / "spring-boot-reference.pdf"
print(pdf_path)


# load pdf, extract text from pdf returns Document objects(often one per page)
loader = PyPDFLoader(file_path=str(pdf_path))
docs = loader.load()

# Split the docs into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents=docs)

# embedding model which takes in chunks and converts it into vector embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, free, fast

# Bridge: Embed + Upsert to Qdrant
client = QdrantClient(host="localhost", port=6333)

# Let's create Qdrant collection to store vectors
# size → how many numbers are in each vector (384 for all-MiniLM-L6-v2)
# distance → how to measure similarity between vectors (always Cosine for text)

client.recreate_collection(
    collection_name="springboot_docs",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

# recreate_collection drops the collection if it already exists and creates a fresh one. Safe to run multiple times during development
print("Collection created")

# texts is now plain list of strings - the raw text of each chunk
texts = [chunk.page_content for chunk in chunks]

BATCH_SIZE = 100
all_embeddings = []

for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i: i+ BATCH_SIZE]
    batch_embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()  # ← lowercase
    all_embeddings.extend(batch_embeddings)
    print(f"Embedded {min(i + BATCH_SIZE, len(texts))} / {len(texts)} chunks")

# model.encode() returns a numpy array of shape (100, 384) — 100 vectors, each with 384 numbers. .tolist() converts it to a plain Python list so Qdrant can accept it.

# upsert to qdrant
# { id, vector, payload(original_text + meta_data) }

points = []

for chunk, embedding in zip(chunks, all_embeddings):
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={
            "text":chunk.page_content,
            "page":chunk.metadata.get("page"),
            "source":chunk.metadata.get("source")
        }
    )
    points.append(point)
    
# Now upsert in batches
for i in range(0, len(points), BATCH_SIZE):
    batch = points[i : i + BATCH_SIZE]
    client.upsert(
        collection_name="springboot_docs",
        points=batch,
        wait=True   # wait for confirmation before moving to next batch
    )
    print(f"Upserted {min(i + BATCH_SIZE, len(points))} / {len(points)} points")

print("Done — all chunks stored in Qdrant")

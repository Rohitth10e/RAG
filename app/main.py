from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag import query_rag

app = FastAPI(title="DocTalk", description="RAG API for Spring Boot docs")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    return query_rag(request.question)
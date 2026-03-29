from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from app.rag import query_rag
import os
from dotenv import load_dotenv

load_dotenv()

groq_llm = LangchainLLMWrapper(ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_retries=1,
))

hf_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

faithfulness.llm = groq_llm
answer_relevancy.llm = groq_llm
answer_relevancy.embeddings = hf_embeddings

QUESTIONS = [
    "What is Spring Boot auto-configuration?",
    "What does @ConditionalOnMissingBean do?",
    "How does Spring Boot handle externalized configuration?",
    "What is the purpose of @SpringBootApplication?",
    "What is Spring Boot Actuator used for?",
]

print("Running evaluation...")
questions, answers, contexts = [], [], []

for i, question in enumerate(QUESTIONS):
    print(f"  [{i+1}/{len(QUESTIONS)}] {question[:55]}...")
    result = query_rag(question)
    questions.append(question)
    answers.append(result["answer"])
    contexts.append([c["text"] for c in result["contexts"]])

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
})

scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy],
    run_config=RunConfig(max_workers=1, timeout=120)
)

print("\nRAGAS Results:")
print(scores)
from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import rag_graph


app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="RAG chatbot using Pinecone, LangGraph and Gemini",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float


@app.get("/")
def home():
    return {
        "message": "Agentic AI RAG Chatbot API is running"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    result = rag_graph.invoke({
        "question": request.question,
        "context": [],
        "answer": "",
        "score": 0.0
    })

    return {
        "query": request.question,
        "final_answer": result["answer"],
        "retrieved_context_chunks": result["context"],
        "confidence_score": result["score"]
    }
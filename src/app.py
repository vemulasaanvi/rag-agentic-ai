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
    answer: str
    context: list[str]
    confidence: float


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
        "answer": result["answer"],
        "context": result["context"],
        "confidence": result["score"]
    }
import os
from typing import TypedDict

from dotenv import load_dotenv
from pinecone import Pinecone
from google import genai
from langgraph.graph import StateGraph, START, END

load_dotenv()





PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-rag-index")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

client = genai.Client(api_key=GEMINI_API_KEY)





class RAGState(TypedDict):
    question: str
    context: list[str]
    answer: str
    score: float





def retrieve(state: RAGState):
    question = state["question"]

    results = index.search(
        namespace="default",
        query={
            "inputs": {
                "text": question
            },
            "top_k": 3
        }
    )

    context = []
    scores = []

    for result in results["result"]["hits"]:
        context.append(result["fields"]["text"])
        scores.append(result.score)

    average_score = (
        sum(scores) / len(scores)
        if scores
        else 0
    )

    return {
        "context": context,
        "score": average_score
    }




def generate(state: RAGState):
    question = state["question"]
    context = state["context"]
    score = state["score"]


     
    if score < 0.30:
        return {
            "answer": (
                "I couldn't find this information in the "
                "provided Agentic AI eBook."
            )
        }

    combined_context = "\n\n".join(context)

    prompt = f"""
You are a RAG chatbot for the Agentic AI eBook.

STRICT RULES:

1. Answer ONLY using the provided eBook context.
2. Do NOT use outside knowledge.
3. Do NOT make up information.
4. If the answer cannot be found in the context, say:
   "I couldn't find this information in the provided Agentic AI eBook."
5. Keep the answer clear and concise.

EBOOK CONTEXT:
{combined_context}

USER QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return {
        "answer": response.text
    }





workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

rag_graph = workflow.compile()




if __name__ == "__main__":

    question = input("Ask a question: ")

    result = rag_graph.invoke({
        "question": question,
        "context": [],
        "answer": "",
        "score": 0.0
    })

    print("\n================ ANSWER ================")
    print(result["answer"])

    print("\n================ CONFIDENCE ================")
    print(result["score"])

    print("\n================ RETRIEVED CONTEXT ================")

    for i, chunk in enumerate(result["context"], 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk[:500])
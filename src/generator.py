import os
from dotenv import load_dotenv
from pinecone import Pinecone
from google import genai

load_dotenv()



PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-rag-index")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)



client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def retrieve_context(question, top_k=3):
    """Retrieve the most relevant chunks from the eBook."""

    results = index.search(
        namespace="default",
        query={
            "inputs": {
                "text": question
            },
            "top_k": top_k
        }
    )

    context = []
    scores = []

    for result in results["result"]["hits"]:
        context.append(result["fields"]["text"])
        scores.append(result["_score"])

    return context, scores


def generate_answer(question):
    """Generate an answer using only the retrieved eBook content."""

    context, scores = retrieve_context(question)

    combined_context = "\n\n".join(context)

    prompt = f"""
You are a RAG chatbot for the Agentic AI eBook.

You MUST follow these rules:

1. Answer ONLY using the provided context.
2. Do NOT use outside knowledge.
3. Do NOT invent or assume information.
4. If the answer is not available in the context, respond exactly:
   "I couldn't find this information in the provided Agentic AI eBook."

CONTEXT FROM THE EBOOK:
{combined_context}

USER QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    answer = response.text

    confidence = (
        sum(scores) / len(scores)
        if scores
        else 0
    )

    return {
        "answer": answer,
        "context": context,
        "confidence": confidence
    }


if __name__ == "__main__":
    question = input("Ask a question: ")

    result = generate_answer(question)

    print("\n================ ANSWER ================")
    print(result["answer"])

    print("\n================ CONFIDENCE ================")
    print(result["confidence"])

    print("\n================ RETRIEVED CONTEXT ================")

    for i, chunk in enumerate(result["context"], 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk[:500])
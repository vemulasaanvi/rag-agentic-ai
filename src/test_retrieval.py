import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-rag-index")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

question = "What is Agentic AI?"

print("Searching Pinecone...")
print()

results = index.search(
    namespace="default",
    query={
        "inputs": {
            "text": question
        },
        "top_k": 3
    }
)

for i, result in enumerate(results["result"]["hits"], 1):
    print(f"--- Result {i} ---")
    print(f"Score: {result['_score']}")
    print(f"Text: {result['fields']['text'][:500]}")
    print()
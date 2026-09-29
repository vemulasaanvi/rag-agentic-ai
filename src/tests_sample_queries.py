from src.graph import rag_graph


sample_questions = [
    "What is Agentic AI?",
    "How does Agentic AI differ from traditional AI?",
    "What are the key characteristics of Agentic AI?",
    "What value does Agentic AI bring to businesses?",
    "How is Agentic AI used in real-world business applications?",
    "Who won the 2022 FIFA World Cup?"
]


for i, question in enumerate(sample_questions, 1):

    print("\n" + "=" * 70)
    print(f"TEST {i}")
    print("=" * 70)

    print(f"\nQuestion: {question}")

    result = rag_graph.invoke({
        "question": question,
        "context": [],
        "answer": "",
        "score": 0.0
    })

    print("\nAnswer:")
    print(result["answer"])

    print("\nConfidence:")
    print(result["score"])

    print("\nRetrieved Context:")
    for j, chunk in enumerate(result["context"], 1):
        print(f"\n--- Chunk {j} ---")
        print(chunk[:300])
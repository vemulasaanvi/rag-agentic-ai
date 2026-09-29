import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

load_dotenv()

PDF_PATH = "data/Ebook-Agentic-AI.pdf"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-rag-index")


def run_ingestion():
    print("Loading PDF...")

    
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )

    index = pc.Index(INDEX_NAME)

    
    records = []

    for i, chunk in enumerate(chunks):
        records.append({
            "_id": f"chunk-{i}",
            "text": chunk.page_content,
            "page": chunk.metadata.get("page", 0),
            "source": "Ebook-Agentic-AI.pdf"
        })

    print("Uploading chunks to Pinecone...")

    
    batch_size = 96

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]

        index.upsert_records(
            namespace="default",
            records=batch
        )

        print(
            f"Uploaded {start + len(batch)} / {len(records)} chunks"
        )

    print("Ingestion completed successfully!")
    print(f"Uploaded {len(records)} chunks to '{INDEX_NAME}'.")


if __name__ == "__main__":
    run_ingestion()
# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly from the provided Agentic AI eBook.

## Features

PDF document ingestion
Recursive text chunking
Pinecone vector search with integrated embeddings
Top-3 relevant context retrieval
LangGraph-based RAG workflow
Gemini-based grounded answer generation
Strict document grounding
Out-of-scope question refusal
Relevance/confidence score
FastAPI REST API

## Architecture

    text
Agentic AI eBook
       ↓
PDF Loader
       ↓
Text Chunking
(1000 characters / 200 overlap)
       ↓
Pinecone
       ↓
Top-3 Context Retrieval
       ↓
LangGraph
       ↓
Gemini
       ↓
Answer + Context + Confidence

## Tech Stack

Python 3.11 – Programming language
LangChain – Document loading and text processing
LangGraph – RAG workflow orchestration
Pinecone – Vector storage and similarity search
Gemini API – Grounded answer generation
FastAPI – REST API
Uvicorn – API server
PyPDF – PDF document loading

## Project Structure

```text
rag-agentic-ai/
│
├── data/
│   └── Ebook-Agentic-AI.pdf
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── test_retrieval.py
│   ├── generator.py
│   ├── graph.py
│   ├── app.py
│   └── tests_sample_queries.py
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt

## Setup & Installation

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd rag-agentic-ai

### 2. Create a Virtual Environment

Create a Python virtual environment:

```bash
python -m venv venv

```markdown
Activate the virtual environment on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1

### 3. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt

### 4. Configure Environment Variables

Create a `.env` file in the project root and add:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-rag-index
GEMINI_API_KEY=your_gemini_api_key

### 5. Add the Agentic AI eBook

Place the provided PDF inside the `data` folder:

```text
data/Ebook-Agentic-AI.pdf

### 6. Ingest the Document

Run the following command to load the PDF, split it into chunks, and upload the chunks to Pinecone:

```bash
python src/ingestion.py

### 7. Run the FastAPI Application

Start the API server:

```bash
uvicorn src.app:app --reload

### The API will be available at:

http://127.0.0.1:8000

### 8. Open API Documentation

Open the following URL in your browser:

`http://127.0.0.1:8000/docs`

The Swagger UI can be used to test the `/chat` endpoint.
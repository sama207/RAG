# CV RAG System with FastAPI + Streamlit

A Retrieval-Augmented Generation (RAG) system for querying candidate CVs using semantic search, Pinecone vector database, FastAPI backend, and a Streamlit frontend UI.

---

# Project Overview

This project allows users to ask natural language questions about a collection of CVs.

The system:

1. Stores CV chunks in Pinecone
2. Retrieves the most relevant chunks using embeddings
3. Sends the retrieved context to an LLM
4. Generates grounded answers based only on the CV data
5. Provides both:
   - FastAPI backend API
   - Streamlit frontend chat interface

---

# Features

- FastAPI backend
- Streamlit frontend UI
- Pinecone vector database integration
- SentenceTransformer embeddings
- LangChain RAG pipeline
- OpenRouter LLM integration
- Context-aware candidate retrieval
- Streaming support
- Docker support
- Environment-based configuration

---

# Tech Stack

## Backend

- Python
- FastAPI
- Uvicorn
- LangChain
- LangChain Core
- LangChain OpenAI

## AI / RAG

- SentenceTransformers
- Pinecone
- OpenRouter
- Mistral-7B-Instruct

## Frontend

- Streamlit

## DevOps

- Docker
- Environment Variables (.env)

---

# Project Structure

```text
app/
├── .env
├── Dockerfile
├── requirements.txt
├── rag.py
├── main.py
├── streamlit_app.py
└── README.md
```

---

# System Architecture

```text
User
   ↓
Streamlit Frontend
   ↓
FastAPI Backend
   ↓
RAG Pipeline
   ↓
Pinecone Vector Database
   ↓
Retrieved CV Chunks
   ↓
OpenRouter LLM
   ↓
Generated Answer
```

---

# API Endpoints

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

---

## Ask Question

```http
POST /ask
```

Request Body:

```json
{
  "question": "Which candidates know Python and machine learning?",
  "show_context": false
}
```

Response:

```json
{
  "question": "Which candidates know Python and machine learning?",
  "answer": "Candidate X has experience with Python and ML.",
  "context": null
}
```

---

## Streaming Endpoint

```http
POST /ask-stream
```

Streams generated answer tokens.

---

# Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <project-folder>
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
PINECONE_API_KEY=your_pinecone_key_here
OPENROUTER_API_KEY=your_openrouter_key_here

EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
PINECONE_INDEX=cv-rag
OPENROUTER_MODEL=mistralai/mistral-7b-instruct:free
RETRIEVER_TOP_K=5
```

---

# Run the Backend

```bash
uvicorn main:app --reload
```

Backend runs on:

```text
http://localhost:8000
```

Swagger Docs:

```text
http://localhost:8000/docs
```

---

# Run the Frontend

In another terminal:

```bash
streamlit run streamlit_app.py
```

Frontend runs on:

```text
http://localhost:8501
```

---

# Docker Setup

## Build Docker Image

```bash
docker build -t cv-rag-app .
```

## Run Container

```bash
docker run -p 8000:8000 cv-rag-app
```

---

# Example Questions

- Which candidates have Python experience?
- Who has machine learning skills?
- Which candidates worked with FastAPI?
- Recommend candidates for a backend role.
- Which candidates have experience with databases?

---

# RAG Workflow

1. User submits a question
2. Question is embedded using SentenceTransformers
3. Pinecone retrieves the most relevant CV chunks
4. Retrieved chunks are formatted into context
5. Context + question are sent to the LLM
6. LLM generates grounded answer
7. Response is returned to the frontend

---

# Future Improvements

- Authentication system
- Multi-user chat sessions
- Upload new CVs dynamically
- Better UI/UX design
- Conversation memory
- Candidate ranking system
- Evaluation metrics dashboard
- Hybrid search
- Redis caching

---

# Notes

- The embedding model and Pinecone index are loaded once during startup.
- The RAG chain is reused for every request.
- `.env` should never be committed to Git.
- The frontend communicates with the backend using HTTP requests.

---

# License

This project is for educational and learning purposes.


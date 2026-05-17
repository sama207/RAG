import os
from typing import List

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

# ── Config (read from .env) ──────────────────────────────────────────────────
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
PINECONE_INDEX     = os.getenv("PINECONE_INDEX", "cv-rag")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")
RETRIEVER_TOP_K    = int(os.getenv("RETRIEVER_TOP_K", 5))

# ── Module-level singletons (loaded once at startup) ─────────────────────────
_retriever  = None   # PineconeRetriever instance
_rag_chain  = None   # the full LCEL chain


# ── Part A: Custom LangChain Retriever ───────────────────────────────────────
class PineconeRetriever(BaseRetriever):
    """LangChain-compatible retriever backed by the Pinecone CV index."""

    index:       object
    embed_model: object
    top_k:       int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Step 1: embed the query
        vector = self.embed_model.encode(
            [query], normalize_embeddings=True
        ).tolist()[0]

        # Step 2: query Pinecone
        response = self.index.query(
            vector=vector, top_k=self.top_k, include_metadata=True
        )

        # Step 3: wrap each match as a LangChain Document
        docs = []
        for match in response["matches"]:
            doc = Document(
                page_content=match["metadata"].get("text", ""),
                metadata={
                    "doc_name": match["metadata"].get("doc_name", "unknown"),
                    "score":    round(match["score"], 4),
                },
            )
            docs.append(doc)

        return docs


# ── Part B: Context formatter ─────────────────────────────────────────────────
def format_docs(docs: List[Document]) -> str:
    """Convert retrieved Documents into a single formatted context string."""
    parts = []
    for doc in docs:
        header = (
            f"[Source: {doc.metadata.get('doc_name','unknown')} "
            f"| Score: {doc.metadata.get('score','?')}]"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n---\n".join(parts)


# ── Startup: load models and build chain ─────────────────────────────────────
def init_rag():
    """
    Called ONCE when the FastAPI app starts.
    Loads Pinecone, the embedding model, and assembles the RAG chain.
    Results are stored in module-level singletons so every request reuses them.
    """
    global _retriever, _rag_chain

    # Cell 3: connect to Pinecone + load embedding model
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    embed_model = SentenceTransformer(EMBEDDING_MODEL,
    device="cpu")

    _retriever = PineconeRetriever(
        index=index,
        embed_model=embed_model,
        top_k=RETRIEVER_TOP_K,
    )

    # Cell 5: prompt template
    SYSTEM_TEMPLATE = """\
You are an expert technical recruiter assistant.

RULES:
- Base your answer ONLY on the CV chunks provided in the context below.
- If the context does not contain enough information, say so explicitly.
- When recommending candidates, always cite their CV filename.

---- CV CONTEXT ----
{context}
--------------------
"""
    HUMAN_TEMPLATE = "{question}"

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
    ])

    # Cell 6: LLM via OpenRouter
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model_name=OPENROUTER_MODEL,
        temperature=0.2,
        default_headers={
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title":      "CV-RAG-FastAPI",
        },
    )

    # Cell 7: assemble the LCEL chain
    _rag_chain = (
        {
            "context":  _retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"RAG chain ready. Index='{PINECONE_INDEX}', Model='{OPENROUTER_MODEL}'")


# ── Query helpers (replaces notebook's ask() function) ───────────────────────
def run_query(question: str, show_context: bool = False) -> dict:
    """Run the RAG chain and return a dict with answer (and optionally context)."""
    answer = _rag_chain.invoke(question)

    result = {"question": question, "answer": answer}

    if show_context:
        docs = _retriever._get_relevant_documents(question)
        result["context"] = format_docs(docs)

    return result


def run_query_stream(question: str):
    """Generator that yields answer tokens one by one (for streaming endpoint)."""
    for chunk in _rag_chain.stream(question):
        yield chunk
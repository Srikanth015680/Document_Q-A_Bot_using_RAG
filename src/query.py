


import os
import chromadb
from google import genai
from dotenv import load_dotenv

from src.config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    DB_DIR,
    COLLECTION_NAME,
    TOP_K,
)

load_dotenv()

genai_client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """
You are a precise, professional document Q&A assistant.

Rules:
1. Answer ONLY using the provided context.
2. Cite sources in the format (filename, Page X).
3. If the answer is not present in the context, respond EXACTLY:
"I am sorry, but the provided documents do not contain the answer to your question."
4. Do not use outside knowledge.
5. Keep answers concise and accurate.
"""

_client_cache = {}

MAX_DISTANCE: float = 0.7


def _get_collection(db_path: str = DB_DIR):
    if db_path not in _client_cache:
        if not os.path.isdir(db_path):
            raise RuntimeError(
                f"Vector database not found at '{db_path}'.\n"
                "Run ingestion first: python -m src.ingest"
            )

        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=COLLECTION_NAME)

        _client_cache[db_path] = collection

    return _client_cache[db_path]


def embed_query(text: str):
    response = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[text],
    )

    return response.embeddings[0].values


def query_rag_pipeline(
    user_query: str,
    db_path: str = DB_DIR,
    k: int = TOP_K
):
    collection = _get_collection(db_path)

    query_embedding = embed_query(user_query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    filtered = []

    for doc, meta, dist in zip(docs, metas, distances):
        if dist <= MAX_DISTANCE:
            filtered.append((doc, meta, dist))

    if not filtered:
        return {
            "answer":
            "I am sorry, but the provided documents do not contain the answer to your question.",
            "citations": [],
            "raw_chunks": [],
            "sources": []
        }

    context_blocks = []
    citations = []
    sources = []
    raw_chunks = []

    for doc, meta, dist in filtered:
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")

        citation = f"{source}, Page {page}"

        context_blocks.append(
            f"[Source: {source}, Page: {page}]\n{doc}"
        )

        citations.append(citation)
        raw_chunks.append(doc)

        sources.append({
            **meta,
            "distance": round(dist, 4)
        })

    context_payload = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
{SYSTEM_PROMPT}

CONTEXT INFORMATION:
{context_payload}

USER QUESTION:
{user_query}

ANSWER:
"""

    response = genai_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    return {
        "answer": response.text.strip(),
        "citations": citations,
        "raw_chunks": raw_chunks,
        "sources": sources,
    }


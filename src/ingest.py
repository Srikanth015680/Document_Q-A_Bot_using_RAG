
import os
import sys
import chromadb
from pypdf import PdfReader
from docx import Document as DocxDocument
from tqdm import tqdm
from google import genai

from src.config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    DATA_DIR,
    DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
    SUPPORTED_EXTENSIONS,
)

# Initialize new google-genai client
genai_client = genai.Client(api_key=GEMINI_API_KEY)



# 1. DOCUMENT EXTRACTION


def extract_pdf(file_path: str) -> list[dict]:
    pages = []
    file_name = os.path.basename(file_path)
    try:
        reader = PdfReader(file_path)
        for idx, page in enumerate(reader.pages):
            raw = page.extract_text()
            if raw and raw.strip():
                clean = " ".join(raw.split())
                pages.append({
                    "text": clean,
                    "metadata": {"source": file_name, "page": idx + 1},
                })
    except Exception as exc:
        print(f"  [ERROR] Could not read PDF '{file_name}': {exc}")
    return pages


def extract_docx(file_path: str) -> list[dict]:
    pages = []
    file_name = os.path.basename(file_path)
    block_words = 500
    try:
        doc = DocxDocument(file_path)
        buffer, word_count, page_num = [], 0, 1
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            buffer.append(text)
            word_count += len(text.split())
            if word_count >= block_words:
                pages.append({
                    "text": " ".join(buffer),
                    "metadata": {"source": file_name, "page": page_num},
                })
                buffer, word_count = [], 0
                page_num += 1
        if buffer:
            pages.append({
                "text": " ".join(buffer),
                "metadata": {"source": file_name, "page": page_num},
            })
    except Exception as exc:
        print(f"  [ERROR] Could not read DOCX '{file_name}': {exc}")
    return pages


def extract_txt(file_path: str) -> list[dict]:
    pages = []
    file_name = os.path.basename(file_path)
    block_words = 500
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        words = content.split()
        for i in range(0, len(words), block_words):
            block = " ".join(words[i: i + block_words])
            pages.append({
                "text": block,
                "metadata": {"source": file_name, "page": (i // block_words) + 1},
            })
    except Exception as exc:
        print(f"  [ERROR] Could not read TXT '{file_name}': {exc}")
    return pages


def load_all_documents(data_dir: str) -> list[dict]:
    all_pages: list[dict] = []
    files = [
        f for f in os.listdir(data_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        print(f"\n[WARNING] No supported documents found in '{data_dir}'.")
        sys.exit(1)

    print(f"\n  Found {len(files)} document(s) in '{data_dir}':")
    for f in files:
        print(f"    • {f}")

    print("\n  Extracting text …")
    for file_name in tqdm(files):
        path = os.path.join(data_dir, file_name)
        ext = os.path.splitext(file_name)[1].lower()
        if ext == ".pdf":
            pages = extract_pdf(path)
        elif ext == ".docx":
            pages = extract_docx(path)
        elif ext == ".txt":
            pages = extract_txt(path)
        else:
            continue
        all_pages.extend(pages)
        print(f"    {file_name}  →  {len(pages)} page block(s)")
    return all_pages



# 2. CHUNKING


def chunk_pages(pages: list[dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[dict]:
    chunks: list[dict] = []
    for page in pages:
        text = page["text"]
        meta = page["metadata"]
        text_len = len(text)
        start = 0
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end]
            search_zone = chunk_text[int(chunk_size * 0.8):]
            for sep in [". ", "? ", "! ", " "]:
                idx = search_zone.rfind(sep)
                if idx != -1:
                    end = start + int(chunk_size * 0.8) + idx + len(sep)
                    chunk_text = text[start:end]
                    break
            chunks.append({
                "text": chunk_text.strip(),
                "metadata": {
                    "source": meta["source"],
                    "page": meta["page"],
                    "chunk_range": f"{start}-{end}",
                },
            })
            start += (chunk_size - chunk_overlap)
    return chunks



# 3. EMBED using new google-genai SDK


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed texts using the new google-genai SDK."""
    response = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
    )
    return [e.values for e in response.embeddings]



# 4. VECTOR DATABASE — PERSIST (manual embeddings, no chromadb embedding fn)


def save_to_vector_db(chunks: list[dict], db_path: str = DB_DIR) -> None:
    os.makedirs(db_path, exist_ok=True)

    client = chromadb.PersistentClient(path=db_path)

    # Drop and recreate for clean index
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"\n   Cleared existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    BATCH_SIZE = 100
    total = len(chunks)
    print(f"\n  Embedding & indexing {total} chunks …")

    for batch_start in tqdm(range(0, total, BATCH_SIZE),desc="Indexing"):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        embeddings = embed_texts(texts)

        collection.add(
            ids=[f"chunk_{batch_start + i}" for i in range(len(batch))],
            documents=texts,
            embeddings=embeddings,
            metadatas=[c["metadata"] for c in batch],
        )

    print(f"\n Successfully indexed {total} chunks into '{db_path}'.")
    print(f" Collection contains {collection.count()} chunks.")


# ENTRY POINT


def run_ingestion() -> None:
    print("=" * 60)
    print("  RAG INGESTION PIPELINE")
    print("=" * 60)

    pages = load_all_documents(DATA_DIR)
    print(f"\n Total page blocks extracted: {len(pages)}")

    chunks = chunk_pages(pages)
    print(f"   Total chunks after splitting: {len(chunks)}")
    print(f"     (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    save_to_vector_db(chunks)
    print("\n  Ingestion complete. You can now run main.py to query your docs.")
    print("=" * 60)


if __name__ == "__main__":
    run_ingestion()

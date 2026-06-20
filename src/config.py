

import os
from dotenv import load_dotenv

load_dotenv()


# API Keys

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set. "
        "Copy .env.example to .env and add your key."
    )


# Model Names

EMBEDDING_MODEL: str = "gemini-embedding-001"
GENERATION_MODEL: str = "gemini-2.5-flash"


# Paths

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR: str = os.path.join(BASE_DIR, "data")
DB_DIR: str = os.path.join(BASE_DIR, "db")


# Chunking Strategy

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200


# Retrieval

TOP_K: int = 4

# Reject weak matches
MAX_DISTANCE: float = 0.7


# ChromaDB

COLLECTION_NAME: str = "document_knowledge_base"


# Supported document types

SUPPORTED_EXTENSIONS: tuple = (
    ".pdf",
    ".docx",
    ".txt",
)


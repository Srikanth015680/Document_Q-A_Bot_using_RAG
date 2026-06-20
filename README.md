
# Document Q&A Bot using RAG

## Project Overview

This project is a Document Question Answering Bot built using the Retrieval-Augmented Generation (RAG) approach. It allows users to ask questions about a collection of documents and receive answers based only on the information available in those documents.

The application supports PDF, DOCX, and TXT files. During the ingestion process, documents are loaded, split into smaller chunks, converted into embeddings, and stored in a ChromaDB vector database. When a user asks a question, the system retrieves the most relevant document chunks and uses Google's Gemini model to generate an answer along with source citations.

This project was developed as part of an AI Engineering Internship assignment to demonstrate understanding of the complete RAG pipeline, including document ingestion, chunking, embedding generation, vector storage, retrieval, and answer generation.

## Tech Stack

Python 3.11+

Google Gemini 2.5 Flash

Gemini Embedding Model (gemini-embedding-001)

ChromaDB

Streamlit

pypdf

python-docx

python-dotenv

tqdm

## Architecture

The application follows a simple RAG workflow:

Documents -Text Extraction -Chunking - Embedding Generation -ChromaDB Vector Store - Retrieval - Gemini - Answer with Citations

### Components

**ingest.py** – Loads documents, extracts text, creates chunks, generates embeddings, and stores them in ChromaDB.

**query.py** – Retrieves relevant chunks from the vector database and generates grounded answers.

**main.py** – Command-line interface for interacting with the bot.

**app.py** – Streamlit web application.

**config.py** – Stores configuration values such as chunk size, overlap, model names, and database paths.

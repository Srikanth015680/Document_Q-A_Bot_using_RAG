

import os
import streamlit as st

from src.query import query_rag_pipeline
from src.config import TOP_K, DATA_DIR, DB_DIR


# Page config

st.set_page_config(
    page_title="Document Q&A Bot",
    layout="wide",
)


# Sidebar

with st.sidebar:
    st.title(" Settings")

    top_k = st.slider(
        "Top-K chunks to retrieve",
        min_value=1,
        max_value=10,
        value=TOP_K,
        help="More chunks = more context, but slower and costlier."
    )

    show_chunks = st.toggle(
        "Show retrieved source chunks",
        value=False
    )

    st.divider()
    st.markdown("###  Indexed Documents")

    data_files = [
        f for f in os.listdir(DATA_DIR)
        if os.path.splitext(f)[1].lower() in (".pdf", ".docx", ".txt")
    ] if os.path.isdir(DATA_DIR) else []

    if data_files:
        for f in data_files:
            st.markdown(f"- `{f}`")
    else:
        st.warning("No documents found in /data. Run ingestion first.")

    db_exists = os.path.isdir(DB_DIR)

    st.divider()
    st.markdown("###  Vector DB Status")

    if db_exists:
        st.success(" Database ready")
    else:
        st.error(" Not built yet — run `python -m src.ingest`")

    st.divider()
    st.markdown(
        "**Tech Stack**\n"
        "- Gemini `gemini-embedding-001`\n"
        "- ChromaDB (local, persistent)\n"
        "- Gemini 2.5 Flash (generation)\n"
        "- Recursive character chunking"
    )


# Main Area

st.title(" Document Q&A Bot")
st.caption(
    "RAG-powered — answers grounded in your documents, with citations."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Documents", len(data_files))

with col2:
    st.metric("Top-K", top_k)

with col3:
    st.metric("Database", "Ready" if db_exists else "Not Ready")

if not db_exists:
    st.error(
        "Vector database not found. "
        "Please run `python -m src.ingest` first, then refresh this page."
    )
    st.stop()


# Chat History

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "citations" in msg:
            st.markdown("** Sources:**")

            for citation in msg["citations"]:
                st.markdown(f"- `{citation}`")

        if (
            msg["role"] == "assistant"
            and show_chunks
            and "chunks" in msg
        ):
            with st.expander(" Retrieved source chunks"):
                for i, (chunk, src) in enumerate(
                    zip(msg["chunks"], msg["sources"]),
                    start=1
                ):
                    st.markdown(
                        f"**Chunk {i}** · "
                        f"`{src['source']}, Page {src['page']}`"
                    )

                    st.text(
                        chunk[:400]
                        + ("…" if len(chunk) > 400 else "")
                    )

                    st.divider()


# User Input
 
if prompt := st.chat_input(
    "Ask a question about your documents …"
):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(" Searching and generating answer …"):
            try:
                result = query_rag_pipeline(
                    prompt,
                    k=top_k,
                )

                answer = result["answer"]
                citations = result["citations"]
                raw_chunks = result["raw_chunks"]
                sources = result["sources"]

                st.markdown(answer)

                st.markdown("---")
                st.markdown("** Sources:**")

                for i, citation in enumerate(citations, start=1):
                    st.markdown(f"{i}. `{citation}`")

                if show_chunks:
                    with st.expander(
                        " Retrieved source chunks"
                    ):
                        for i, (chunk, src) in enumerate(
                            zip(raw_chunks, sources),
                            start=1
                        ):
                            st.markdown(
                                f"**Chunk {i}** · "
                                f"`{src['source']}, Page {src['page']}`"
                            )

                            st.text(
                                chunk[:400]
                                + ("…" if len(chunk) > 400 else "")
                            )

                            st.divider()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "chunks": raw_chunks,
                        "sources": sources,
                    }
                )

            except Exception as exc:
                err_msg = f" Error: {exc}"

                st.error(err_msg)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": err_msg,
                    }
                )


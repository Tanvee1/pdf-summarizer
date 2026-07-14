import fitz
import faiss
import numpy as np
import streamlit as st

from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Summarization model
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    token=st.secrets["huggingface"]["token"]
)


def extract_text_from_pdf(pdf_file):
    doc = fitz.open(
        stream=pdf_file.read(),
        filetype="pdf"
    )

    text = ""

    for page in doc:
        text += page.get_text()

    return text


def chunk_text(
    text,
    chunk_size=1000,
    overlap=200
):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def embed_chunks(chunks):
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    return embeddings.astype("float32")


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


def retrieve_top_chunks(
    query,
    index,
    chunks,
    top_k=5
):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    return [chunks[i] for i in indices[0]]


def summarize_context(context):
    summaries = []

    context_chunks = [
        context[i:i+1000]
        for i in range(0, len(context), 1000)
    ]

    for chunk in context_chunks:
        result = summarizer(
            chunk,
            max_length=200,
            min_length=50,
            do_sample=False
        )

        summaries.append(
            result[0]["summary_text"]
        )

    return "\n\n".join(summaries)


def run_rag_pipeline(
    pdf_file,
    query
):
    text = extract_text_from_pdf(
        pdf_file
    )

    chunks = chunk_text(
        text
    )

    embeddings = embed_chunks(
        chunks
    )

    index = build_faiss_index(
        embeddings
    )

    relevant_chunks = retrieve_top_chunks(
        query,
        index,
        chunks
    )

    context = "\n".join(
        relevant_chunks
    )

    answer = summarize_context(
        context
    )

    return answer

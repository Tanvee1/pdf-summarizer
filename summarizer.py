import fitz
import faiss
import numpy as np
import streamlit as st

from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# ---------------- HF Login ----------------
login(token=st.secrets["huggingface"]["token"])

# ---------------- Embedding Model ----------------
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ---------------- Summarization Model ----------------
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

# ---------------- PDF Extraction ----------------
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(
        stream=pdf_file.read(),
        filetype="pdf"
    )

    text = ""

    for page in doc:
        text += page.get_text()

    return text


# ---------------- Chunking ----------------
def chunk_text(
    text,
    chunk_size=1000,
    overlap=200
):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunks.append(
            text[start:end]
        )

        start += chunk_size - overlap

    return chunks


# ---------------- Embeddings ----------------
def embed_chunks(chunks):
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    return embeddings.astype("float32")


# ---------------- FAISS ----------------
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# ---------------- Retrieval ----------------
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

    retrieved_chunks = []

    for idx in indices[0]:
        if idx < len(chunks):
            retrieved_chunks.append(
                chunks[idx]
            )

    return retrieved_chunks


# ---------------- Summarization ----------------
def summarize_text(text):
    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=200,
        min_length=50,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary


# ---------------- Main Pipeline ----------------
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

    retrieved_chunks = retrieve_top_chunks(
        query,
        index,
        chunks
    )

    context = "\n".join(
        retrieved_chunks
    )

    summary = summarize_text(
        context
    )

    return summary

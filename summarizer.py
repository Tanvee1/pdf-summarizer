import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
import streamlit as st

# Use API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

def embed_chunks(chunks):
    return embedding_model.encode(chunks)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_top_chunks(query, index, chunks, top_k=5):
    q_embed = embedding_model.encode([query])
    distances, indices = index.search(q_embed, top_k)
    return [chunks[i] for i in indices[0]]

def summarize_with_openai(text):
    prompt = f"Summarize the following:\n\n{text}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def run_rag_pipeline(pdf_file, query):
    text = extract_text_from_pdf(pdf_file)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(np.array(embeddings))
    top_chunks = retrieve_top_chunks(query, index, chunks)
    return summarize_with_openai(" ".join(top_chunks))

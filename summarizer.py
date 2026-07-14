import fitz
import faiss
import numpy as np
import nltk
import streamlit as st

from sentence_transformers import SentenceTransformer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Download NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


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
    chunk_size=500,
    overlap=50
):
    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size - overlap
    ):
        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def embed_chunks(chunks):
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    return embeddings.astype(
        "float32"
    )


def build_faiss_index(
    embeddings
):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

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

    retrieved_chunks = []

    for idx in indices[0]:
        if idx < len(chunks):
            retrieved_chunks.append(
                chunks[idx]
            )

    return retrieved_chunks


def summarize_text(
    text,
    sentence_count=5
):
    parser = PlaintextParser.from_string(
        text,
        Tokenizer("english")
    )

    summarizer = LexRankSummarizer()

    summary = summarizer(
        parser.document,
        sentence_count
    )

    return " ".join(
        str(sentence)
        for sentence in summary
    )


def run_rag_pipeline(
    pdf_file,
    query
):
    text = extract_text_from_pdf(
        pdf_file
    )

    if not text.strip():
        return (
            "No text could be extracted "
            "from the PDF."
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

    top_chunks = retrieve_top_chunks(
        query,
        index,
        chunks
    )

    context = " ".join(
        top_chunks
    )

    summary = summarize_text(
        context
    )

    return summary

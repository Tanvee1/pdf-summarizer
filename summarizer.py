import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""

    for page in doc:
        text += page.get_text()

    return text


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []

    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])

    return chunks


def embed_chunks(chunks):
    embeddings = embedding_model.encode(chunks)
    return np.array(embeddings).astype("float32")


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


def retrieve_top_chunks(query, index, chunks, top_k=5):
    query_embedding = embedding_model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    return [chunks[i] for i in indices[0]]


def summarize_text(text, sentence_count=5):
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
        [str(sentence) for sentence in summary]
    )


def run_rag_pipeline(pdf_file, query):
    text = extract_text_from_pdf(pdf_file)

    chunks = chunk_text(text)

    embeddings = embed_chunks(chunks)

    index = build_faiss_index(embeddings)

    retrieved_chunks = retrieve_top_chunks(
        query,
        index,
        chunks
    )

    context = " ".join(retrieved_chunks)

    return summarize_text(context)

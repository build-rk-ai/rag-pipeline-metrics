from datetime import datetime
import hashlib
import fitz 
from pathlib import Path
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import pickle

def create_or_get_index(index_name: str, dimension: int = 1536, metric: str = "cosine"):
    # creates Pinecone index if it doesn't exist, returns it either way
    pc = Pinecone()
    if index_name not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)


def load_pdf(file_path):
    # extracts text page by page, skips blank pages
    pdf_path = Path(file_path)
    if not pdf_path.exists(): 
        raise FileNotFoundError(f"file not found {file_path}")
    pages = [] 
    with fitz.open(file_path) as doc: 
        for index, page in enumerate(doc): 
            text = page.get_text("text").strip()
            if text: 
                pages.append({ 
                    "page_num": index + 1, 
                    "text": text, 
                    "source": str(Path(file_path))
                })
    return pages 


def improved_chunking(docs, chunk_size=800, chunk_overlap=150):
    # splits full document text into overlapping chunks, respecting paragraph and sentence boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )
    full_text = "\n\n".join(doc['text'] for doc in docs)
    return splitter.split_text(full_text)


def get_llm():
    # returns OpenAI client using OPENAI_API_KEY from environment
    return OpenAI()


def get_embedding(text, model="text-embedding-3-small") -> list[float]:
    # embeds a single text string using OpenAI
    text = text.replace("\n", " ")
    response = get_llm().embeddings.create(input=text, model=model)
    return response.data[0].embedding


def get_embeddings(chunks: list[str], model="text-embedding-3-small") -> list[list[float]]:
    # embeds a list of chunks sequentially
    return [get_embedding(chunk, model) for chunk in chunks]


def get_pinecone_index():
    # returns handle to the rag-basics Pinecone index
    return Pinecone().Index("rag-basics")


def get_hash(chunk, file_path):
    # deterministic 16-char ID based on file path and chunk content
    key = f"{file_path}::{chunk}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def upsert_pinecone(index, chunks, embeddings, file_path):
    # stores chunks and embeddings in Pinecone with metadata
    batch = []
    for chunk, embedding in zip(chunks, embeddings):
        batch.append({
            "id": get_hash(chunk, file_path),
            "values": embedding,
            "metadata": {
                "text": chunk,
                "source": file_path or "unknown",
                "timestamp": datetime.now().timestamp()
            }
        })
    index.upsert(vectors=batch)


def delete_pinecone(index):
    # wipes all vectors from the index
    index.delete(delete_all=True)


def build_bm25_index(chunks):
    # builds BM25 keyword index from chunks and saves to disk
    tokenized = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    with open("bm25_index.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"BM25 index saved with {len(chunks)} chunks")
    return bm25


def main():
    SCRIPT_DIR = Path(__file__).parent
    FILE_PATH = SCRIPT_DIR / "data/lifecycle_genai.pdf"

    index = create_or_get_index("rag-basics")
    delete_pinecone(index)
    pages = load_pdf(FILE_PATH)
    chunks = improved_chunking(pages)
    embeds = get_embeddings(chunks)
    upsert_pinecone(index, chunks, embeds, file_path=str(FILE_PATH))
    build_bm25_index(chunks)


if __name__ == "__main__":
    main()
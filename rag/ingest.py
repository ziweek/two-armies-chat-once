from dotenv import load_dotenv
load_dotenv()

import faiss
from langchain_community.vectorstores import FAISS

from utils.loader import load_pdf
from utils.splitter import split_documents
from utils.embed import get_embeddings


def ingest_documents(file_path):
    raw_documents = load_pdf(file_path=file_path)
    splitted_documents = split_documents(raw_documents=raw_documents)
    embeddings_ollama = get_embeddings()
    
    print("ingest.py ingesting...")
    vector_store = FAISS.from_documents(
        documents=splitted_documents,
        embedding=embeddings_ollama
        )
    vector_store.save_local("rag/.faiss_index_taco")
    print("ingest.py ingested")

if __name__ == "__main__":
    ingest_documents("rag/src/8A-Blue-Book-en.pdf")
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

def load_pdf(file_path: str):
    loader = PyPDFLoader(
        file_path=file_path
    )
    raw_documents = loader.load()
    print(f"utils/loader.py loaded {len(raw_documents)} documents")
    return raw_documents

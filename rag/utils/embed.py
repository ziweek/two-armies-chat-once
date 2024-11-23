from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import OllamaEmbeddings

def get_embeddings():
    embeddings_ollama = OllamaEmbeddings(
        model="nomic-embed-text",
    )
    # embedded_documents = embeddings_ollama.embed_documents(splitted_documents)
    return embeddings_ollama

# embeddings.py
from langchain.embeddings import HuggingFaceEmbeddings

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Return a Hugging Face embedding model wrapper for LangChain.
    Default model: sentence-transformers/all-MiniLM-L6-v2
    """
    # If you need to use HF_TOKEN for private models, set the environment variable HF_TOKEN
    return HuggingFaceEmbeddings(model_name=model_name)

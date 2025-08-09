# rag_pipeline.py
import os
from typing import List, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from embeddings import get_embedding_model
from gemini_llm import GeminiLLM

class RAGPipeline:
    """
    RAG pipeline:
      - receives list of Documents (already preprocessed by utils)
      - optionally chunk long docs using RecursiveCharacterTextSplitter
      - create/load Chroma vectorstore (persisted)
      - provide a RetrievalQA chain using GeminiLLM (gemini-2.5-flash) by default
    """

    def __init__(self, persist_dir: str = "./vectorstore", embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", llm_choice: str = "gemini"):
        os.makedirs(persist_dir, exist_ok=True)
        self.persist_dir = persist_dir
        self.embedding = get_embedding_model(embedding_model_name)
        self.db: Optional[Chroma] = None
        self.llm_choice = llm_choice

    def chunk_documents(self, docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(docs)

    def create_or_load_db(self, docs: List[Document], collection_name: str = "default", overwrite: bool = False):
        """
        Create a chroma collection from documents (with chunking).
        If overwrite=True and collection exists, it will be replaced.
        """
        # chunk them (CSV rows are typically short so this is fast)
        chunks = self.chunk_documents(docs)
        # create Chroma vectorstore
        self.db = Chroma.from_documents(documents=chunks, embedding=self.embedding, persist_directory=self.persist_dir, collection_name=collection_name)
        # persist to disk
        self.db.persist()
        return self.db

    def get_retriever(self, k: int = 6):
        if not self.db:
            raise ValueError("Vector DB not created yet; call create_or_load_db first.")
        return self.db.as_retriever(search_kwargs={"k": k})

    def get_qa_chain(self, temperature: float = 0.0):
        # choose LLM
        if self.llm_choice == "gemini":
            llm = GeminiLLM(model="gemini-2.5-flash", temperature=temperature)
        else:
            # fallback: if you want to support OpenAI, integrate it here
            from langchain.llms import OpenAI
            llm = OpenAI(temperature=temperature)

        retriever = self.get_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        return qa

    def persist(self):
        if self.db:
            self.db.persist()

# app.py
import os
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings

# Load HuggingFace embeddings (free & local)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to read multiple CSVs and merge them into a list of Documents
def load_csvs(files):
    docs = []
    for file in files:
        df = pd.read_csv(file)
        text_data = df.to_string(index=False)
        docs.append(Document(page_content=text_data, metadata={"source": file.name}))
    return docs

# Function to chunk data
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# Streamlit UI
st.set_page_config(page_title="Multi-CSV RAG", layout="wide")
st.title("📊 CSV Data Q&A (Offline Embeddings)")

uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing files..."):
        docs = load_csvs(uploaded_files)
        chunks = chunk_documents(docs)

        # Store in Chroma (local vector DB)
        vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Gemini for answering (Google API Key required)
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

    query = st.text_input("Ask a question about your data:")
    if query:
        with st.spinner("Searching..."):
            result = qa({"query": query})
            st.write("**Answer:**", result["result"])
            st.write("**Sources:**", [doc.metadata["source"] for doc in result["source_documents"]])

# utils.py
import os
from pathlib import Path
import pandas as pd
from langchain.document_loaders import (
    TextLoader,
    UnstructuredWordDocumentLoader,
    PyPDFLoader
)
from langchain.schema import Document
from typing import List

def save_uploaded_file(uploaded, target_dir: str) -> str:
    os.makedirs(target_dir, exist_ok=True)
    file_path = Path(target_dir) / uploaded.name
    with open(file_path, "wb") as f:
        f.write(uploaded.getbuffer())
    return str(file_path)

def load_file_to_documents(path: str) -> List[Document]:
    """
    Load a file to a list of LangChain Document objects.
    - CSV/Excel: create one Document per row (fast & granular)
    - PDF/DOCX/TXT: use loaders, produce Documents (may further be chunked)
    """
    lower = path.lower()
    docs = []
    if lower.endswith(".csv"):
        df = pd.read_csv(path, low_memory=False)
        # Convert each row to a short text: "col1: val1; col2: val2; ..."
        for idx, row in df.iterrows():
            # create a compact text per row; truncate long fields if needed
            parts = []
            for col in df.columns:
                val = row[col]
                # stringify with small truncation
                textval = str(val)
                if len(textval) > 400:
                    textval = textval[:400] + "..."
                parts.append(f"{col}: {textval}")
            content = " ; ".join(parts)
            meta = {"source": Path(path).name, "row": int(idx)}
            docs.append(Document(page_content=content, metadata=meta))
        return docs

    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        df = pd.read_excel(path)
        for idx, row in df.iterrows():
            parts = []
            for col in df.columns:
                val = row[col]
                textval = str(val)
                if len(textval) > 400:
                    textval = textval[:400] + "..."
                parts.append(f"{col}: {textval}")
            content = " ; ".join(parts)
            meta = {"source": Path(path).name, "row": int(idx)}
            docs.append(Document(page_content=content, metadata=meta))
        return docs

    if lower.endswith(".pdf"):
        loader = PyPDFLoader(path)
        return loader.load()

    if lower.endswith(".docx") or lower.endswith(".doc"):
        loader = UnstructuredWordDocumentLoader(path)
        return loader.load()

    # fallback text
    loader = TextLoader(path, encoding="utf-8")
    return loader.load()

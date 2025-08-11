import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from pydantic import BaseModel

from .rag import answer_question

load_dotenv(override=True)

app = FastAPI(title="GCP RAG Chatbot", version="0.1.0")


class AskRequest(BaseModel):
    question: str


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
async def ask(req: AskRequest, k: Optional[int] = Query(default=None)) -> dict:
    result = answer_question(req.question, top_k=k)
    return result
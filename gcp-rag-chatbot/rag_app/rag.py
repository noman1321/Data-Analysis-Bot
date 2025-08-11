import os
import logging
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from google.cloud import bigquery

from vertexai import init as vertexai_init
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

load_dotenv(override=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
BQ_DATASET = os.getenv("BQ_DATASET", "rag_demo")
BQ_TABLE = os.getenv("BQ_TABLE", "docs")
BQ_VECTOR_INDEX = os.getenv("BQ_VECTOR_INDEX", f"idx_{os.getenv('BQ_TABLE', 'docs')}_emb")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL", "gemini-1.5-pro-002")

# Lazy-initialized globals
_vertexai_inited: bool = False
_embedding_model: Optional[TextEmbeddingModel] = None
_generation_model: Optional[GenerativeModel] = None
_bq_client: Optional[bigquery.Client] = None


def _ensure_initialized() -> None:
    global _vertexai_inited, _embedding_model, _generation_model, _bq_client
    if not PROJECT_ID:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT must be set in environment")
    if not _vertexai_inited:
        vertexai_init(project=PROJECT_ID, location=LOCATION)
        _vertexai_inited = True
    if _embedding_model is None:
        _embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    if _generation_model is None:
        _generation_model = GenerativeModel(GENERATION_MODEL_NAME)
    if _bq_client is None:
        _bq_client = bigquery.Client(project=PROJECT_ID)


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    _ensure_initialized()
    embeddings = _embedding_model.get_embeddings(texts)  # type: ignore[arg-type]
    return [e.values for e in embeddings]


def search_bigquery(question: str, top_k: int | None = None) -> List[Dict[str, Any]]:
    if top_k is None:
        top_k = DEFAULT_TOP_K

    # 1) Embed the query
    query_vec = embed_texts([question])[0]

    # 2) Vector search using BigQuery VECTOR_SEARCH table function
    table_fqn = f"`{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`"
    index_fqn = f"`{PROJECT_ID}.{BQ_DATASET}.{BQ_VECTOR_INDEX}`"
    sql = f"""
    SELECT id, text, distance
    FROM VECTOR_SEARCH(
      TABLE {table_fqn},
      INDEX {index_fqn},
      query_vector => @qvec,
      top_k => @k
    )
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("qvec", "FLOAT64", query_vec),
            bigquery.ScalarQueryParameter("k", "INT64", top_k),
        ]
    )

    _ensure_initialized()
    try:
        rows = list(_bq_client.query(sql, job_config=job_config).result())  # type: ignore[union-attr]
    except Exception as exc:
        logger.error("BigQuery VECTOR_SEARCH failed: %s", exc)
        raise

    results: List[Dict[str, Any]] = []
    for row in rows:
        results.append({
            "id": row.get("id"),
            "text": row.get("text"),
            "distance": float(row.get("distance")) if row.get("distance") is not None else None,
        })
    return results


def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    context_block = "\n\n".join([f"[Doc {i+1} | id={c.get('id')}]\n{c.get('text')}" for i, c in enumerate(contexts)])
    prompt = (
        "You are a helpful assistant. Use the provided documents to answer the user's question. "
        "If the answer cannot be found in the documents, say you don't know.\n\n"
        f"Question:\n{question}\n\n"
        f"Documents:\n{context_block}\n\n"
        "Answer with citations like [Doc N] where appropriate."
    )
    return prompt


def generate_answer(question: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    _ensure_initialized()
    prompt = build_prompt(question, contexts)
    response = _generation_model.generate_content(prompt)  # type: ignore[union-attr]
    answer_text = response.text or ""

    return {
        "answer": answer_text.strip(),
        "sources": [
            {"id": c.get("id"), "text": c.get("text"), "distance": c.get("distance")}
            for c in contexts
        ],
    }


def answer_question(question: str, top_k: int | None = None) -> Dict[str, Any]:
    contexts = search_bigquery(question, top_k=top_k)
    return generate_answer(question, contexts)
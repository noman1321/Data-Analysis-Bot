import os
import uuid
import math
import csv
from typing import List, Dict, Any

from dotenv import load_dotenv
from google.cloud import bigquery

from vertexai import init as vertexai_init
from vertexai.language_models import TextEmbeddingModel

load_dotenv(override=True)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
BQ_DATASET = os.getenv("BQ_DATASET", "rag_demo")
BQ_TABLE = os.getenv("BQ_TABLE", "docs")
CSV_PATH = os.getenv("CSV_PATH")
CSV_TEXT_COLUMN = os.getenv("CSV_TEXT_COLUMN", "text")
CSV_ID_COLUMN = os.getenv("CSV_ID_COLUMN")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))

if not PROJECT_ID:
    raise RuntimeError("GOOGLE_CLOUD_PROJECT must be set in environment")
if not CSV_PATH:
    raise RuntimeError("CSV_PATH must be set in environment")

vertexai_init(project=PROJECT_ID, location=LOCATION)
embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)

bq_client = bigquery.Client(project=PROJECT_ID)


def ensure_dataset_and_table() -> None:
    dataset_ref = bigquery.Dataset(f"{PROJECT_ID}.{BQ_DATASET}")
    try:
        bq_client.get_dataset(dataset_ref)
    except Exception:
        bq_client.create_dataset(dataset_ref, exists_ok=True)

    table_ref = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
    schema = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("text", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("emb", "FLOAT", mode="REPEATED"),  # ARRAY<FLOAT64>
    ]
    table = bigquery.Table(table_ref, schema=schema)
    try:
        bq_client.create_table(table)
        print(f"Created table {table_ref}")
    except Exception:
        # Already exists
        pass


def create_vector_index() -> None:
    table_fqn = f"`{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`"
    index_name = f"idx_{BQ_TABLE}_emb"
    index_fqn = f"`{PROJECT_ID}.{BQ_DATASET}.{index_name}`"
    # Attempt to create vector index (ignore if exists)
    sql = f"""
    CREATE VECTOR INDEX {index_fqn}
    ON {table_fqn}(emb)
    OPTIONS(distance_type = "COSINE");
    """
    try:
        bq_client.query(sql).result()
        print(f"Created vector index {index_name}")
    except Exception as exc:
        # Likely already exists or feature not enabled
        print(f"Vector index creation skipped or failed: {exc}")


def embed_batch(texts: List[str]) -> List[List[float]]:
    results = embedding_model.get_embeddings(texts)
    return [r.values for r in results]


def read_csv_rows(csv_path: str, text_col: str, id_col: str | None) -> tuple[List[str], List[str]]:
    texts: List[str] = []
    ids: List[str] = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if text_col not in reader.fieldnames:
            raise RuntimeError(
                f"CSV_TEXT_COLUMN '{text_col}' not found. Columns: {reader.fieldnames}"
            )
        for row in reader:
            raw_text = row.get(text_col) or ""
            text_value = str(raw_text).strip()
            if not text_value:
                continue
            texts.append(text_value)
            if id_col and id_col in (reader.fieldnames or []) and row.get(id_col):
                ids.append(str(row.get(id_col)))
            else:
                ids.append(str(uuid.uuid4()))
    return ids, texts


def main() -> None:
    ensure_dataset_and_table()

    ids, texts = read_csv_rows(CSV_PATH, CSV_TEXT_COLUMN, CSV_ID_COLUMN)

    if not texts:
        print("No non-empty text rows found to ingest")
        return

    # Embed in batches
    all_embeddings: List[List[float]] = []
    num_batches = math.ceil(len(texts) / BATCH_SIZE)
    for b in range(num_batches):
        start = b * BATCH_SIZE
        end = min((b + 1) * BATCH_SIZE, len(texts))
        batch_texts = texts[start:end]
        batch_embs = embed_batch(batch_texts)
        all_embeddings.extend(batch_embs)
        print(f"Embedded batch {b+1}/{num_batches} ({len(batch_texts)} rows)")

    # Insert into BigQuery
    table_ref = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
    rows_to_insert: List[Dict[str, Any]] = []

    for i in range(len(texts)):
        rows_to_insert.append({
            "id": ids[i],
            "text": texts[i],
            "emb": all_embeddings[i],
        })

    # Use streaming inserts via insert_rows_json
    errors = bq_client.insert_rows_json(table_ref, rows_to_insert)
    if errors:
        raise RuntimeError(f"BigQuery insert had errors: {errors}")
    print(f"Inserted {len(rows_to_insert)} rows into {table_ref}")

    # Create vector index
    create_vector_index()


if __name__ == "__main__":
    main()
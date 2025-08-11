# GCP RAG Chatbot (BigQuery Vector + Vertex AI)

This project ingests a CSV into BigQuery with embeddings, creates a vector index, and exposes a FastAPI chatbot that answers questions using Vertex AI Gemini with retrieval-augmented generation (RAG).

## Architecture
- Ingest CSV rows, embed text with Vertex AI Text Embeddings, store in BigQuery with a vector column.
- Create a BigQuery Vector Index for fast nearest-neighbor search.
- At query time: embed the question → vector search in BigQuery → send context to Gemini → return answer with citations.

## Prerequisites
- gcloud authenticated with Application Default Credentials
- Roles: BigQuery Admin, Vertex AI User (or equivalent fine-grained roles)
- Python 3.10+

## Setup
1. Clone/open this workspace.
2. Create and edit your `.env` based on `.env.example`.
3. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
   If your system is PEP 668 managed, use:
   ```bash
   python3 -m pip install --break-system-packages -r requirements.txt
   ```
4. Enable required APIs (only once per project):
   ```bash
   gcloud services enable bigquery.googleapis.com aiplatform.googleapis.com
   ```

## Ingest CSV and Build Vector Index
Edit `.env` with:
- `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`
- `BQ_DATASET`, `BQ_TABLE` (optional `BQ_VECTOR_INDEX`)
- `CSV_PATH`, `CSV_TEXT_COLUMN` (and optionally `CSV_ID_COLUMN` if you have one)
- Optional models: `EMBEDDING_MODEL`, `GENERATION_MODEL`

Run ingestion:
```bash
python scripts/ingest_csv_to_bq.py
```
This will:
- Create dataset/table if they do not exist
- Compute embeddings in batches
- Insert rows with `id`, `text`, `emb`
- Create a BigQuery vector index

## Run the Chat API
```bash
uvicorn rag_app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8080}
```

- POST `/ask` with JSON `{ "question": "..." }`
- Optional query parameter `k` to override `TOP_K`

Example:
```bash
curl -s -X POST "http://localhost:8080/ask?k=5" \
  -H 'Content-Type: application/json' \
  -d '{"question": "What does the data say about X?"}' | jq .
```

## Notes
- Ensure BigQuery Vector is available in your region. The code uses `VECTOR_SEARCH` table function and `CREATE VECTOR INDEX`.
- For large CSVs, consider streaming to BigQuery then running a separate batch to embed rows incrementally.

## Cleanup
- Delete the dataset to remove all data and the vector index:
  ```sql
  DROP INDEX IF EXISTS `your-project`.`your_dataset`.idx_docs_emb;
  DROP TABLE IF EXISTS `your-project`.`your_dataset`.`your_table`;
  DROP SCHEMA IF EXISTS `your-project`.`your_dataset`;
  ```
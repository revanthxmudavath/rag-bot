# RAG Bot

## Docker Commands

- Build image: `docker build -t rag-bot .`
- Run FastAPI API: `docker run --rm -p 8000:8000 --env-file .env rag-bot`
- Run Discord bot: `docker run --rm --env-file .env -e RUN_MODE=bot rag-bot`

## Uploading Documents / Files

Example ingestion request (replace placeholders):

```bash
curl -X POST "http://localhost:8000/api/ingest" \
  -H "Content-Type: application/json" \
  -d '{
        "documents": [
          {
            "title": "Getting Started Guide",
            "content": "Your document text here...",
            "metadata": {"source": "manual"}
          }
        ],
        "chunk_size": 500,
        "chunk_overlap": 50,
        "replace_existing": true
      }'
```


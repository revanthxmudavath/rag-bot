# RAG Bot

AI Community Assistant by PM Accelerator - A Discord bot powered by RAG (Retrieval-Augmented Generation) for intelligent community question answering.

## Run Commands

### Running the Application

- **Run FastAPI API only:** `python run.py` (or `$env:RUN_MODE="api"; python run.py`)
- **Run Discord bot only:** `$env:RUN_MODE="bot"; python run.py`
- **Run both API and Bot:** `$env:RUN_MODE="both"; python run.py`

The API will be available at `http://localhost:8000` (configurable in `.env`)

## Uploading Documents / Files

### Ingesting Documents to Knowledge Base

- **Using bash script:** `./ingest.sh prompts.txt`
- **Using curl:**
```bash
curl -X POST "http://localhost:8000/api/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "title": "Your Document Title",
        "content": "Your document content here...",
        "metadata": {"source": "manual"}
      }
    ],
    "chunk_size": 500,
    "chunk_overlap": 50,
    "replace_existing": true
  }'
```

## Discord Bot Commands

Once the Discord bot is running, use these commands in your Discord server:

### Main Commands

| Command | Description | Usage |
|---------|-------------|-------|
| `!ask <question>` | Ask any AI/ML question using RAG pipeline | `!ask What is machine learning?` |
| `!community` | Show bot overview and help | `!community` |
| `!info` | Display PM Accelerator AI Community information | `!info` |
| `!help` | List all available commands | `!help` |

### Resource Commands

| Command | Description | Usage |
|---------|-------------|-------|
| `!resources` | Get AI/ML learning resources and materials | `!resources` |
| `!projects` | List available AI/ML community projects | `!projects` |

### Utility Commands

| Command | Description | Usage |
|---------|-------------|-------|
| `!status` | Show bot health and performance metrics | `!status` |

### Interactive Features

- **Feedback System:** React to bot answers with 👍 (helpful), 👎 (not helpful), or 🤔 (need clarification)
- **Auto-responses:** The bot automatically suggests help when AI/ML keywords are mentioned
- **@Mentions:** Tag the bot for quick assistance and command suggestions

## API Endpoints

- **Health Check:** `GET /health`
- **RAG Query:** `POST /api/rag-query`
- **Ingest Documents:** `POST /api/ingest`
- **Feedback:** `POST /api/feedback`
- **Metrics:** `GET /api/metrics`
- **API Documentation:** `http://localhost:8000/docs` (when API is running)

## Configuration

Edit the `.env` file to configure:
- Discord bot token
- MongoDB connection URI
- Azure OpenAI API credentials
- Server host/port
- RAG parameters (chunk size, embedding model, etc.)


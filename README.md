# Local-First Multi-Agent GenAI Platform

Production-style multi-agent AI system with:

- FastAPI gateway
- LangGraph orchestration
- RAG retrieval pipeline
- Tool execution framework (document/sql/python/web)
- Critic agent for validation
- Structured tracing, logs, metrics, and eval runner

## 1) Quick Start (Windows PowerShell)

```powershell
cd e:\multi-agent-genai
C:/Python313/python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python data/init_db.py
python -m uvicorn api.main:app --reload --port 8000
```

Open UI in browser:

```powershell
start http://127.0.0.1:8000/ui
```

Use UI flow:

1. In **Add Document**, paste text (or upload `.txt`/`.md`) and click **Ingest Document**.
2. In **Ask Agent**, enter your prompt and click **Run Query**.
3. Read the friendly **Answer** and **Summary** panels; expand **Raw JSON** only when debugging.

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

Compatibility health route:

```powershell
curl http://127.0.0.1:8000/api/health
```

Query:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/query" -Method Post -ContentType "application/json" -Body '{"user_id":"123","query":"Summarize this document and extract action items","top_k":4}'
```

Compatibility query route also works:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/query" -Method Post -ContentType "application/json" -Body '{"user_id":"123","query":"Summarize this document and extract action items","top_k":4}'
```

Tip:

- `http://127.0.0.1:8000` returns service JSON metadata.
- `http://127.0.0.1:8000/ui` is the interactive web UI.
- UI sends data to `/ingest` and `/query` on the same server origin.

## 2) Project Structure

```text
project/
├── api/
├── agents/
├── tools/
├── rag/
├── eval/
├── data/
├── logs/
├── docker/
└── frontend/
```

## 3) Design Notes

- Local-first LLM via Ollama (`OLLAMA_BASE_URL`, `OLLAMA_MODEL`)
- Graceful fallback behavior when Ollama is unavailable
- SQLite for structured data and local persistence
- Local JSON vector store for zero-cost RAG
- Critic node validates quality and triggers one retry path

## 4) Testing & Eval

Run tests:

```powershell
pytest -q
```

Run evaluation dataset:

```powershell
python eval/run_eval.py
```

Run no-server demo:

```powershell
python run_demo.py
```

# TalentMatch AI — Intelligent Recruitment & Candidate Matching Platform
> This file was written with the assistance of Gemini AI.

**Target Hardware:** 8GB RAM, i5 6th Gen, CPU-Only

---

## 1. System Architecture

```
+------------------------------------------------------------------------------------+
|                                  Frontend (Web UI)                                 |
|   - Recruitment Workspace: semantic ranking, RAG evidence, evaluation scorecards   |
|   - Candidate Portal: resume upload, profile editor, matched jobs, tips            |
+----------------------------+-------------------------------------------------------+
                             | HTTP / REST (FastAPI)
                             v
+------------------------------------------------------------------------------------+
|                               FastAPI Async Backend                                |
|   - JWT Auth & RBAC (Recruiter / Candidate roles)                                  |
|   - Async BackgroundTasks for CV parsing and vector indexing                       |
|   - Structured rotating JSON logger                                                |
+------------------+-----------------------------+-----------------------------------+
                   |                             |
                   v                             v
+-------------------------------+  +---------------------------------------------+
|   SQLModel + SQLite           |  |   Qdrant In-Memory Vector Store             |
|   - Users, Jobs, Candidates   |  |   - 384-dim dense vectors (MiniLM)          |
|   - CVChunk mappings          |  |   - Cosine similarity retrieval             |
+-------------------------------+  +---------------------------------------------+
                                              |
                                              v
+------------------------------------------------------------------------------------+
|                          LangGraph Multi-Agent Pipeline                            |
|  1. ParserAgent:   PyMuPDF + spaCy NER + regex skill extraction                   |
|  2. EmbedderAgent: LangChain chunker + all-MiniLM-L6-v2 embeddings               |
|  3. MatcherAgent:  0.7*max_sim + 0.3*mean_sim + skill bonus (capped at 95%)       |
|  4. AnalystAgent:  Grok (xAI) / HuggingFace SLM with JSON output guardrails       |
+------------------------------------------------------------------------------------+
```

---

## 2. Memory Budget (8GB RAM)

| Component | RAM Usage |
| :--- | :--- |
| OS + Browser | ~2.5 GB |
| FastAPI + Python | ~400 MB |
| Qdrant In-Memory | ~200 MB |
| MiniLM Embeddings | ~100 MB |
| spaCy (en_core_web_sm) | ~50 MB |
| Grok API (optional) | ~0 MB |
| HuggingFace SLM (fallback) | ~300 MB |
| **Headroom** | **~4+ GB** |

---

## 3. Tech Stack

- **Backend:** FastAPI, Uvicorn, SQLModel, SQLite, Qdrant (in-memory), LangGraph
- **NLP & Parsing:** PyMuPDF, spaCy, sentence-transformers (all-MiniLM-L6-v2), langchain-text-splitters
- **LLM:** Grok (xAI API) with automatic fallback to HuggingFace SmolLM2-135M in-process
- **Auth:** bcrypt, python-jose (JWT — 15min access / 7-day refresh)
- **Testing:** pytest, lightweight RAGAS benchmark suite
- **Frontend:** Vanilla HTML/CSS/JS — built with the assistance of Gemini AI

---

## 4. API Endpoints

### Authentication
- `POST /api/auth/register` — Create candidate or recruiter account
- `POST /api/auth/login` — Authenticate and receive JWT tokens
- `POST /api/auth/refresh` — Refresh access token
- `GET  /api/auth/me` — Get current authenticated user

### Jobs
- `GET  /api/jobs` — List all job requisitions
- `GET  /api/jobs/{job_id}` — Get a single job
- `POST /api/jobs` — Create a new job requisition
- `GET  /api/jobs/{job_id}/matches` — Rank candidates for a job semantically

### Candidates
- `GET  /api/candidates` — List all candidates
- `GET  /api/candidates/{id}` — Get candidate profile
- `POST /api/candidates` — Create or update candidate profile
- `POST /api/candidates/upload-cv` — Upload PDF/TXT resume (202 Accepted, async)
- `GET  /api/candidates/{id}/analysis/{job_id}` — Generate AI evaluation scorecard
- `POST /api/candidates/{id}/resume-tips` — Generate resume improvement suggestions
- `POST /api/match/candidate/{id}` — Rank jobs for a specific candidate
- `DELETE /api/candidates/{id}` — Delete candidate

### System
- `GET  /api/health` — Database, vector store, and LLM status
- `POST /api/admin/reset-db` — Reset to original demo dataset

---

## 5. Running Locally

### 1. Install Dependencies
```bash
pip install -r backend/requirements.txt
python -m spacy download en_core_web_sm
```

### 2. (Optional) Configure Grok API Key
```bash
GROK_API_KEY=xai-your-key-here
GROK_MODEL=grok-2-latest
```
Without a Grok key, the app runs fully offline using the built-in HuggingFace SLM.

### 3. Start the Application
```bash
python run.py
```
- Web UI: `http://127.0.0.1:8000`
- API Docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/api/health`

---

## 6. Tests & Benchmarks

```bash
# Run test suite
python -m pytest backend/tests -v

# Run benchmark evaluation
python benchmark/evaluate.py
```

---

## 7. License
MIT — free to use, modify, and deploy.

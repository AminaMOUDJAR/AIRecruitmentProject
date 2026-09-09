---
title: TalentMatch AI
emoji: 💼
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# TalentMatch AI — Intelligent Recruitment & Candidate Matching Platform

A recruiting tool that matches candidates to jobs using dense vector embeddings instead of simple keyword filtering. Built with FastAPI + PyTorch on the backend and a modern glassmorphic vanilla JS frontend.

---

## 🌟 Overview & Key Features

The application consists of two integrated portals:

1. **Recruiter Hub** — Post job requisitions, rank candidates by semantic similarity (0–100% match scores), inspect retrieved CV evidence snippets, and generate custom technical interview scorecards based on resume gaps.
2. **Job Seeker Portal** — Upload a CV (PDF or TXT), edit applicant profile, see matched openings, and receive AI resume optimization feedback.

---

## 🏛️ 1. System Architecture

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
|  4. AnalystAgent:  Grok (xAI) / Groq / HuggingFace SLM with JSON guardrails        |
+------------------------------------------------------------------------------------+
```

---

## ⚡ 2. Memory Budget (Target: 8GB RAM, CPU-Only)

| Component | RAM Usage |
| :--- | :--- |
| OS + Browser | ~2.5 GB |
| FastAPI + Python | ~400 MB |
| Qdrant In-Memory | ~200 MB |
| MiniLM Embeddings | ~100 MB |
| spaCy (en_core_web_sm) | ~50 MB |
| Grok / Groq Cloud API | ~0 MB |
| HuggingFace SLM (local fallback) | ~300 MB |
| **Headroom** | **~4+ GB** |

---

## 🛠️ 3. Tech Stack

- **Backend:** FastAPI, Uvicorn, SQLModel, SQLite, Qdrant (in-memory), LangGraph
- **NLP & Parsing:** PyMuPDF, spaCy, sentence-transformers (`all-MiniLM-L6-v2`), langchain-text-splitters
- **LLM Reasoning:** Grok (xAI API) & Groq (`llama-3.3-70b-versatile`) with automatic fallback to local HuggingFace SmolLM2-135M
- **Authentication:** bcrypt, python-jose (JWT — 15min access / 7-day refresh)
- **Frontend:** Vanilla HTML5, modern CSS3 (glassmorphic aesthetic), Vanilla JavaScript
- **Testing:** pytest, benchmark evaluation suite

---

## 📡 4. API Endpoints

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

### System & Health
- `GET  /api/health` — Database, vector store, and LLM status
- `POST /api/admin/reset-db` — Reset to original demo dataset

---

## 🚀 5. Running Locally

### 1. Install Dependencies
```bash
pip install -r backend/requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure Environment (Optional for Cloud LLM)
Create a `.env` file in the root directory:
```env
GROK_API_KEY=your_key_here
LLM_PROVIDER=auto
```
*Note: If no API key is provided, the system automatically uses the local HuggingFace SLM.*

### 3. Start the Server
```bash
python run.py
```
- **Web UI:** `http://127.0.0.1:8000`
- **API Docs:** `http://127.0.0.1:8000/docs`
- **Health Check:** `http://127.0.0.1:8000/api/health`

---

## 🧪 6. Tests & Benchmarks

```bash
# Run test suite
python -m pytest backend/tests -v

# Run benchmark evaluation
python benchmark/evaluate.py
```

---

## 📄 7. License
MIT — Free to use, modify, and deploy.

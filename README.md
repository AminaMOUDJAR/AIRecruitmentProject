# TalentMatch AI — Semantic Resume Matching Platform

A recruiting tool that matches candidates to jobs using dense vector embeddings instead of simple keyword filtering. Built with FastAPI + PyTorch on the backend and a vanilla JS frontend.

## What it does

The app has two sides:

**Recruiter Hub** — Post job requisitions, rank candidates by semantic similarity (0–100% match scores), inspect retrieved CV evidence snippets, and generate custom technical interview questions based on resume gaps.

**Job Seeker Portal** — Upload a CV (PDF or TXT), edit your profile, see matched openings, and get resume optimization feedback.

## Tech stack

- **Frontend:** Vanilla HTML/JS (I used AI to plan the UI structure and write most of the CSS — I tweaked the layout and colors afterward)
- **Backend:** FastAPI, PyMuPDF for CV parsing
- **Embeddings:** sentence-transformers (MiniLM) via HuggingFace + PyTorch
- **Vector search:** Dense cosine similarity, sub-50ms retrieval
- **Reasoning models:** SmolLM2 and Qwen2.5 via the Transformers library for match rationale, skill gap extraction, and interview question generation

The SLM reasoning runs locally — no API keys needed.

## Architecture
Frontend (vanilla JS)  <--REST-->  FastAPI backend
|
+-----------------+-----------------+
|                                   |
Vector Engine                         Reasoning Engine
(MiniLM embeddings,                  (SmolLM2 / Qwen2.5,
cosine similarity,                    Transformers pipeline)
LangChain chunking)


## Quickstart

```bash
cd project
pip install -r backend/requirements.txt
python run.py
```
Opens at http://127.0.0.1:8000.
Notes
I used AI tools to help plan the overall project structure and to generate the CSS. The backend logic, data handling, and integration were written by me.
The reasoning models are small enough to run on CPU but work best with a GPU if you have one.
License
MIT

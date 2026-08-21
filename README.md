# TalentMatch AI — Smart AI Recruiting & Semantic Matching Platform

## Project Overview

**TalentMatch AI** is a AI recruiting and resume matching platform that replaces rigid keyword filters with **dense semantic vector matching** and **Small Language Model reasoning**.

The platform provides a dual-persona interface:
1. **Recruiter Hub**: Post job requisitions, semantically rank candidates (with realistic 0–100% match scores), inspect RAG-retrieved CV evidence snippets, and automatically generate **custom technical interview questions** targeting candidate resume gaps.
2. **Job Seeker Portal**: Upload CVs (PDF / TXT), fine-tune skills and work history, view matched jobs across the organization, and receive **SLM-powered resume optimization tips**.

---

## Architecture & AI Pipeline

```
+-----------------------------------------------------------------------------------+
|                           Frontend                                                |
|  - Dual Role Switcher: Recruiter Hub & Job Seeker Portal                          |
|  - Radial Score Meters, Skill Radar Tags, Dynamic Interview Question Viewer       |
+------------------------------------------+----------------------------------------+
                                           | REST API
                                           v
+-----------------------------------------------------------------------------------+
|                           FastAPI High-Performance Backend                        |
|  - Modular endpoints: /api/jobs, /api/candidates, /api/match, /api/ai/analyze     |
|  - PyMuPDF PDF parser for instant CV text ingestion                               |
+--------------------+-------------------------------------+------------------------+
                     |                                     |
                     v                                     v
+------------------------------------+  +-------------------------------------------+
|    RAG & Vector Semantic Engine    |  |            SLM Reasoning Engine           |
| - PyTorch & HuggingFace Embeddings |  | - Small Language Model (SLM) Architecture |
|   (sentence-transformers MiniLM)   |  |   (SmolLM2 / Qwen2.5 / Transformers)      |
| - LangChain Document Chunking      |  | - Match Rationale & Skill Gap Extraction  |
| - Dense Cosine Vector Similarity   |  | - Dynamic Gap-Targeted Interview Qs       |
| - Sub-50ms Candidate Retrieval     |  | - Actionable Candidate CV Optimizer       |
+------------------------------------+  +-------------------------------------------+
```


## 🏁 Quickstart & How to Run

### 1. Clone & Install Dependencies
```bash
# Navigate to project folder
cd project

# Install requirements
pip install -r backend/requirements.txt
```

### 2. Launch the Application
```bash
python run.py
```
*The app will start at `http://127.0.0.1:8000` and automatically open in your default web browser.*



## 📄 License
MIT License. Free to use, modify, and showcase in personal portfolios and interviews.

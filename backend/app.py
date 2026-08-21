import os
import io
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import fitz  # PyMuPDF

from backend.data.storage import store
from backend.ai.rag_engine import rag_engine
from backend.ai.slm_reasoner import slm_reasoner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TalentMatchAPI")

app = FastAPI(
    title="TalentMatch AI API",
    description="Smart AI Recruiting & Resume Matching Platform powered by SLMs, RAG, PyTorch & LangChain",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Pydantic Models -----------------

class JobCreateRequest(BaseModel):
    title: str = Field(..., example="Senior Generative AI Engineer")
    company: str = Field(..., example="Apex AI Labs")
    location: str = Field(..., example="San Francisco, CA (Remote)")
    type: str = Field("Full-Time", example="Full-Time")
    experience_level: str = Field("Senior (4+ years)", example="Senior (4+ years)")
    description: str = Field(..., example="Build RAG pipelines and deploy Small Language Models...")
    required_skills: List[str] = Field(default_factory=list, example=["Python", "PyTorch", "Hugging Face", "LangChain"])
    nice_to_have_skills: List[str] = Field(default_factory=list, example=["Docker", "vLLM"])
    min_experience_years: int = Field(3, example=3)

class ExperienceItem(BaseModel):
    role: str
    company: str
    duration: str
    description: str

class EducationItem(BaseModel):
    degree: str
    institution: str
    year: str

class CandidateProfileRequest(BaseModel):
    id: Optional[str] = None
    name: str
    title: str
    email: str
    location: str = "Remote"
    years_experience: int = 3
    skills: List[str] = Field(default_factory=list)
    bio: str = ""
    experience: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    raw_cv_text: Optional[str] = ""

class CandidateAnalysisRequest(BaseModel):
    candidate_id: str
    job_id: str


# ----------------- API Routes -----------------

@app.get("/api/health")
def health_check():
    return {"status": "ok", "system": "TalentMatch AI Platform", "version": "1.0.0"}

# --- Jobs ---
@app.get("/api/jobs")
def get_jobs():
    return store.get_all_jobs()

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = store.get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/api/jobs")
def create_job(job: JobCreateRequest):
    data = job.model_dump()
    created = store.add_job(data)
    return {"message": "Job posted successfully", "job": created}

# --- Candidates ---
@app.get("/api/candidates")
def get_candidates():
    return store.get_all_candidates()

@app.get("/api/candidates/{cand_id}")
def get_candidate(cand_id: str):
    cand = store.get_candidate_by_id(cand_id)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return cand

@app.post("/api/candidates")
def save_candidate_profile(cand: CandidateProfileRequest):
    data = cand.model_dump()
    saved = store.upsert_candidate(data)
    return {"message": "Candidate profile saved successfully", "candidate": saved}

@app.delete("/api/candidates/{cand_id}")
def delete_candidate(cand_id: str):
    success = store.delete_candidate(cand_id)
    if not success:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return {"message": "Candidate deleted"}

# --- CV File / Text Upload & Parser ---
@app.post("/api/candidates/upload-cv")
async def upload_cv(
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None),
    candidate_name: Optional[str] = Form("Job Seeker"),
    job_title: Optional[str] = Form("Software Engineer"),
    email: Optional[str] = Form("seeker@example.com")
):
    extracted_text = ""
    if file:
        content = await file.read()
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            try:
                pdf_doc = fitz.open(stream=content, filetype="pdf")
                pages = [page.get_text() for page in pdf_doc]
                extracted_text = "\n".join(pages)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")
        else:
            try:
                extracted_text = content.decode("utf-8")
            except UnicodeDecodeError:
                extracted_text = content.decode("latin-1", errors="ignore")
    elif raw_text:
        extracted_text = raw_text.strip()
    else:
        raise HTTPException(status_code=400, detail="Please provide a CV file or raw resume text.")

    # Simple heuristic skill & experience extractor to auto-populate profile fields
    common_skills = [
        "Python", "PyTorch", "TensorFlow", "Hugging Face", "LangChain", "RAG",
        "FastAPI", "Docker", "Kubernetes", "AWS", "GCP", "SQL", "PostgreSQL",
        "React", "JavaScript", "TypeScript", "Node.js", "MLOps", "vLLM",
        "NLP", "Scikit-Learn", "Pandas", "Git", "CI/CD", "Linux"
    ]
    detected_skills = [s for s in common_skills if s.lower() in extracted_text.lower()]
    if not detected_skills:
        detected_skills = ["Python", "FastAPI", "Machine Learning"]

    new_candidate = {
        "name": candidate_name or "Uploaded Candidate",
        "title": job_title or "AI Specialist",
        "email": email or "candidate@example.com",
        "location": "Remote",
        "years_experience": 3,
        "skills": detected_skills,
        "bio": f"Automated profile imported from CV. Key focus: {', '.join(detected_skills[:4])}.",
        "experience": [
            {
                "role": job_title or "Engineer",
                "company": "Recent Organization",
                "duration": "2021 - Present",
                "description": "Experience extracted from uploaded resume document."
            }
        ],
        "education": [
            {
                "degree": "B.S. in Computer Science / Related Field",
                "institution": "University",
                "year": "2021"
            }
        ],
        "raw_cv_text": extracted_text
    }

    saved = store.upsert_candidate(new_candidate)
    return {
        "message": "CV parsed and imported successfully!",
        "candidate": saved,
        "detected_skills": detected_skills,
        "extracted_text_preview": extracted_text[:400] + ("..." if len(extracted_text) > 400 else "")
    }

# --- Smart AI Semantic Matching ---
@app.post("/api/match/job/{job_id}")
def match_candidates_for_job(job_id: str):
    job = store.get_job_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    candidates = store.get_all_candidates()
    ranked_results = rag_engine.match_candidates_for_job(job, candidates)
    return {
        "job": job,
        "total_candidates": len(candidates),
        "matches": ranked_results
    }

@app.post("/api/match/candidate/{cand_id}")
def match_jobs_for_candidate(cand_id: str):
    candidate = store.get_candidate_by_id(cand_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    jobs = store.get_all_jobs()
    ranked_jobs = rag_engine.match_jobs_for_candidate(candidate, jobs)
    return {
        "candidate": candidate,
        "total_jobs": len(jobs),
        "matched_jobs": ranked_jobs
    }

# --- SLM Reasoning & Dynamic Analysis ---
@app.post("/api/ai/candidate-analysis")
def generate_deep_candidate_analysis(req: CandidateAnalysisRequest):
    candidate = store.get_candidate_by_id(req.candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    job = store.get_job_by_id(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Match and retrieve RAG evidence
    matches = rag_engine.match_candidates_for_job(job, [candidate])
    match_data = matches[0] if matches else {
        "match_score": 75.0,
        "matched_skills": [],
        "missing_skills": [],
        "rag_evidence": []
    }

    analysis = slm_reasoner.generate_candidate_analysis(
        candidate=candidate,
        job=job,
        match_score=match_data["match_score"],
        matched_skills=match_data["matched_skills"],
        missing_skills=match_data["missing_skills"],
        rag_evidence=match_data.get("rag_evidence", [])
    )

    return analysis

@app.post("/api/ai/resume-tips/{cand_id}")
def get_resume_optimization_tips(cand_id: str, target_job_id: Optional[str] = None):
    candidate = store.get_candidate_by_id(cand_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    target_job = None
    if target_job_id:
        target_job = store.get_job_by_id(target_job_id)

    tips = slm_reasoner.generate_resume_optimizer_tips(candidate, target_job)
    return {
        "candidate_id": cand_id,
        "candidate_name": candidate.get("name"),
        "target_job": target_job.get("title") if target_job else "General AI & Tech Roles",
        "tips": tips
    }

# --- System Reset ---
@app.post("/api/reset-data")
def reset_demo_data():
    store.reset_to_defaults()
    return {"message": "Datasets reset to original demo state."}


# ----------------- Static Frontend Hosting -----------------
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "TalentMatch AI Backend Running. Frontend index.html not found."}

import os
import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, status
from pydantic import BaseModel
from sqlmodel import Session, select

from backend.app.database import get_session, engine
from backend.app.models import Candidate, CVChunk, Job, CandidateCreate, AnalystReport
from backend.app.dependencies import validate_uploaded_file
from backend.app.agents.embedder import embedder_agent
from backend.app.agents.matcher import matcher_agent
from backend.app.agents.analyst import analyst_agent
from backend.app.agents.pipeline import process_cv_ingestion
from backend.app.logger import logger

router = APIRouter(prefix="", tags=["Candidates"])

class CandidateAnalysisRequest(BaseModel):
    candidate_id: str
    job_id: str

def _find_candidate(cand_id_str: str, session: Session) -> Optional[Candidate]:
    # Look up candidate by exact UUID or prefix
    try:
        cid = uuid.UUID(cand_id_str)
        cand = session.get(Candidate, cid)
        if cand:
            return cand
    except Exception:
        pass

    all_cands = session.exec(select(Candidate)).all()
    for c in all_cands:
        if str(c.id) == cand_id_str or str(c.id).startswith(cand_id_str):
            return c
    return None

def _find_job(job_id_str: str, session: Session) -> Optional[Job]:
    # Look up job by exact UUID or prefix
    try:
        jid = uuid.UUID(job_id_str)
        job = session.get(Job, jid)
        if job:
            return job
    except Exception:
        pass

    all_jobs = session.exec(select(Job)).all()
    for j in all_jobs:
        if str(j.id) == job_id_str or str(j.id).startswith(job_id_str):
            return j
    return None

def process_cv_pipeline(candidate_id_str: str, raw_bytes: Optional[bytes], raw_text: Optional[str], filename: str):
    # Background worker: extracts text, parses sections, embeds chunks into Qdrant
    logger.info(f"Processing CV for candidate {candidate_id_str}...")
    try:
        # Run the shared LangGraph parser -> embedder ingestion nodes
        ingestion = process_cv_ingestion(
            raw_bytes=raw_bytes,
            raw_text=raw_text,
            filename=filename,
            candidate_id=candidate_id_str
        )
        parsed = ingestion["profile"]

        with Session(engine) as session:
            cand = _find_candidate(candidate_id_str, session)
            if not cand:
                logger.error(f"Candidate {candidate_id_str} not found in database")
                return

            # Update candidate profile fields (parsed values must be
            # non-fabricated to overwrite; empty means "not found")
            cand.name = parsed.get("name") or cand.name
            cand.title = parsed.get("title") or cand.title
            if parsed.get("email"):
                cand.email = parsed["email"]
            cand.skills = parsed.get("skills") or cand.skills
            cand.bio = parsed.get("bio") or cand.bio
            cand.experience = parsed.get("experience") or cand.experience
            cand.education = parsed.get("education") or cand.education
            cand.raw_cv_text = parsed.get("raw_cv_text") or cand.raw_cv_text
            cand.parsed_entities = parsed.get("parsed_entities") or cand.parsed_entities

            # Chunk and index text into Qdrant vector collection
            indexed_records = embedder_agent.chunk_and_index_candidate(str(cand.id), cand.model_dump())

            # Store chunk metadata for quick RAG retrieval
            for record in indexed_records:
                chunk = CVChunk(
                    candidate_id=cand.id,
                    chunk_text=record["chunk_text"],
                    chunk_type=record["chunk_type"],
                    vector_id=record["vector_id"]
                )
                session.add(chunk)

            cand.embedding_status = "ready"
            session.add(cand)
            session.commit()
            logger.info(f"Finished processing and indexing candidate {cand.name}")
    except Exception as e:
        logger.error(f"Error processing CV: {e}")
        with Session(engine) as session:
            cand = _find_candidate(candidate_id_str, session)
            if cand:
                cand.embedding_status = "failed"
                session.add(cand)
                session.commit()

@router.get("/api/candidates")
def get_all_candidates(session: Session = Depends(get_session)):
    candidates = session.exec(select(Candidate)).all()
    return [c.model_dump() for c in candidates]

@router.get("/api/candidates/{cand_id}")
def get_candidate_by_id(cand_id: str, session: Session = Depends(get_session)):
    cand = _find_candidate(cand_id, session)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return cand.model_dump()

@router.post("/api/candidates")
def save_candidate_profile(cand_in: CandidateCreate, session: Session = Depends(get_session)):
    existing = None
    if cand_in.id:
        existing = _find_candidate(cand_in.id, session)

    if not existing and cand_in.email:
        existing = session.exec(select(Candidate).where(Candidate.email == cand_in.email)).first()

    if existing:
        existing.name = cand_in.name
        existing.title = cand_in.title
        existing.location = cand_in.location
        existing.years_experience = cand_in.years_experience
        existing.skills = cand_in.skills
        existing.bio = cand_in.bio
        if cand_in.experience:
            existing.experience = cand_in.experience
        if cand_in.education:
            existing.education = cand_in.education
        if cand_in.raw_cv_text:
            existing.raw_cv_text = cand_in.raw_cv_text

        # Refresh vector embeddings
        embedder_agent.chunk_and_index_candidate(str(existing.id), existing.model_dump())
        existing.embedding_status = "ready"
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return {"message": "Candidate profile updated successfully", "candidate": existing.model_dump()}

    new_cand = Candidate(
        name=cand_in.name,
        title=cand_in.title,
        email=cand_in.email,
        location=cand_in.location,
        years_experience=cand_in.years_experience,
        skills=cand_in.skills,
        bio=cand_in.bio,
        experience=cand_in.experience,
        education=cand_in.education,
        raw_cv_text=cand_in.raw_cv_text,
        embedding_status="ready"
    )
    session.add(new_cand)
    session.commit()
    session.refresh(new_cand)

    embedder_agent.chunk_and_index_candidate(str(new_cand.id), new_cand.model_dump())

    return {"message": "Candidate profile created successfully", "candidate": new_cand.model_dump()}

@router.post("/api/candidates/upload-cv", status_code=status.HTTP_202_ACCEPTED)
async def upload_cv(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None),
    candidate_name: Optional[str] = Form(None),
    job_title: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    session: Session = Depends(get_session)
):
    raw_bytes = None
    filename = ""

    if file:
        validate_uploaded_file(file)
        raw_bytes = await file.read()
        filename = file.filename or "uploaded_resume.pdf"
    elif not raw_text:
        raise HTTPException(status_code=400, detail="Please provide a resume file or raw text.")

    temp_name = candidate_name or (filename.replace(".pdf", "").replace(".txt", "").replace("_", " ").title() if filename else "Uploaded Candidate")
    new_cand = Candidate(
        name=temp_name,
        title=job_title or "Software Specialist",
        email=email or f"candidate_{uuid.uuid4().hex[:6]}@example.com",
        raw_cv_text=raw_text or "Processing uploaded document...",
        embedding_status="processing"
    )
    session.add(new_cand)
    session.commit()
    session.refresh(new_cand)

    cand_id_str = str(new_cand.id)

    background_tasks.add_task(
        process_cv_pipeline,
        cand_id_str,
        raw_bytes,
        raw_text,
        filename
    )

    return {
        "message": "Resume accepted for processing.",
        "candidate_id": cand_id_str,
        "candidate": new_cand.model_dump(),
        "status": "processing"
    }

@router.get("/api/candidates/{cand_id}/analysis/{job_id}")
def get_candidate_analysis(cand_id: str, job_id: str, session: Session = Depends(get_session)):
    cand = _find_candidate(cand_id, session)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    job = _find_job(job_id, session)
    if not job:
        raise HTTPException(status_code=404, detail="Job requisition not found")

    matches = matcher_agent.match_candidates_for_job(job.model_dump(), [cand.model_dump()])
    match_data = matches[0] if matches else {"match_score": 75.0, "evidence_chunks": []}

    report = analyst_agent.generate_analysis(
        candidate=cand.model_dump(),
        job=job.model_dump(),
        match_score=match_data["match_score"],
        evidence_chunks=match_data.get("evidence_chunks", [])
    )
    return report.model_dump()

@router.post("/api/ai/candidate-analysis")
def post_candidate_analysis(req: CandidateAnalysisRequest, session: Session = Depends(get_session)):
    return get_candidate_analysis(req.candidate_id, req.job_id, session)

@router.post("/api/candidates/{cand_id}/resume-tips")
@router.post("/api/ai/resume-tips/{cand_id}")
def get_candidate_resume_tips(cand_id: str, target_job_id: Optional[str] = None, session: Session = Depends(get_session)):
    cand = _find_candidate(cand_id, session)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found")

    target_job = None
    if target_job_id:
        target_job = _find_job(target_job_id, session)

    tips = analyst_agent.generate_resume_tips(
        cand.model_dump(),
        target_job.model_dump() if target_job else None
    )

    return {
        "candidate_id": str(cand.id),
        "candidate_name": cand.name,
        "target_job": target_job.title if target_job else "Tech Role",
        "tips": tips
    }

@router.post("/api/match/candidate/{cand_id}")
def match_jobs_for_candidate(cand_id: str, session: Session = Depends(get_session)):
    cand = _find_candidate(cand_id, session)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found")

    jobs = session.exec(select(Job)).all()
    job_dicts = [j.model_dump() for j in jobs]

    ranked_jobs = matcher_agent.match_jobs_for_candidate(cand.model_dump(), job_dicts)

    return {
        "candidate": cand.model_dump(),
        "total_jobs": len(jobs),
        "matched_jobs": ranked_jobs
    }

@router.delete("/api/candidates/{cand_id}")
def delete_candidate(cand_id: str, session: Session = Depends(get_session)):
    cand = _find_candidate(cand_id, session)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # Remove vector index entries and chunk records alongside the profile
    embedder_agent.delete_candidate_chunks(str(cand.id))
    for chunk in session.exec(select(CVChunk).where(CVChunk.candidate_id == cand.id)).all():
        session.delete(chunk)
    session.delete(cand)
    session.commit()
    return {"message": "Candidate deleted successfully"}

import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlmodel import Session, select

from backend.app.database import get_session
from backend.app.models import Job, JobCreate, Candidate
from backend.app.agents.matcher import matcher_agent
from backend.app.agents.embedder import embedder_agent
from backend.app.logger import logger

router = APIRouter(prefix="", tags=["Jobs"])

def _find_job(job_id_str: str, session: Session) -> Optional[Job]:
    # Look up job by UUID or prefix
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

@router.get("/api/jobs")
def get_all_jobs(session: Session = Depends(get_session)):
    jobs = session.exec(select(Job)).all()
    return [j.model_dump() for j in jobs]

@router.get("/api/jobs/{job_id}")
def get_job_by_id(job_id: str, session: Session = Depends(get_session)):
    job = _find_job(job_id, session)
    if not job:
        raise HTTPException(status_code=404, detail="Job requisition not found")
    return job.model_dump()

@router.post("/api/jobs")
def create_job(job_in: JobCreate, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    new_job = Job(
        title=job_in.title,
        company=job_in.company,
        location=job_in.location,
        type=job_in.type,
        experience_level=job_in.experience_level,
        description=job_in.description,
        required_skills=job_in.required_skills,
        nice_to_have_skills=job_in.nice_to_have_skills,
        min_experience_years=job_in.min_experience_years,
        embedding_status="ready"
    )
    session.add(new_job)
    session.commit()
    session.refresh(new_job)

    # Cache job search vector in the background
    def pre_embed(query_text: str):
        embedder_agent.embed_query(query_text)
    
    query = matcher_agent.build_job_query(new_job.model_dump())
    background_tasks.add_task(pre_embed, query)

    return {"message": "Job requisition created successfully", "job": new_job.model_dump()}

@router.get("/api/jobs/{job_id}/matches")
@router.post("/api/match/job/{job_id}")
def match_candidates_for_job(job_id: str, session: Session = Depends(get_session)):
    job = _find_job(job_id, session)
    if not job:
        raise HTTPException(status_code=404, detail="Job requisition not found")

    candidates = session.exec(select(Candidate)).all()
    cand_dicts = [c.model_dump() for c in candidates]

    # Calculate dense vector fit scores
    ranked_matches = matcher_agent.match_candidates_for_job(job.model_dump(), cand_dicts)

    return {
        "job": job.model_dump(),
        "total_candidates": len(candidates),
        "matches": ranked_matches
    }

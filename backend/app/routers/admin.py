import httpx
import os
from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from backend.app.config import settings
from backend.app.database import get_session, reset_database_data
from backend.app.models import Job, Candidate
from backend.app.agents.embedder import embedder_agent
from backend.app.logger import logger

router = APIRouter(prefix="", tags=["Admin & System"])

@router.get("/api/health")
def health_check(session: Session = Depends(get_session)):
    # Database check
    db_ok = True
    job_count = 0
    candidate_count = 0
    try:
        job_count = len(session.exec(select(Job)).all())
        candidate_count = len(session.exec(select(Candidate)).all())
    except Exception as e:
        db_ok = False
        logger.error(f"Database error during health check: {e}")

    # Vector store status
    vector_ok = True
    vector_count = 0
    try:
        res = embedder_agent.qdrant.get_collection(embedder_agent.collection_name)
        vector_count = res.points_count if hasattr(res, "points_count") else 0
    except Exception:
        vector_ok = False

    # Check which AI model is currently active
    grok_key = getattr(settings, "GROK_API_KEY", None) or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
    if grok_key and grok_key.strip():
        if grok_key.strip().startswith("gsk_"):
            llm_provider = "groq (cloud API)"
            active_model = "llama-3.3-70b-versatile"
        else:
            llm_provider = "grok (xAI API)"
            active_model = getattr(settings, "GROK_MODEL", "grok-2-latest")
        llm_ready = True
    else:
        llm_provider = "huggingface_slm (in-process)"
        active_model = getattr(settings, "SLM_MODEL_NAME", "HuggingFaceTB/SmolLM2-135M-Instruct")
        llm_ready = True

    return {
        "status": "healthy" if db_ok else "degraded",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "database": {
            "status": "connected" if db_ok else "error",
            "type": "SQLite + SQLModel",
            "jobs_count": job_count,
            "candidates_count": candidate_count
        },
        "vector_store": {
            "status": "online" if vector_ok else "error",
            "type": "Qdrant In-Memory",
            "collection": embedder_agent.collection_name,
            "dimension": settings.VECTOR_DIMENSION
        },
        "llm_engine": {
            "status": "online" if llm_ready else "offline",
            "provider": llm_provider,
            "active_model": active_model
        },
        "active_models": [active_model]
    }

@router.post("/api/admin/reset-db")
@router.post("/api/reset-data")
def reset_database():
    # Reseed database tables from dataset files
    reset_database_data()
    
    # Re-embed all candidate profiles into vector memory
    from sqlmodel import Session, select
    from backend.app.database import engine
    with Session(engine) as session:
        candidates = session.exec(select(Candidate)).all()
        for c in candidates:
            embedder_agent.chunk_and_index_candidate(str(c.id), c.model_dump())
            c.embedding_status = "ready"
            session.add(c)
        session.commit()

    return {"message": "Database and vector index successfully reset to clean dataset."}

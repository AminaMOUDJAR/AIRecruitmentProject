import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from backend.app.config import settings
from backend.app.logger import logger
from backend.app.database import init_db, engine
from backend.app.models import Candidate
from backend.app.agents.embedder import embedder_agent
from backend.app.routers import auth, jobs, candidates, admin

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Set up database tables and index existing candidates into vector memory
    logger.info("Starting up TalentMatch AI backend...")
    init_db()
    with Session(engine) as session:
        cands = session.exec(select(Candidate)).all()
        for c in cands:
            embedder_agent.chunk_and_index_candidate(str(c.id), c.model_dump())
            c.embedding_status = "ready"
            session.add(c)
        session.commit()
    logger.info(f"Loaded and indexed {len(cands)} candidates into memory")
    yield
    # Cleanup on exit
    logger.info("TalentMatch AI backend stopped.")

app = FastAPI(
    title=settings.APP_NAME,
    description="Intelligent recruitment platform with semantic candidate matching.",
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# Allow local frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(auth.router)
app.include_router(jobs.router)
app.include_router(candidates.router)
app.include_router(admin.router)

# Serve static frontend files
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend")

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_root():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "TalentMatch AI backend running. Frontend not found at root."}

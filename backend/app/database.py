import os
import json
import uuid
import bcrypt
from typing import Generator
from sqlmodel import SQLModel, create_engine, Session, select
from backend.app.config import settings
from backend.app.models import User, Job, Candidate, CVChunk
from backend.app.logger import logger

os.makedirs(settings.DATA_DIR, exist_ok=True)

connect_args = {"check_same_thread": False}
engine = create_engine(settings.DATABASE_URL, echo=False, connect_args=connect_args)

def _hash_seed_pw(pwd: str) -> str:
    # Hash default passwords using bcrypt
    return bcrypt.hashpw(pwd.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")

def init_db():
    # Create tables if they don't exist yet and populate seed data
    SQLModel.metadata.create_all(engine)
    seed_initial_data()

def get_session() -> Generator[Session, None, None]:
    # FastAPI session dependency
    with Session(engine) as session:
        yield session

def seed_initial_data():
    # Insert default users, jobs, and candidates if the database is clean
    with Session(engine) as session:
        # Create default test accounts if missing
        admin_user = session.exec(select(User).where(User.email == "recruiter@talentmatch.ai")).first()
        if not admin_user:
            recruiter_user = User(
                id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
                email="recruiter@talentmatch.ai",
                role="recruiter",
                hashed_password=_hash_seed_pw("recruiter123")
            )
            candidate_user = User(
                id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
                email="sophie.marchand@example.com",
                role="candidate",
                hashed_password=_hash_seed_pw("candidate123")
            )
            session.add(recruiter_user)
            session.add(candidate_user)
            session.commit()
            logger.info("Created seed accounts: recruiter and candidate")

        # Load Algerian job listings from dataset
        existing_jobs = session.exec(select(Job)).all()
        if not existing_jobs:
            jobs_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "jobs_dataset.json")
            if os.path.exists(jobs_file):
                with open(jobs_file, "r", encoding="utf-8") as f:
                    jobs_data = json.load(f)
                seen_titles = set()
                for j in jobs_data:
                    # Skip duplicate job titles in seed file
                    unique_key = (j.get("title"), j.get("company"))
                    if unique_key in seen_titles:
                        continue
                    seen_titles.add(unique_key)

                    try:
                        jid = uuid.UUID(j.get("id"))
                    except Exception:
                        jid = uuid.uuid5(uuid.NAMESPACE_DNS, j.get("id", str(uuid.uuid4())))
                    
                    job = Job(
                        id=jid,
                        title=j.get("title"),
                        company=j.get("company", "Tech Enterprise"),
                        location=j.get("location", "Algiers, Algeria"),
                        type=j.get("type", "Full-Time"),
                        experience_level=j.get("experience_level", "Mid-Senior (3+ years)"),
                        description=j.get("description", ""),
                        required_skills=j.get("required_skills", []),
                        nice_to_have_skills=j.get("nice_to_have_skills", []),
                        min_experience_years=j.get("min_experience_years", 3),
                        embedding_status="ready"
                    )
                    session.add(job)
                session.commit()
                logger.info(f"Imported {len(seen_titles)} Algerian jobs into database")

        # Load candidate profiles from dataset
        existing_candidates = session.exec(select(Candidate)).all()
        if not existing_candidates:
            cv_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cv_dataset.json")
            if os.path.exists(cv_file):
                with open(cv_file, "r", encoding="utf-8") as f:
                    candidates_data = json.load(f)
                seen_emails = set()
                for c in candidates_data:
                    email = c.get("email", "candidate@example.com")
                    if email in seen_emails:
                        continue
                    seen_emails.add(email)

                    try:
                        cid = uuid.UUID(c.get("id"))
                    except Exception:
                        cid = uuid.uuid5(uuid.NAMESPACE_DNS, c.get("id", str(uuid.uuid4())))

                    cand = Candidate(
                        id=cid,
                        name=c.get("name"),
                        title=c.get("title", "Software Engineer"),
                        email=email,
                        location=c.get("location", "Algiers, Algeria"),
                        years_experience=c.get("years_experience", 3),
                        bio=c.get("bio", ""),
                        skills=c.get("skills", []),
                        experience=c.get("experience", []),
                        education=c.get("education", []),
                        raw_cv_text=c.get("raw_cv_text", ""),
                        embedding_status="pending"
                    )
                    session.add(cand)
                session.commit()
                logger.info(f"Imported {len(seen_emails)} candidates into database")

def reset_database_data():
    # Clear out tables and reseed with fresh data
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    seed_initial_data()

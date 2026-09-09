from uuid import UUID, uuid4
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlmodel import SQLModel, Field, JSON
from pydantic import BaseModel

# Database models

class User(SQLModel, table=True):
    __tablename__ = "user"
    __table_args__ = {"extend_existing": True}
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    email: str = Field(index=True, unique=True, nullable=False)
    role: str = Field(default="candidate")  # recruiter or candidate
    hashed_password: str = Field(nullable=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Job(SQLModel, table=True):
    __tablename__ = "job"
    __table_args__ = {"extend_existing": True}
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    recruiter_id: Optional[UUID] = Field(default=None, foreign_key="user.id", nullable=True)
    title: str = Field(index=True)
    company: str = Field(default="Tech Enterprise")
    location: str = Field(default="Algiers, Algeria")
    type: str = Field(default="Full-Time")
    experience_level: str = Field(default="Mid-Senior (3+ years)")
    description: str
    required_skills: List[str] = Field(default_factory=list, sa_type=JSON)
    nice_to_have_skills: List[str] = Field(default_factory=list, sa_type=JSON)
    min_experience_years: int = Field(default=3)
    embedding_status: str = Field(default="pending")  # pending, ready, failed
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Candidate(SQLModel, table=True):
    __tablename__ = "candidate"
    __table_args__ = {"extend_existing": True}
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: Optional[UUID] = Field(default=None, foreign_key="user.id", nullable=True)
    name: str = Field(index=True)
    title: str = Field(default="Software Engineer")
    email: str = Field(default="candidate@example.com", index=True)
    location: str = Field(default="Algiers, Algeria")
    years_experience: int = Field(default=3)
    bio: str = Field(default="")
    skills: List[str] = Field(default_factory=list, sa_type=JSON)
    experience: List[Dict[str, Any]] = Field(default_factory=list, sa_type=JSON)
    education: List[Dict[str, Any]] = Field(default_factory=list, sa_type=JSON)
    raw_cv_text: Optional[str] = Field(default=None)
    parsed_entities: Dict[str, Any] = Field(default_factory=dict, sa_type=JSON)
    embedding_status: str = Field(default="pending")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class CVChunk(SQLModel, table=True):
    __tablename__ = "cv_chunk"
    __table_args__ = {"extend_existing": True}
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    candidate_id: UUID = Field(foreign_key="candidate.id", index=True)
    chunk_text: str
    chunk_type: str = Field(default="experience")
    vector_id: str = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Request and response schemas

class UserRegister(BaseModel):
    email: str
    password: str
    role: str = "candidate"

class UserLogin(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: str
    role: str
    email: str

class JobCreate(BaseModel):
    title: str
    company: str = "Tech Enterprise"
    location: str = "Algiers, Algeria"
    type: str = "Full-Time"
    experience_level: str = "Mid-Senior (3+ years)"
    description: str
    required_skills: List[str] = []
    nice_to_have_skills: List[str] = []
    min_experience_years: int = 3

class CandidateCreate(BaseModel):
    id: Optional[str] = None
    name: str
    title: str = "Software Engineer"
    email: str = "candidate@example.com"
    location: str = "Algiers, Algeria"
    years_experience: int = 3
    bio: str = ""
    skills: List[str] = []
    experience: List[Dict[str, Any]] = []
    education: List[Dict[str, Any]] = []
    raw_cv_text: Optional[str] = None

class InterviewQuestion(BaseModel):
    category: str
    topic: str
    question: str
    rationale: str

class AnalystReport(BaseModel):
    candidate_id: Optional[str] = None
    candidate_name: Optional[str] = None
    job_id: Optional[str] = None
    match_score: Optional[float] = None
    executive_summary: str
    strengths: List[str] = []
    gaps: List[str] = []
    risks: List[str] = []
    recommendation: Optional[str] = None
    hiring_recommendation: str = "Consider"
    interview_questions: List[InterviewQuestion] = []

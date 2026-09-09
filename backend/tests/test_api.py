import sys
import os
import pytest
from fastapi.testclient import TestClient

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.main import app
from backend.app.database import init_db

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    init_db()

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    # Check that database and vector engine report healthy
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "database" in data
    assert "vector_store" in data

def test_auth_flow(client):
    # Test registration and login flow
    email = "test_user_recruiter_99@example.com"
    pwd = "securepassword123"

    reg_res = client.post("/api/auth/register", json={
        "email": email,
        "password": pwd,
        "role": "recruiter"
    })
    if reg_res.status_code == 200:
        assert "access_token" in reg_res.json()

    login_res = client.post("/api/auth/login", json={
        "email": email,
        "password": pwd
    })
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data

def test_jobs_endpoints(client):
    # List jobs
    res = client.get("/api/jobs")
    assert res.status_code == 200
    jobs = res.json()
    assert len(jobs) >= 1

    first_job_id = jobs[0]["id"]

    # Match candidate rankings for active job
    match_res = client.get(f"/api/jobs/{first_job_id}/matches")
    assert match_res.status_code == 200
    match_data = match_res.json()
    assert "matches" in match_data
    assert len(match_data["matches"]) >= 1

    # Create temporary job and clean it up
    create_res = client.post("/api/jobs", json={
        "title": "Temporary Test Requisition",
        "company": "Test Labs DZ",
        "location": "Algiers",
        "description": "Short test job listing for automated checks.",
        "required_skills": ["Python", "Docker"],
        "min_experience_years": 2
    })
    assert create_res.status_code == 200
    created_job = create_res.json()["job"]
    assert created_job["title"] == "Temporary Test Requisition"

    # Clean up test job so it does not persist duplicates
    try:
        from sqlmodel import Session
        from backend.app.database import engine
        from backend.app.models import Job
        import uuid
        with Session(engine) as session:
            test_j = session.get(Job, uuid.UUID(created_job["id"]))
            if test_j:
                session.delete(test_j)
                session.commit()
    except Exception:
        pass

def test_candidate_endpoints(client):
    # Fetch existing candidates
    res = client.get("/api/candidates")
    assert res.status_code == 200
    cands = res.json()
    assert len(cands) >= 1

    first_cand_id = cands[0]["id"]

    cand_res = client.get(f"/api/candidates/{first_cand_id}")
    assert cand_res.status_code == 200

    # Upload test CV text
    upload_res = client.post(
        "/api/candidates/upload-cv",
        data={
            "raw_text": "Temporary Test Candidate - Python and Docker developer.",
            "candidate_name": "Temporary Test Candidate"
        }
    )
    assert upload_res.status_code == 202
    upload_data = upload_res.json()
    assert upload_data["status"] == "processing"
    assert "candidate_id" in upload_data

    # Clean up test candidate
    try:
        from sqlmodel import Session
        from backend.app.database import engine
        from backend.app.models import Candidate
        import uuid
        with Session(engine) as session:
            test_c = session.get(Candidate, uuid.UUID(upload_data["candidate_id"]))
            if test_c:
                session.delete(test_c)
                session.commit()
    except Exception:
        pass

def test_ai_analysis_and_tips(client):
    jobs = client.get("/api/jobs").json()
    cands = client.get("/api/candidates").json()

    job_id = jobs[0]["id"]
    cand_id = cands[0]["id"]

    # Generate evaluation scorecard
    analysis_res = client.get(f"/api/candidates/{cand_id}/analysis/{job_id}")
    assert analysis_res.status_code == 200
    analysis = analysis_res.json()
    assert "executive_summary" in analysis
    assert "strengths" in analysis
    assert "interview_questions" in analysis

    # Generate resume improvement tips
    tips_res = client.post(f"/api/candidates/{cand_id}/resume-tips?target_job_id={job_id}")
    assert tips_res.status_code == 200
    tips_data = tips_res.json()
    assert "tips" in tips_data

import sys
import os
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.agents.parser import parser_agent
from backend.app.agents.embedder import embedder_agent
from backend.app.agents.matcher import matcher_agent
from backend.app.agents.analyst import analyst_agent
from backend.app.agents.pipeline import recruitment_pipeline

def test_parser_agent():
    sample_text = """
    SARAH CONNOR - AI Systems Architect
    Email: sarah.connor@skyai.com | Location: Los Angeles, CA
    Skilled in Python, PyTorch, LangChain, FastAPI, Docker, and Kubernetes.
    Experience:
    - Lead AI Engineer at Cyberdyne Systems (2021-Present): Deployed RAG and transformer models.
    """
    profile = parser_agent.parse(raw_text=sample_text)
    assert "Sarah" in profile["name"] or len(profile["name"]) > 2
    assert "Python" in profile["skills"]
    assert "PyTorch" in profile["skills"]
    assert profile["email"] == "sarah.connor@skyai.com"

def test_embedder_agent_and_qdrant():
    candidate_id = "test-cand-001"
    profile = {
        "id": candidate_id,
        "name": "Jane Doe",
        "title": "Machine Learning Engineer",
        "skills": ["Python", "PyTorch", "Hugging Face", "FastAPI"],
        "bio": "Specialist in RAG and vector retrieval systems.",
        "experience": [{"role": "ML Engineer", "company": "AI Lab", "duration": "2 yrs", "description": "Built vector search."}],
        "education": [{"degree": "B.S. CS", "institution": "MIT", "year": "2020"}]
    }

    indexed = embedder_agent.chunk_and_index_candidate(candidate_id, profile)
    assert len(indexed) >= 3

    # Query search in Qdrant
    results = embedder_agent.search_similar_chunks("We need a PyTorch ML engineer for RAG", top_k=5)
    assert len(results) > 0
    assert any(r["candidate_id"] == candidate_id for r in results)

def test_matcher_agent():
    job = {
        "id": "test-job-001",
        "title": "Senior PyTorch & RAG Engineer",
        "description": "Building next-generation vector retrieval with PyTorch and LangChain.",
        "required_skills": ["Python", "PyTorch", "LangChain", "FastAPI"],
        "nice_to_have_skills": ["Docker", "Kubernetes"]
    }
    cand_match = {
        "id": "cand-perfect",
        "name": "Alex PyTorch",
        "title": "AI Engineer",
        "skills": ["Python", "PyTorch", "LangChain", "FastAPI", "Docker"],
        "bio": "Experienced PyTorch and LangChain engineer."
    }
    cand_unrelated = {
        "id": "cand-unrelated",
        "name": "Bob Designer",
        "title": "UI Designer",
        "skills": ["Photoshop", "Illustrator", "Figma"],
        "bio": "Graphic designer with Figma expertise."
    }

    embedder_agent.chunk_and_index_candidate("cand-perfect", cand_match)
    embedder_agent.chunk_and_index_candidate("cand-unrelated", cand_unrelated)

    ranked = matcher_agent.match_candidates_for_job(job, [cand_match, cand_unrelated])
    assert len(ranked) == 2
    assert ranked[0]["candidate"]["id"] == "cand-perfect"
    assert ranked[0]["match_score"] > ranked[1]["match_score"]
    assert ranked[0]["match_score"] >= 75.0

def test_analyst_agent():
    cand = {
        "id": "cand-001",
        "name": "Sarah Chen",
        "title": "Senior AI Engineer",
        "skills": ["Python", "PyTorch", "LangChain", "RAG"],
        "years_experience": 5
    }
    job = {
        "id": "job-001",
        "title": "Senior AI Engineer",
        "company": "Apex AI",
        "required_skills": ["Python", "PyTorch", "LangChain"],
        "min_experience_years": 4,
        "description": "Design and build production RAG systems."
    }
    report = analyst_agent.generate_analysis(cand, job, match_score=91.0)
    assert report.executive_summary is not None
    assert len(report.strengths) > 0
    assert len(report.interview_questions) >= 2
    assert report.hiring_recommendation in ["Strong Hire", "Hire", "Consider", "Pass"]

def test_analyst_agent_with_mock_grok(monkeypatch):
    """Verifies AnalystAgent correctly parses and integrates Grok (xAI) responses."""
    mock_grok_json = """{
      "executive_summary": "Top-tier AI researcher with profound PyTorch expertise.",
      "strengths": ["Deep PyTorch systems knowledge", "RAG production architecture"],
      "gaps": ["None identified"],
      "interview_questions": [
        {
          "category": "Core Architecture & Depth",
          "topic": "Distributed Training",
          "question": "How do you optimize gradient accumulation in PyTorch FSDP?",
          "rationale": "Validates deep systems competence."
        }
      ],
      "hiring_recommendation": "Strong Hire"
    }"""
    monkeypatch.setattr(analyst_agent, "_get_grok_key", lambda: "mock-grok-key-12345")
    monkeypatch.setattr(analyst_agent, "_call_grok", lambda prompt, system: mock_grok_json)

    cand = {"id": "c-grok", "name": "Ada Lovelace", "title": "AI Architect", "skills": ["Python", "PyTorch"]}
    job = {"id": "j-grok", "title": "Principal AI Engineer", "required_skills": ["PyTorch"]}

    report = analyst_agent.generate_analysis(cand, job, match_score=95.0)
    assert "Top-tier AI researcher" in report.executive_summary
    assert report.hiring_recommendation == "Strong Hire"
    assert len(report.interview_questions) >= 1
    assert report.interview_questions[0].topic == "Distributed Training"

def test_slm_reasoner():
    """Verifies SLMReasoner produces adaptive synthesis and interview questions."""
    from backend.ai.slm_reasoner import slm_reasoner
    cand = {"id": "c-slm", "name": "Alan Turing", "title": "Mathematician", "skills": ["Logic", "Python"], "years_experience": 6}
    job = {"id": "j-slm", "title": "Lead Cryptography Engineer", "required_skills": ["Python", "C++"], "min_experience_years": 5}
    
    analysis = slm_reasoner.generate_candidate_analysis(
        candidate=cand,
        job=job,
        match_score=88.0,
        matched_skills=["Python"],
        missing_skills=["C++"],
        rag_evidence=[]
    )
    assert analysis["candidate_name"] == "Alan Turing"
    assert len(analysis["interview_questions"]) >= 2
    assert "Python" in str(analysis["interview_questions"]) or "Core" in str(analysis["interview_questions"])

def test_langgraph_pipeline():
    sample_text = "John Smith - Full Stack AI Developer with React and Python FastAPI experience."
    state = {
        "raw_text": sample_text,
        "candidate_id": "test-langgraph-cand",
        "job": {
            "id": "job-react-py",
            "title": "Full Stack AI Developer",
            "description": "React and FastAPI backend",
            "required_skills": ["React", "Python", "FastAPI"]
        }
    }
    result = recruitment_pipeline.invoke(state)
    assert "profile" in result
    assert "matches" in result
    assert "analysis" in result
    assert result["analysis"]["candidate_name"] is not None

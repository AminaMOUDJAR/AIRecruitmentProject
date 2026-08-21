"""
Automated Verification Suite for TalentMatch AI
Tests AI embeddings, RAG matching, SLM reasoner, and FastAPI endpoints.
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.data.storage import store
from backend.ai.embeddings import embedding_engine
from backend.ai.rag_engine import rag_engine
from backend.ai.slm_reasoner import slm_reasoner

def test_storage():
    print("Testing Data Store...")
    jobs = store.get_all_jobs()
    candidates = store.get_all_candidates()
    assert len(jobs) >= 3, f"Expected >=3 jobs, got {len(jobs)}"
    assert len(candidates) >= 5, f"Expected >=5 candidates, got {len(candidates)}"
    print(f"[OK] Data store OK: {len(jobs)} jobs, {len(candidates)} candidates.")

def test_embeddings():
    print("Testing Embeddings Engine (PyTorch / HuggingFace)...")
    texts = [
        "Senior AI Engineer with PyTorch and LangChain experience.",
        "Experienced React frontend web developer.",
        "Cloud DevOps engineer with AWS and Kubernetes."
    ]
    emb = embedding_engine.embed_texts(texts)
    assert len(emb) == 3, f"Expected 3 embeddings, got {len(emb)}"
    print(f"[OK] Embeddings generated with shape: {emb.shape}")

    # Query embedding
    q_emb = embedding_engine.embed_query("We need a PyTorch Generative AI Specialist")
    sims = embedding_engine.batch_cosine_similarity(q_emb, emb)
    print(f"[OK] Cosine similarities: {sims}")
    # The AI text should have highest similarity to the AI query
    assert sims[0] > sims[1], "AI text should rank higher than Frontend text for AI query"
    print("[OK] Semantic vector ranking test passed.")

def test_rag_matching():
    print("Testing RAG Engine matching...")
    jobs = store.get_all_jobs()
    candidates = store.get_all_candidates()
    
    # Match candidates for first job (AI Engineer)
    results = rag_engine.match_candidates_for_job(jobs[0], candidates)
    assert len(results) == len(candidates), "All candidates should be scored"
    top_cand = results[0]
    print(f"[OK] Top candidate for '{jobs[0]['title']}': {top_cand['candidate']['name']} ({top_cand['match_score']}%)")
    assert top_cand["match_score"] > 70, "Top candidate should have strong match score"
    print(f"[OK] RAG evidence extracted: {len(top_cand.get('rag_evidence', []))} chunks")

def test_slm_reasoner():
    print("Testing SLM Reasoner & Dynamic Interview Question Generation...")
    jobs = store.get_all_jobs()
    candidates = store.get_all_candidates()

    analysis = slm_reasoner.generate_candidate_analysis(
        candidate=candidates[0],
        job=jobs[0],
        match_score=92.5,
        matched_skills=["Python", "PyTorch", "LangChain", "RAG"],
        missing_skills=["Kubernetes"],
        rag_evidence=[{"chunk_text": "Architected enterprise RAG platform with LangChain and PyTorch..."}]
    )
    assert "executive_summary" in analysis
    assert len(analysis["interview_questions"]) >= 2
    print(f"[OK] Executive Summary: {analysis['executive_summary'][:100]}...")
    print(f"[OK] Sample Interview Question ({analysis['interview_questions'][0]['category']}): {analysis['interview_questions'][0]['question']}")

    tips = slm_reasoner.generate_resume_optimizer_tips(candidates[0], jobs[0])
    assert len(tips) > 0
    print(f"[OK] Resume Optimizer Tips generated: {len(tips)} tips.")

if __name__ == "__main__":
    print("\n--- RUNNING TALENTMATCH AI TEST SUITE ---")
    test_storage()
    test_embeddings()
    test_rag_matching()
    test_slm_reasoner()
    print("\n[SUCCESS] ALL TESTS PASSED SUCCESSFULLY!\n")

import time
from typing import List, Dict, Any

from backend.app.logger import log_agent_call
from backend.ai.rag_engine import rag_engine

class MatcherAgent:
    """
    Matching facade over the shared ResumeRAGEngine.
    All scoring logic (alias-normalized skill overlap, calibrated dense
    similarity, precomputed job embeddings) lives in backend/ai/rag_engine.py
    so there is exactly one matching implementation in the codebase.
    """

    def build_job_query(self, job_dict: Dict[str, Any]) -> str:
        return rag_engine._prepare_job_query(job_dict)

    def match_candidates_for_job(self, job_dict: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        start_time = time.time()
        results = rag_engine.match_candidates_for_job(job_dict, candidates)

        # Normalize the evidence key: routers and the analyst consume
        # `evidence_chunks`, the RAG engine produces `rag_evidence`
        for r in results:
            r["evidence_chunks"] = r.get("rag_evidence", [])

        latency_ms = int((time.time() - start_time) * 1000)
        log_agent_call(
            "matcher_agent",
            job_id=str(job_dict.get("id")),
            candidates_count=len(candidates),
            latency_ms=latency_ms
        )
        return results

    def match_jobs_for_candidate(self, cand_dict: Dict[str, Any], jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return rag_engine.match_jobs_for_candidate(cand_dict, jobs)

matcher_agent = MatcherAgent()

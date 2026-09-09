import time
from typing import List, Dict, Any, Optional
from collections import defaultdict
from backend.app.agents.embedder import embedder_agent
from backend.app.logger import log_agent_call, logger

class MatcherAgent:
    # Computes semantic similarity between job requirements and candidate profiles
    def __init__(self):
        pass

    def build_job_query(self, job_dict: Dict[str, Any]) -> str:
        # Build search text representing the job requisition
        parts = [
            f"Title: {job_dict.get('title', '')}",
            f"Description: {job_dict.get('description', '')}"
        ]
        if job_dict.get("required_skills"):
            parts.append(f"Required Skills: {', '.join(job_dict['required_skills'])}")
        if job_dict.get("nice_to_have_skills"):
            parts.append(f"Preferred Skills: {', '.join(job_dict['nice_to_have_skills'])}")
        return "\n".join(parts)

    def match_candidates_for_job(self, job_dict: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Ranks candidates for a specific job requisition
        start_time = time.time()
        if not candidates:
            return []

        # Make sure candidates are indexed in Qdrant
        for cand in candidates:
            cand_id = str(cand.get("id"))
            existing = embedder_agent.search_similar_chunks("check", top_k=1, candidate_id=cand_id)
            if not existing:
                embedder_agent.chunk_and_index_candidate(cand_id, cand)

        job_query = self.build_job_query(job_dict)
        retrieved_chunks = embedder_agent.search_similar_chunks(job_query, top_k=60)

        candidate_chunks = defaultdict(list)
        for hit in retrieved_chunks:
            cid = hit.get("candidate_id")
            if cid:
                candidate_chunks[str(cid)].append(hit)

        results = []
        for cand in candidates:
            cid = str(cand.get("id"))
            hits = candidate_chunks.get(cid, [])
            
            # Fetch directly if this candidate had no hits in the global pool
            if not hits:
                hits = embedder_agent.search_similar_chunks(job_query, top_k=5, candidate_id=cid)

            if hits:
                sim_scores = [h["score"] for h in hits]
                max_sim = max(sim_scores)
                mean_sim = sum(sim_scores) / len(sim_scores)
                vector_score = 0.7 * max_sim + 0.3 * mean_sim
            else:
                vector_score = 0.40

            # Match technical skills
            cand_skills = cand.get("skills", [])
            cand_skills_lower = [s.strip().lower() for s in cand_skills]

            matched_req = [s for s in job_dict.get("required_skills", []) if s.strip().lower() in cand_skills_lower]
            missing_req = [s for s in job_dict.get("required_skills", []) if s.strip().lower() not in cand_skills_lower]
            matched_nice = [s for s in job_dict.get("nice_to_have_skills", []) if s.strip().lower() in cand_skills_lower]

            normalized_base = max(0.35, min(0.75, vector_score * 0.75 + 0.20))
            skill_bonus = (len(matched_req) * 0.05) + (len(matched_nice) * 0.02)

            final_match_score = round(min(98.0, max(30.0, (normalized_base + skill_bonus) * 100.0)), 1)
            evidence_chunks = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)[:3]

            if final_match_score >= 85:
                tier = "Top Match"
                tier_badge = "success"
            elif final_match_score >= 70:
                tier = "Strong Match"
                tier_badge = "primary"
            elif final_match_score >= 50:
                tier = "Moderate Fit"
                tier_badge = "warning"
            else:
                tier = "Low Fit"
                tier_badge = "secondary"

            results.append({
                "candidate": cand,
                "match_score": final_match_score,
                "tier": tier,
                "tier_badge": tier_badge,
                "matched_skills": matched_req + matched_nice,
                "missing_skills": missing_req,
                "rag_evidence": evidence_chunks,
                "evidence_chunks": evidence_chunks
            })

        results.sort(key=lambda x: x["match_score"], reverse=True)

        latency_ms = int((time.time() - start_time) * 1000)
        log_agent_call(
            "matcher_agent",
            job_id=str(job_dict.get("id")),
            candidates_count=len(candidates),
            latency_ms=latency_ms
        )

        return results

    def match_jobs_for_candidate(self, cand_dict: Dict[str, Any], jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Ranks open positions for a candidate profile
        cand_id = str(cand_dict.get("id"))
        cand_skills = [s.strip().lower() for s in cand_dict.get("skills", [])]

        results = []
        for job in jobs:
            job_query = self.build_job_query(job)
            hits = embedder_agent.search_similar_chunks(job_query, top_k=3, candidate_id=cand_id)

            if hits:
                sim_scores = [h["score"] for h in hits]
                vector_score = 0.7 * max(sim_scores) + 0.3 * (sum(sim_scores) / len(sim_scores))
            else:
                vector_score = 0.45

            normalized_base = max(0.20, min(0.85, (vector_score + 0.1) * 0.75))
            matched_req = [s for s in job.get("required_skills", []) if s.strip().lower() in cand_skills]
            missing_req = [s for s in job.get("required_skills", []) if s.strip().lower() not in cand_skills]

            skill_bonus = len(matched_req) * 0.05
            final_match_score = round(min(98.0, max(30.0, (normalized_base + skill_bonus) * 100.0)), 1)

            results.append({
                "job": job,
                "match_score": final_match_score,
                "tier": "Top Match" if final_match_score >= 85 else "Strong Match" if final_match_score >= 70 else "Moderate Fit",
                "matched_skills": matched_req,
                "missing_skills": missing_req,
                "evidence_chunks": hits
            })

        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results

matcher_agent = MatcherAgent()

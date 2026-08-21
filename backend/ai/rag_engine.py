import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings import embedding_engine

logger = logging.getLogger(__name__)

class ResumeRAGEngine:
    """
    RAG & Vector Retrieval Engine for Candidate CVs.
    Uses LangChain text chunking, HuggingFace embeddings, and PyTorch similarity matching.
    """
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=60,
            separators=["\n\n", "\n", "- ", ". ", " "]
        )

    def _prepare_candidate_corpus(self, candidate: Dict[str, Any]) -> str:
        """Constructs a comprehensive textual representation of a candidate."""
        parts = []
        if candidate.get("name"):
            parts.append(f"Candidate: {candidate['name']}")
        if candidate.get("title"):
            parts.append(f"Title / Role: {candidate['title']}")
        if candidate.get("bio"):
            parts.append(f"Summary: {candidate['bio']}")
        if candidate.get("skills"):
            parts.append(f"Skills: {', '.join(candidate['skills'])}")
        
        if candidate.get("experience"):
            exp_text = []
            for exp in candidate["experience"]:
                exp_text.append(f"- {exp.get('role')} at {exp.get('company')} ({exp.get('duration')}): {exp.get('description')}")
            parts.append("Work Experience:\n" + "\n".join(exp_text))

        if candidate.get("education"):
            edu_text = []
            for edu in candidate["education"]:
                edu_text.append(f"- {edu.get('degree')} from {edu.get('institution')} ({edu.get('year')})")
            parts.append("Education:\n" + "\n".join(edu_text))

        if candidate.get("raw_cv_text"):
            parts.append("Resume Details:\n" + candidate["raw_cv_text"])

        return "\n\n".join(parts)

    def _prepare_job_query(self, job: Dict[str, Any]) -> str:
        """Constructs a search query from a job posting."""
        parts = [
            f"Job Title: {job.get('title', '')}",
            f"Description: {job.get('description', '')}",
        ]
        if job.get("required_skills"):
            parts.append(f"Required Skills: {', '.join(job['required_skills'])}")
        if job.get("nice_to_have_skills"):
            parts.append(f"Preferred Skills: {', '.join(job['nice_to_have_skills'])}")
        if job.get("experience_level"):
            parts.append(f"Experience Level: {job['experience_level']}")
        return "\n".join(parts)

    def retrieve_relevant_cv_chunks(self, candidate: Dict[str, Any], job_query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        RAG Component: Chunks a candidate's CV using LangChain, embeds chunks,
        and retrieves the top_k most relevant snippets for the job query.
        """
        full_text = self._prepare_candidate_corpus(candidate)
        chunks = self.text_splitter.split_text(full_text)
        if not chunks:
            return []

        chunk_embeddings = embedding_engine.embed_texts(chunks)
        query_embedding = embedding_engine.embed_query(job_query_text)

        similarities = embedding_engine.batch_cosine_similarity(query_embedding, chunk_embeddings)
        
        # Rank chunks by score
        indexed_scores = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        top_chunks = []
        for idx, score in indexed_scores[:top_k]:
            top_chunks.append({
                "chunk_text": chunks[idx],
                "relevance_score": round(float(score) * 100, 1)
            })
        return top_chunks

    def match_candidates_for_job(self, job: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Semantic matching and ranking of all candidates against a job description.
        Combines dense vector similarity with skill overlap bonus for balanced 0-100% scores.
        """
        if not candidates:
            return []

        job_query = self._prepare_job_query(job)
        job_emb = embedding_engine.embed_query(job_query)

        # Generate candidate representations
        cand_texts = [self._prepare_candidate_corpus(c) for c in candidates]
        cand_embeddings = embedding_engine.embed_texts(cand_texts)

        # Cosine similarities
        dense_scores = embedding_engine.batch_cosine_similarity(job_emb, cand_embeddings)

        job_skills = set(s.lower().strip() for s in job.get("required_skills", []))
        nice_skills = set(s.lower().strip() for s in job.get("nice_to_have_skills", []))

        results = []
        for idx, candidate in enumerate(candidates):
            raw_cosine = float(dense_scores[idx]) if len(dense_scores) > idx else 0.5
            # Scale cosine similarity from typical range [0.2 - 0.9] to realistic [40% - 98%]
            base_score = max(0.0, min(1.0, (raw_cosine - 0.15) / 0.70))

            # Skill overlap bonus/penalty
            cand_skills = set(s.lower().strip() for s in candidate.get("skills", []))
            
            matched_req = [s for s in job.get("required_skills", []) if s.lower().strip() in cand_skills]
            missing_req = [s for s in job.get("required_skills", []) if s.lower().strip() not in cand_skills]
            matched_nice = [s for s in job.get("nice_to_have_skills", []) if s.lower().strip() in cand_skills]

            req_ratio = len(matched_req) / max(len(job_skills), 1) if job_skills else 1.0
            nice_ratio = len(matched_nice) / max(len(nice_skills), 1) if nice_skills else 0.0

            # Blended score: 65% dense semantic vector + 25% required skills + 10% nice-to-have
            blended_score = (base_score * 0.65) + (req_ratio * 0.25) + (nice_ratio * 0.10)
            final_percentage = round(min(98.0, max(25.0, blended_score * 100.0)), 1)

            # Retrieve top RAG context chunks
            top_chunks = self.retrieve_relevant_cv_chunks(candidate, job_query, top_k=2)

            # Determine fit tier
            if final_percentage >= 85:
                tier = "Top Match"
                tier_badge = "success"
            elif final_percentage >= 70:
                tier = "Strong Match"
                tier_badge = "primary"
            elif final_percentage >= 50:
                tier = "Moderate Fit"
                tier_badge = "warning"
            else:
                tier = "Low Fit"
                tier_badge = "secondary"

            results.append({
                "candidate": candidate,
                "match_score": final_percentage,
                "tier": tier,
                "tier_badge": tier_badge,
                "matched_skills": matched_req + matched_nice,
                "missing_skills": missing_req,
                "rag_evidence": top_chunks
            })

        # Sort by match score descending
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results

    def match_jobs_for_candidate(self, candidate: Dict[str, Any], jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Job Seeker View: Given a candidate, ranks available jobs by semantic fit.
        """
        if not jobs:
            return []

        cand_text = self._prepare_candidate_corpus(candidate)
        cand_emb = embedding_engine.embed_query(cand_text)

        job_texts = [self._prepare_job_query(j) for j in jobs]
        job_embeddings = embedding_engine.embed_texts(job_texts)

        dense_scores = embedding_engine.batch_cosine_similarity(cand_emb, job_embeddings)

        cand_skills = set(s.lower().strip() for s in candidate.get("skills", []))

        results = []
        for idx, job in enumerate(jobs):
            raw_cosine = float(dense_scores[idx]) if len(dense_scores) > idx else 0.5
            base_score = max(0.0, min(1.0, (raw_cosine - 0.15) / 0.70))

            job_skills = set(s.lower().strip() for s in job.get("required_skills", []))
            matched_req = [s for s in job.get("required_skills", []) if s.lower().strip() in cand_skills]
            missing_req = [s for s in job.get("required_skills", []) if s.lower().strip() not in cand_skills]

            req_ratio = len(matched_req) / max(len(job_skills), 1) if job_skills else 1.0
            blended_score = (base_score * 0.70) + (req_ratio * 0.30)
            final_percentage = round(min(98.0, max(25.0, blended_score * 100.0)), 1)

            if final_percentage >= 85:
                tier = "Top Match"
            elif final_percentage >= 70:
                tier = "Strong Match"
            elif final_percentage >= 50:
                tier = "Moderate Fit"
            else:
                tier = "Low Fit"

            results.append({
                "job": job,
                "match_score": final_percentage,
                "tier": tier,
                "matched_skills": matched_req,
                "missing_skills": missing_req
            })

        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results


rag_engine = ResumeRAGEngine()

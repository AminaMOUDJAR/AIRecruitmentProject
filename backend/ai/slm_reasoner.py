import os
import json
import logging
from typing import Dict, Any, List, Optional
import torch
import httpx

from backend.app.config import settings

logger = logging.getLogger(__name__)

# Preferred lightweight Small Language Model (SLM) from Hugging Face
DEFAULT_SLM_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

class SLMReasoner:
    """
    Small Language Model (SLM) & Grok Reasoning Engine for TalentMatch AI.
    Performs candidate qualitative synthesis, dynamic interview question generation,
    and CV optimization advice using real generative inference with graceful fallback.
    """
    def __init__(self, model_name: str = DEFAULT_SLM_MODEL):
        self.model_name = getattr(settings, "SLM_MODEL_NAME", model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipeline = None
        self._is_slm_loaded = False
        # Lazy load model on demand to maintain instant server startup
        self._init_attempted = False

    def _try_load_slm(self):
        """Attempts to load a lightweight Hugging Face SLM pipeline."""
        if self._init_attempted:
            return
        self._init_attempted = True
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            token = getattr(settings, "HF_TOKEN", None) or os.getenv("HF_TOKEN")
            logger.info(f"Loading SLM '{self.model_name}' on {self.device}...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                low_cpu_mem_usage=True,
                token=token
            )
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            self._is_slm_loaded = True
            logger.info("SLM pipeline loaded successfully.")
        except Exception as e:
            logger.info(f"SLM local weights not pre-cached ({e}). Running in fast analytical reasoning mode.")
            self._is_slm_loaded = False

    def _call_grok(self, prompt: str, system_prompt: str) -> Optional[str]:
        """Calls Grok (xAI API) if configured."""
        api_key = getattr(settings, "GROK_API_KEY", None) or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
        if not api_key or not api_key.strip():
            return None
        try:
            base_url = getattr(settings, "GROK_BASE_URL", "https://api.x.ai/v1").rstrip("/")
            model = getattr(settings, "GROK_MODEL", "grok-2-latest")
            headers = {
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"}
            }
            with httpx.Client(timeout=getattr(settings, "LLM_TIMEOUT_SECONDS", 30.0)) as client:
                res = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Grok API call in SLMReasoner failed: {e}")
        return None

    def _generate_with_slm(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Executes actual generative inference with the loaded HuggingFace SLM."""
        if not self._is_slm_loaded or self._pipeline is None:
            return ""
        try:
            outputs = self._pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._pipeline.tokenizer.eos_token_id if hasattr(self._pipeline, "tokenizer") and self._pipeline.tokenizer else None,
                return_full_text=False
            )
            if outputs and len(outputs) > 0:
                return outputs[0].get("generated_text", "").strip()
        except Exception as e:
            logger.error(f"Error during SLM model generation: {e}")
        return ""

    def generate_candidate_analysis(
        self,
        candidate: Dict[str, Any],
        job: Dict[str, Any],
        match_score: float,
        matched_skills: List[str],
        missing_skills: List[str],
        rag_evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generates qualitative AI analysis for a recruiter:
        - Executive Assessment Summary
        - Strengths
        - Gaps / Risks
        - Tailored Technical Interview Questions
        Executes Grok or Hugging Face SLM directly when available, falling back to analytical templates.
        """
        cand_name = candidate.get("name", "Candidate")
        job_title = job.get("title", "the role")
        cand_title = candidate.get("title", "Specialist")
        years_exp = candidate.get("years_experience", 3)
        min_years = job.get("min_experience_years", 3)

        # 1. Check Grok API
        grok_system = (
            "You are a technical recruiting analyst. Output valid JSON matching:\n"
            '{"executive_summary": "...", "recommendation": "...", "strengths": ["..."], "risks": ["..."]}'
        )
        grok_prompt = (
            f"Candidate: {cand_name}, {cand_title} ({years_exp} yrs exp). "
            f"Target Job: {job_title} (requires {min_years} yrs). "
            f"Match Score: {match_score}%. "
            f"Matched Skills: {', '.join(matched_skills) if matched_skills else 'None'}. "
            f"Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}."
        )
        grok_resp = self._call_grok(grok_prompt, grok_system)
        if grok_resp:
            try:
                parsed = json.loads(grok_resp)
                questions = self._generate_tailored_interview_questions(candidate, job, missing_skills, matched_skills)
                return {
                    "candidate_id": candidate.get("id"),
                    "candidate_name": cand_name,
                    "job_id": job.get("id"),
                    "match_score": match_score,
                    "executive_summary": parsed.get("executive_summary", ""),
                    "recommendation": parsed.get("recommendation", "Candidate screened."),
                    "strengths": parsed.get("strengths", []),
                    "risks": parsed.get("risks", []),
                    "interview_questions": questions
                }
            except Exception:
                pass

        # 2. Check direct HuggingFace SLM pipeline
        self._try_load_slm()
        if self._is_slm_loaded:
            slm_prompt = (
                f"<|im_start|>system\nYou are a hiring assessment AI. Provide a concise 2-sentence executive summary evaluating candidate technical fit.<|im_end|>\n"
                f"<|im_start|>user\nCandidate: {cand_name} ({cand_title}, {years_exp} years experience).\n"
                f"Applying for: {job_title}.\n"
                f"Fit Score: {match_score}%. Key Skills: {', '.join(matched_skills[:4]) if matched_skills else 'general'}. Gaps: {', '.join(missing_skills[:3]) if missing_skills else 'none'}.\n"
                f"Executive Assessment:<|im_end|>\n<|im_start|>assistant\n"
            )
            generated_summary = self._generate_with_slm(slm_prompt, max_new_tokens=120)
            if generated_summary and len(generated_summary.strip()) > 30:
                summary = generated_summary.strip()
                if match_score >= 85:
                    recommendation = "Strong candidate for immediate technical phone screen."
                elif match_score >= 68:
                    recommendation = "Recommended for interview focusing on missing stack items."
                else:
                    recommendation = "Keep on file for roles more aligned with their primary skill set."

                strengths = [f"Direct alignment on key technologies: {', '.join(matched_skills[:4])}."] if matched_skills else [f"Relevant background as {cand_title}."]
                if years_exp >= min_years:
                    strengths.append(f"Meets seniority benchmark with {years_exp} years relevant experience.")
                if rag_evidence:
                    strengths.append(f"Practical achievement: \"{rag_evidence[0].get('chunk_text', '')[:100]}...\"")

                risks = [f"Unverified proficiency in: {', '.join(missing_skills[:3])}."] if missing_skills else ["No critical technical blockers identified."]
                if years_exp < min_years:
                    risks.append(f"Experience ({years_exp} yrs) is below requested {min_years} yrs benchmark.")

                questions = self._generate_tailored_interview_questions(candidate, job, missing_skills, matched_skills)
                return {
                    "candidate_id": candidate.get("id"),
                    "candidate_name": cand_name,
                    "job_id": job.get("id"),
                    "match_score": match_score,
                    "executive_summary": summary,
                    "recommendation": recommendation,
                    "strengths": strengths,
                    "risks": risks,
                    "interview_questions": questions
                }

        # 3. Deterministic analytical reasoning fallback
        if match_score >= 85:
            summary = (
                f"{cand_name} demonstrates exceptional alignment ({match_score}%) with the {job_title} position. "
                f"With {years_exp} years of background as a {cand_title} and verified hands-on experience in "
                f"{', '.join(matched_skills[:4]) if matched_skills else 'core stack'}, their profile strongly satisfies primary technical criteria."
            )
            recommendation = "Strong candidate for immediate technical phone screen."
        elif match_score >= 68:
            summary = (
                f"{cand_name} is a solid contender ({match_score}%) for {job_title}. "
                f"Possesses strong foundational skills in {', '.join(matched_skills[:3]) if matched_skills else 'key areas'}. "
                f"Some domain-specific adjustments or onboarding on {', '.join(missing_skills[:2]) if missing_skills else 'advanced tooling'} may be needed."
            )
            recommendation = "Recommended for interview focusing on missing stack items."
        else:
            summary = (
                f"{cand_name} has notable experience in {cand_title}, but displays a technical gap ({match_score}%) for {job_title}. "
                f"Candidate would require significant upskilling in {', '.join(missing_skills[:3]) if missing_skills else 'core requirements'}."
            )
            recommendation = "Keep on file for roles more aligned with their primary skill set."

        strengths = []
        if matched_skills:
            strengths.append(f"Direct match on key requirements: {', '.join(matched_skills[:4])}.")
        if years_exp >= min_years:
            strengths.append(f"Meets seniority benchmark with {years_exp} years relevant experience.")
        if rag_evidence:
            strengths.append(f"Resume highlights practical achievement: \"{rag_evidence[0].get('chunk_text', '')[:120]}...\"")
        if not strengths:
            strengths.append("Demonstrated professional background in related technical domain.")

        risks = []
        if missing_skills:
            risks.append(f"Unverified proficiency in: {', '.join(missing_skills[:3])}.")
        if years_exp < min_years:
            risks.append(f"Experience ({years_exp} yrs) is below requested {min_years} yrs.")
        if not risks:
            risks.append("No critical technical blockers identified from resume review.")

        questions = self._generate_tailored_interview_questions(candidate, job, missing_skills, matched_skills)

        return {
            "candidate_id": candidate.get("id"),
            "candidate_name": cand_name,
            "job_id": job.get("id"),
            "match_score": match_score,
            "executive_summary": summary,
            "recommendation": recommendation,
            "strengths": strengths,
            "risks": risks,
            "interview_questions": questions
        }

    def _generate_tailored_interview_questions(self, candidate: Dict[str, Any], job: Dict[str, Any], missing_skills: List[str], matched_skills: List[str]) -> List[Dict[str, str]]:
        """Generates dynamic interview questions tailored to candidate's strengths and resume gaps."""
        cand_name = candidate.get("name", "the candidate")
        job_title = job.get("title", "Role")

        # Try Hugging Face SLM generation for adaptive questioning if loaded
        if self._is_slm_loaded:
            primary_topic = matched_skills[0] if matched_skills else "System Architecture"
            gap_topic = missing_skills[0] if missing_skills else "Scalability"
            slm_q_prompt = (
                f"<|im_start|>system\nGenerate one technical interview question evaluating {primary_topic} depth for a {job_title}. Output only the question.<|im_end|>\n"
                f"<|im_start|>user\nCandidate skills: {', '.join(matched_skills[:3])}.\nQuestion:<|im_end|>\n<|im_start|>assistant\n"
            )
            generated_q = self._generate_with_slm(slm_q_prompt, max_new_tokens=60)
            if generated_q and "?" in generated_q:
                return [
                    {
                        "category": "Core Architecture & Depth",
                        "topic": primary_topic,
                        "question": generated_q.strip(),
                        "rationale": f"Validates technical proficiency and depth in {primary_topic}."
                    },
                    {
                        "category": "Skill Gap Evaluation",
                        "topic": gap_topic,
                        "question": f"How would you approach ramping up on {gap_topic} within a high-tempo engineering team?",
                        "rationale": f"Assesses adaptability and ramp-up trajectory in {gap_topic}."
                    },
                    {
                        "category": "Engineering Trade-offs",
                        "topic": "System Design",
                        "question": "Describe a scenario where you made a critical trade-off between latency and complexity. What guided your decision?",
                        "rationale": "Evaluates architectural judgment and decision-making rigor."
                    }
                ]

        # Analytical rule-based question set fallback
        questions = []
        if matched_skills:
            primary_skill = matched_skills[0]
            questions.append({
                "category": "Core Architecture & Depth",
                "topic": primary_skill,
                "question": f"Can you walk us through a recent project where you applied {primary_skill} to solve a high-scale or latency-critical problem?",
                "rationale": f"Validates depth in candidate's declared strength ({primary_skill})."
            })

        if missing_skills:
            gap_skill = missing_skills[0]
            questions.append({
                "category": "Skill Gap Evaluation",
                "topic": gap_skill,
                "question": f"Our team utilizes {gap_skill} extensively for this {job_title} role. What is your conceptual understanding of {gap_skill}, and how would you ramp up quickly?",
                "rationale": f"Tests adaptability and willingness to bridge the gap in {gap_skill}."
            })
        else:
            questions.append({
                "category": "System Design",
                "topic": "Scalability & Reliability",
                "question": "How do you approach benchmarking, caching, and observability in a distributed production service?",
                "rationale": "Evaluates end-to-end engineering rigor beyond isolated code tasks."
            })

        questions.append({
            "category": "Behavioral & Engineering Trade-offs",
            "topic": "Technical Decision Making",
            "question": "Describe a scenario where you had to compromise between model accuracy / system throughput and development deadline. How did you decide?",
            "rationale": "Assesses pragmatic engineering mindset and trade-off analysis."
        })

        return questions

    def generate_resume_optimizer_tips(self, candidate: Dict[str, Any], target_job: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """
        AI Resume Optimizer for Job Seekers:
        Generates actionable suggestions to boost profile visibility and match score.
        """
        tips = []
        cand_skills = candidate.get("skills", [])

        if target_job:
            job_skills = set(s.lower() for s in target_job.get("required_skills", []))
            user_skills = set(s.lower() for s in cand_skills)
            missing = [s for s in target_job.get("required_skills", []) if s.lower() not in user_skills]

            if missing:
                tips.append({
                    "priority": "High",
                    "title": f"Highlight experience with {', '.join(missing[:2])}",
                    "description": f"The target job '{target_job.get('title')}' emphasizes {', '.join(missing[:3])}. If you have adjacent experience or side projects with these, add them prominently to your skills and work achievements."
                })

        # General CV optimization rules
        raw_text = candidate.get("raw_cv_text", "")
        if "%" not in raw_text and "reduced" not in raw_text.lower() and "improved" not in raw_text.lower():
            tips.append({
                "priority": "Medium",
                "title": "Quantify Impact with Concrete Metrics",
                "description": "Recruiters and AI matchers favor measurable outcomes. Quantify achievements (e.g. 'reduced latency by 35%', 'increased throughput by 2x', 'saved $15k in GPU cloud costs')."
            })

        if len(cand_skills) < 6:
            tips.append({
                "priority": "Medium",
                "title": "Expand Core Technical Keywords",
                "description": "Include foundational tooling and frameworks (e.g., PyTorch, Docker, FastAPI, CI/CD, Git) to maximize semantic retrieval recall in RAG pipelines."
            })

        tips.append({
            "priority": "Tip",
            "title": "Highlight RAG & SLM Modern Stack",
            "description": "Modern AI hiring focuses heavily on Small Language Models (SLMs), quantization, dense retrieval, and agentic workflows. Emphasize these in your recent project bullets."
        })

        return tips

slm_reasoner = SLMReasoner()


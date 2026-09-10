import json
import time
import re
import os
from typing import Dict, Any, List, Optional
import httpx
from pydantic import ValidationError

from backend.app.config import settings
from backend.app.models import AnalystReport, InterviewQuestion
from backend.app.logger import log_agent_call, logger
from backend.ai.slm_reasoner import slm_reasoner

SYSTEM_PROMPT = """You are an experienced technical interviewer and engineering lead.
Analyze the candidate's profile against the target job requirements and evidence chunks.
Respond ONLY in valid JSON format with no markdown formatting.

JSON schema:
{
  "executive_summary": "2 concise sentences summarizing technical fit",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "gaps": ["gap 1", "gap 2"],
  "interview_questions": [
    {
      "category": "Core Architecture & Depth",
      "topic": "Python / Framework",
      "question": "Question text",
      "rationale": "Why ask this"
    },
    {
      "category": "Skill Gap Evaluation",
      "topic": "Missing Tooling",
      "question": "Question text",
      "rationale": "Why ask this"
    },
    {
      "category": "System Design & Trade-offs",
      "topic": "Production Considerations",
      "question": "Question text",
      "rationale": "Why ask this"
    }
  ],
  "hiring_recommendation": "Strong Hire"
}
Valid hiring_recommendation values: "Strong Hire", "Hire", "Consider", "Pass".
"""

class AnalystAgent:
    # Generates candidate evaluations, scorecards, and tailored interview questions
    def __init__(self):
        self.grok_base_url = getattr(settings, "GROK_BASE_URL", "https://api.x.ai/v1")
        self.grok_model = getattr(settings, "GROK_MODEL", "grok-2-latest")
        self.timeout = getattr(settings, "LLM_TIMEOUT_SECONDS", 30.0)

    def _get_grok_key(self) -> Optional[str]:
        return getattr(settings, "GROK_API_KEY", None) or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")

    def _call_grok(self, prompt: str, system: str, temperature: float = 0.2) -> Optional[str]:
        # Optional cloud inference via Grok / xAI
        api_key = self._get_grok_key()
        if not api_key or not api_key.strip():
            return None

        # Auto-detect Groq vs xAI if default URL is used
        base_url = self.grok_base_url
        model = self.grok_model
        if api_key.startswith("gsk_") and "api.x.ai" in base_url:
            base_url = "https://api.groq.com/openai/v1"
            if "grok" in model.lower():
                model = "llama-3.3-70b-versatile"

        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload
                )
                if res.status_code == 200:
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "")
                else:
                    logger.warning(f"Grok API status {res.status_code}: {res.text}")
        except Exception as e:
            logger.debug(f"Grok call skipped: {e}")
        return None

    def _summarize_with_slm(self, prompt: str) -> Optional[str]:
        """
        Local in-process Hugging Face SLM, used ONLY for free-text summary
        generation. A 135M-parameter model cannot reliably emit the full JSON
        scorecard schema, so structured fields always come from Grok or the
        deterministic rules — never from the SLM.
        """
        try:
            slm_reasoner._try_load_slm()
            if not slm_reasoner._is_slm_loaded:
                return None

            instruct_prompt = (
                "<|im_start|>system\nYou are a technical recruiting analyst. "
                "Write exactly two concise sentences evaluating the candidate's fit. "
                "Plain text only, no lists, no JSON.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            raw = slm_reasoner._generate_with_slm(instruct_prompt, max_new_tokens=120)
            if raw and len(raw.strip()) > 30:
                sentence = raw.strip().split("\n")[0].strip()
                return sentence if sentence.endswith((".", "!")) else sentence + "."
            return None
        except Exception as e:
            logger.debug(f"SLM summary skipped: {e}")
        return None

    def _clean_json_string(self, text: str) -> str:
        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def generate_analysis(
        self,
        candidate: Dict[str, Any],
        job: Dict[str, Any],
        match_score: float = 75.0,
        evidence_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> AnalystReport:
        # Generates structured evaluation report with strengths, gaps, and interview questions
        start_time = time.time()
        cand_name = candidate.get("name", "Candidate")
        cand_title = candidate.get("title", "Engineer")
        job_title = job.get("title", "Role")
        req_skills = job.get("required_skills", [])
        cand_skills = candidate.get("skills", [])
        years_exp = candidate.get("years_experience")
        years_display = f"{years_exp} yrs exp" if years_exp else "experience length not stated"

        evidence_texts = []
        if evidence_chunks:
            for ec in evidence_chunks[:3]:
                evidence_texts.append(f"- {ec.get('chunk_text', '')}")

        user_prompt = f"""
Candidate Profile:
- Name: {cand_name}
- Title: {cand_title} ({years_display})
- Key Skills: {', '.join(cand_skills)}
- Bio: {candidate.get('bio', '')}
- Resume Evidence:
{chr(10).join(evidence_texts) if evidence_texts else 'Standard credentials.'}

Target Job:
- Position: {job_title} at {job.get('company', 'Company')}
- Required: {', '.join(req_skills)}
- Preferred: {', '.join(job.get('nice_to_have_skills', []))}
- Min Experience: {job.get('min_experience_years', 3)} years
- Description: {job.get('description', '')}

Fit Score: {match_score}%
Generate the JSON evaluation.
"""

        parsed_data = None
        model_used = None

        # Check Grok cloud API first (structured JSON scorecard)
        if self._get_grok_key():
            raw_response = self._call_grok(user_prompt, SYSTEM_PROMPT)
            if raw_response:
                try:
                    cleaned = self._clean_json_string(raw_response)
                    parsed_data = json.loads(cleaned)
                    model_used = f"grok:{self.grok_model}"
                except Exception:
                    pass

        # Fallback to local heuristic synthesis
        if not parsed_data:
            model_used = "local_rules"
            parsed_data = self._generate_fallback_analysis(candidate, job, match_score, evidence_chunks)
            # Local SLM contributes only the free-text summary, never structure
            slm_summary = self._summarize_with_slm(
                f"Candidate: {cand_name}, {cand_title} ({years_display}). "
                f"Applying for: {job_title} at {job.get('company', 'Company')}. "
                f"Fit score: {match_score}%. "
                f"Matched skills: {', '.join(cand_skills[:4]) or 'none listed'}. "
                f"Gaps: {', '.join(parsed_data.get('gaps', [])) or 'none'}."
            )
            if slm_summary:
                parsed_data["executive_summary"] = slm_summary
                model_used = f"hf:{getattr(settings, 'SLM_MODEL_NAME', 'smollm2')}+local_rules"

        hiring_rec = parsed_data.get("hiring_recommendation")
        if not hiring_rec or hiring_rec not in ["Strong Hire", "Hire", "Consider", "Pass"]:
            if match_score >= 85:
                hiring_rec = "Strong Hire"
            elif match_score >= 70:
                hiring_rec = "Hire"
            elif match_score >= 50:
                hiring_rec = "Consider"
            else:
                hiring_rec = "Pass"

        questions_list = []
        for q in parsed_data.get("interview_questions", []):
            if isinstance(q, dict):
                questions_list.append(InterviewQuestion(
                    category=q.get("category", "Technical Competency"),
                    topic=q.get("topic", "System Architecture"),
                    question=q.get("question", "Describe your approach to building scalable systems."),
                    rationale=q.get("rationale", "Validates technical proficiency.")
                ))

        if not questions_list:
            questions_list = self._generate_default_questions(candidate, job)

        strengths = parsed_data.get("strengths") or [f"Strong alignment with {job_title} requirements."]
        gaps = parsed_data.get("gaps") or parsed_data.get("risks") or ["Candidate requires validation during technical screen."]

        report = AnalystReport(
            candidate_id=str(candidate.get("id", "")),
            candidate_name=cand_name,
            job_id=str(job.get("id", "")),
            match_score=match_score,
            executive_summary=parsed_data.get("executive_summary", f"{cand_name} demonstrates strong alignment with the {job_title} position."),
            strengths=strengths,
            gaps=gaps,
            risks=gaps,
            recommendation=f"{hiring_rec} for {job_title}",
            hiring_recommendation=hiring_rec,
            interview_questions=questions_list
        )

        latency_ms = int((time.time() - start_time) * 1000)
        log_agent_call(
            "analyst_agent",
            candidate_id=str(candidate.get("id")),
            job_id=str(job.get("id")),
            model=model_used,
            latency_ms=latency_ms,
            json_valid=True
        )

        return report

    def _generate_fallback_analysis(
        self,
        candidate: Dict[str, Any],
        job: Dict[str, Any],
        match_score: float,
        evidence_chunks: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        # Formulate structured candidate assessment when offline
        cand_name = candidate.get("name", "Candidate")
        job_title = job.get("title", "the role")
        cand_title = candidate.get("title", "Specialist")
        years_exp = candidate.get("years_experience")
        min_years = job.get("min_experience_years", 3)

        cand_skills = set(s.strip().lower() for s in candidate.get("skills", []))
        matched = [s for s in job.get("required_skills", []) if s.strip().lower() in cand_skills]
        missing = [s for s in job.get("required_skills", []) if s.strip().lower() not in cand_skills]

        exp_phrase = f"{years_exp} years of experience as a {cand_title}" if years_exp else f"experience as a {cand_title} (length not stated)"

        if match_score >= 85:
            summary = (
                f"{cand_name} displays strong technical alignment ({match_score}%) with the {job_title} requisition. "
                f"With {exp_phrase} and direct skills in "
                f"{', '.join(matched[:3]) if matched else 'core technologies'}, they meet key hiring criteria."
            )
            rec = "Strong Hire"
        elif match_score >= 70:
            summary = (
                f"{cand_name} is a solid match ({match_score}%) for {job_title}. "
                f"They bring practical experience in {', '.join(matched[:2]) if matched else 'the required stack'}. "
                f"Interview questions should focus on {', '.join(missing[:2]) if missing else 'system architecture'}."
            )
            rec = "Hire"
        elif match_score >= 50:
            summary = (
                f"{cand_name} shows foundational skills in {cand_title}, but presents skill gaps ({match_score}%) for {job_title}. "
                f"Onboarding on {', '.join(missing[:2]) if missing else 'core tools'} would be needed."
            )
            rec = "Consider"
        else:
            summary = (
                f"{cand_name}'s profile has limited overlap ({match_score}%) with the required {job_title} stack. "
                f"Key requirements in {', '.join(missing[:3]) if missing else 'target areas'} are unverified."
            )
            rec = "Pass"

        strengths = []
        if matched:
            strengths.append(f"Demonstrated hands-on experience with: {', '.join(matched[:4])}.")
        if years_exp is not None and years_exp >= min_years:
            strengths.append(f"Meets seniority benchmark with {years_exp} years relevant experience.")
        if evidence_chunks and len(evidence_chunks) > 0:
            strengths.append(f"Resume excerpt: \"{evidence_chunks[0].get('chunk_text', '')[:110]}...\"")
        if not strengths:
            strengths.append("Professional background in related technical disciplines.")

        gaps = []
        if missing:
            gaps.append(f"Needs screening on: {', '.join(missing[:3])}.")
        if years_exp is not None and years_exp < min_years:
            gaps.append(f"Experience level ({years_exp} yrs) is below requested {min_years} yrs.")
        elif years_exp is None:
            gaps.append(f"Total years of experience not stated on resume; verify against the {min_years} yrs requirement.")
        if not gaps:
            gaps.append("No major skill gaps identified from resume review.")

        questions = self._generate_default_questions(candidate, job, matched, missing)

        return {
            "executive_summary": summary,
            "strengths": strengths,
            "gaps": gaps,
            "hiring_recommendation": rec,
            "interview_questions": [q.model_dump() for q in questions]
        }

    def _generate_default_questions(
        self,
        candidate: Dict[str, Any],
        job: Dict[str, Any],
        matched_skills: Optional[List[str]] = None,
        missing_skills: Optional[List[str]] = None
    ) -> List[InterviewQuestion]:
        # Generate targeted interview questions covering strengths and gaps
        questions = []
        if matched_skills and len(matched_skills) > 0:
            s = matched_skills[0]
            questions.append(InterviewQuestion(
                category="Core Architecture & Depth",
                topic=s,
                question=f"Can you walk us through a recent project where you used {s} to solve a key technical challenge?",
                rationale=f"Validates depth and technical experience in candidate strength ({s})."
            ))
        else:
            questions.append(InterviewQuestion(
                category="Core Architecture & Depth",
                topic="System Design",
                question="How do you architect web services and APIs to ensure high availability and responsiveness under load?",
                rationale="Evaluates backend and architectural fundamentals."
            ))

        if missing_skills and len(missing_skills) > 0:
            m = missing_skills[0]
            questions.append(InterviewQuestion(
                category="Skill Gap Evaluation",
                topic=m,
                question=f"This role works with {m}. How would you approach getting up to speed with {m} in production?",
                rationale=f"Checks learning curve and adaptability in {m}."
            ))
        else:
            questions.append(InterviewQuestion(
                category="Skill Gap Evaluation",
                topic="DevOps & Reliability",
                question="What is your workflow for testing, CI/CD automation, and deployment monitoring?",
                rationale="Assesses engineering discipline and testing practices."
            ))

        questions.append(InterviewQuestion(
            category="System Design & Trade-offs",
            topic="Engineering Decision Making",
            question="Describe a situation where you had to balance delivery speed against technical debt or architecture quality.",
            rationale="Assesses decision-making and practical engineering trade-offs."
        ))

        return questions

    def generate_resume_tips(self, candidate: Dict[str, Any], target_job: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        # Actionable resume optimization advice
        tips = []
        cand_skills = set(s.strip().lower() for s in candidate.get("skills", []))

        if target_job:
            job_req = target_job.get("required_skills", [])
            missing = [s for s in job_req if s.strip().lower() not in cand_skills]
            if missing:
                tips.append({
                    "priority": "High",
                    "title": f"Highlight experience with {', '.join(missing[:2])}",
                    "description": f"The position '{target_job.get('title')}' emphasizes {', '.join(missing[:3])}. If you have side projects or related experience with these, feature them prominently."
                })

        raw_text = candidate.get("raw_cv_text") or candidate.get("bio") or ""
        if "%" not in raw_text and "reduced" not in raw_text.lower() and "improved" not in raw_text.lower():
            tips.append({
                "priority": "Medium",
                "title": "Quantify Impact with Measurable Metrics",
                "description": "Recruiters and hiring managers look for measurable results (e.g. 'reduced latency by 40%', 'served 10,000+ daily active users', 'decreased error rate')."
            })

        if len(cand_skills) < 6:
            tips.append({
                "priority": "Medium",
                "title": "Expand Core Technical Keywords",
                "description": "List relevant frameworks, databases, and tooling (e.g., Python, Docker, React, FastAPI, SQL, Git) to improve search discoverability."
            })

        tips.append({
            "priority": "Tip",
            "title": "Keep Project Accomplishments Front and Center",
            "description": "Highlight concrete technical achievements, architecture decisions, and business impact in your recent role descriptions."
        })

        return tips

analyst_agent = AnalystAgent()

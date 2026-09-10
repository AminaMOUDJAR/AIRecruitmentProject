import re
import time
from typing import Dict, Any, List, Optional
import pymupdf
from backend.app.logger import log_agent_call, logger

# Common technical terms and frameworks to scan for
COMMON_TECH_SKILLS = [
    "Python", "PyTorch", "TensorFlow", "Hugging Face", "LangChain", "RAG", "vLLM",
    "FastAPI", "Flask", "Django", "Docker", "Kubernetes", "AWS", "GCP", "Azure",
    "SQL", "PostgreSQL", "SQLite", "MySQL", "MongoDB", "Redis", "Qdrant", "Chroma",
    "React", "JavaScript", "TypeScript", "Node.js", "Next.js", "Tailwind CSS", "Vue.js",
    "PHP", "Laravel", "Power BI", "DAX", "ETL", "SSIS", "Business Intelligence",
    "ITIL", "Active Directory", "Network Security", "ERP Systems",
    "MLOps", "NLP", "Computer Vision", "Scikit-Learn", "Pandas", "NumPy",
    "Git", "CI/CD", "Linux", "Terraform", "Prometheus", "Grafana", "GraphQL",
    "REST APIs", "Microservices", "Deep Learning", "Transformers", "LoRA", "Fine-tuning"
]

class ParserAgent:
    # Extracts text from PDFs or raw text and parses candidate fields
    def __init__(self):
        self._nlp = None
        self._load_spacy()

    def _load_spacy(self):
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"spaCy model not loaded ({e}), using regex-based extraction")
            self._nlp = None

    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        # Read text from PDF stream
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        return "\n".join(pages_text).strip()

    def parse(self, raw_bytes: Optional[bytes] = None, raw_text: Optional[str] = None, filename: str = "") -> Dict[str, Any]:
        start_time = time.time()
        text = ""

        if raw_bytes:
            if filename.lower().endswith(".pdf") or raw_bytes.startswith(b"%PDF"):
                try:
                    text = self.extract_text_from_pdf(raw_bytes)
                except Exception as e:
                    logger.error(f"Failed to read PDF stream: {e}")
                    text = raw_bytes.decode("utf-8", errors="ignore")
            else:
                try:
                    text = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    text = raw_bytes.decode("latin-1", errors="ignore")
        elif raw_text:
            text = raw_text.strip()

        if not text:
            text = "No CV text provided."

        # Extract email (None if absent — never fabricate contact data)
        email_match = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
        extracted_email = email_match.group(0) if email_match else None

        # Extract years of experience (e.g. "8+ years", "over 7 years of experience")
        years_experience = None
        for pattern in [
            r'(\d{1,2})\s*\+?\s*years?(?!\s*\b\d)',          # "8 years", "8+ years"
            r'(?:over|more than|above)\s*(\d{1,2})\s*years?',  # "over 7 years"
        ]:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                years_experience = min(int(m.group(1)), 45)
                break

        # Named entity extraction
        extracted_name = None
        extracted_locations = []
        extracted_orgs = []

        if self._nlp:
            doc = self._nlp(text[:1500])
            for ent in doc.ents:
                if ent.label_ == "PERSON" and not extracted_name and len(ent.text.split()) <= 4:
                    clean_name = ent.text.strip().replace("\n", " ")
                    if len(clean_name) > 2 and not any(kw in clean_name.lower() for kw in ["curriculum", "resume", "page", "email"]):
                        extracted_name = clean_name
                elif ent.label_ == "GPE":
                    extracted_locations.append(ent.text.strip())
                elif ent.label_ == "ORG":
                    extracted_orgs.append(ent.text.strip())

        if not extracted_name:
            # Fallback: check first readable line
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if lines:
                first_line = lines[0]
                first_line = re.sub(r'[-–|•].*$', '', first_line).strip()
                if 2 < len(first_line) < 40 and not any(kw in first_line.lower() for kw in ["resume", "cv", "page", "profile"]):
                    extracted_name = first_line

        if not extracted_name:
            extracted_name = "Candidate" if not filename else filename.replace(".pdf", "").replace(".txt", "").replace("_", " ").title()

        # Skill keyword matching
        text_lower = text.lower()
        detected_skills = []
        for skill in COMMON_TECH_SKILLS:
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                detected_skills.append(skill)

        # Infer job title
        title_candidates = [
            "AI Engineer", "ML Engineer", "Data Scientist", "Data Analyst",
            "Full Stack Developer", "Frontend Developer", "Web Developer",
            "IT Manager", "Tech Lead", "BI Analyst", "DevOps Engineer", "Software Engineer"
        ]
        detected_title = "Software Engineer"
        for tc in title_candidates:
            if re.search(r'\b' + re.escape(tc.lower()) + r'\b', text_lower):
                detected_title = tc
                break

        # Experience entries are only created from detected organizations;
        # no synthetic employment history
        experiences = []
        for org in list(dict.fromkeys(extracted_orgs))[:2]:
            experiences.append({
                "role": detected_title,
                "company": org,
                "duration": "Not stated",
                "description": f"Worked on tech initiatives at {org}."
            })

        latency_ms = int((time.time() - start_time) * 1000)
        log_agent_call(
            "parser_agent",
            name=extracted_name,
            skills_count=len(detected_skills),
            latency_ms=latency_ms
        )

        return {
            "name": extracted_name,
            "title": detected_title,
            "email": extracted_email,
            "location": extracted_locations[0] if extracted_locations else "Not specified",
            "skills": detected_skills,
            "years_experience": years_experience,
            "bio": f"{detected_title}" + (f" with experience in {', '.join(detected_skills[:4])}." if detected_skills else "."),
            "experience": experiences,
            "education": [],  # no reliable education extractor; left empty rather than fabricated
            "raw_cv_text": text,
            "parsed_entities": {
                "organizations": list(dict.fromkeys(extracted_orgs))[:5],
                "locations": list(dict.fromkeys(extracted_locations))[:3],
                "skills": detected_skills
            }
        }

parser_agent = ParserAgent()

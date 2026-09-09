import uuid
from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from backend.app.agents.parser import parser_agent
from backend.app.agents.embedder import embedder_agent
from backend.app.agents.matcher import matcher_agent
from backend.app.agents.analyst import analyst_agent
from backend.app.logger import logger

class RecruitmentState(TypedDict, total=False):
    # Ingestion inputs
    raw_bytes: Optional[bytes]
    raw_text: Optional[str]
    filename: Optional[str]
    candidate_id: Optional[str]
    
    # Candidate profile
    profile: Dict[str, Any]
    indexed_chunks: List[Dict[str, Any]]
    
    # Matching inputs & outputs
    job: Dict[str, Any]
    matches: List[Dict[str, Any]]
    
    # Analysis output
    analysis: Dict[str, Any]

# Pipeline graph steps
def parser_node(state: RecruitmentState) -> Dict[str, Any]:
    # Extract text and details from the resume
    profile = parser_agent.parse(
        raw_bytes=state.get("raw_bytes"),
        raw_text=state.get("raw_text"),
        filename=state.get("filename", "")
    )
    if state.get("candidate_id"):
        profile["id"] = state["candidate_id"]
    return {"profile": profile}

def embedder_node(state: RecruitmentState) -> Dict[str, Any]:
    # Create chunks and index in vector memory
    profile = state.get("profile", {})
    candidate_id = profile.get("id") or state.get("candidate_id") or str(uuid.uuid4())
    indexed_chunks = embedder_agent.chunk_and_index_candidate(str(candidate_id), profile)
    return {"indexed_chunks": indexed_chunks}

def matcher_node(state: RecruitmentState) -> Dict[str, Any]:
    # Match candidate against selected job
    job = state.get("job")
    if not job:
        return {"matches": []}
    
    profile = state.get("profile")
    candidates = [profile] if profile else []
    matches = matcher_agent.match_candidates_for_job(job, candidates)
    return {"matches": matches}

def analyst_node(state: RecruitmentState) -> Dict[str, Any]:
    # Generate scorecard and interview questions
    profile = state.get("profile", {})
    job = state.get("job", {})
    matches = state.get("matches", [])
    
    match_score = matches[0]["match_score"] if matches else 75.0
    evidence = matches[0].get("evidence_chunks", []) if matches else []
    
    report = analyst_agent.generate_analysis(
        candidate=profile,
        job=job,
        match_score=match_score,
        evidence_chunks=evidence
    )
    return {"analysis": report.model_dump()}

# Assemble workflow graph
workflow = StateGraph(RecruitmentState)

workflow.add_node("parser", parser_node)
workflow.add_node("embedder", embedder_node)
workflow.add_node("matcher", matcher_node)
workflow.add_node("analyst", analyst_node)

workflow.set_entry_point("parser")
workflow.add_edge("parser", "embedder")
workflow.add_edge("embedder", "matcher")
workflow.add_edge("matcher", "analyst")
workflow.add_edge("analyst", END)

recruitment_pipeline = workflow.compile()

def process_cv_ingestion(
    raw_bytes: Optional[bytes] = None,
    raw_text: Optional[str] = None,
    filename: str = "",
    candidate_id: Optional[str] = None
) -> Dict[str, Any]:
    # Helper to parse and embed an uploaded resume
    initial_state: RecruitmentState = {
        "raw_bytes": raw_bytes,
        "raw_text": raw_text,
        "filename": filename,
        "candidate_id": candidate_id
    }
    
    parsed = parser_node(initial_state)
    profile = parsed["profile"]
    if candidate_id:
        profile["id"] = candidate_id
    
    embed_res = embedder_node({"profile": profile, "candidate_id": candidate_id})
    return {
        "profile": profile,
        "indexed_chunks": embed_res["indexed_chunks"]
    }

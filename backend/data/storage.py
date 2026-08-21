import json
import os
import uuid
from typing import List, Dict, Any, Optional

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CV_DATASET_PATH = os.path.join(DATA_DIR, "cv_dataset.json")
JOBS_DATASET_PATH = os.path.join(DATA_DIR, "jobs_dataset.json")
ACTIVE_CV_PATH = os.path.join(DATA_DIR, "candidates_store.json")
ACTIVE_JOBS_PATH = os.path.join(DATA_DIR, "jobs_store.json")


class DataStore:
    def __init__(self):
        self._ensure_storage_initialized()

    def _ensure_storage_initialized(self):
        # Initialize candidates store if not exists
        if not os.path.exists(ACTIVE_CV_PATH):
            if os.path.exists(CV_DATASET_PATH):
                with open(CV_DATASET_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = []
            self.save_candidates(data)

        # Initialize jobs store if not exists
        if not os.path.exists(ACTIVE_JOBS_PATH):
            if os.path.exists(JOBS_DATASET_PATH):
                with open(JOBS_DATASET_PATH, "r", encoding="utf-8") as f:
                    jobs = json.load(f)
            else:
                jobs = []
            self.save_jobs(jobs)

    # --- Candidates CRUD ---
    def get_all_candidates(self) -> List[Dict[str, Any]]:
        try:
            with open(ACTIVE_CV_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def save_candidates(self, candidates: List[Dict[str, Any]]) -> None:
        with open(ACTIVE_CV_PATH, "w", encoding="utf-8") as f:
            json.dump(candidates, f, indent=2, ensure_ascii=False)

    def get_candidate_by_id(self, cand_id: str) -> Optional[Dict[str, Any]]:
        candidates = self.get_all_candidates()
        for cand in candidates:
            if cand.get("id") == cand_id:
                return cand
        return None

    def upsert_candidate(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        candidates = self.get_all_candidates()
        cand_id = candidate_data.get("id") or f"cand-{uuid.uuid4().hex[:6]}"
        candidate_data["id"] = cand_id

        # Update if exists, else append
        existing_index = next((i for i, c in enumerate(candidates) if c.get("id") == cand_id), None)
        if existing_index is not None:
            candidates[existing_index] = candidate_data
        else:
            candidates.append(candidate_data)

        self.save_candidates(candidates)
        return candidate_data

    def delete_candidate(self, cand_id: str) -> bool:
        candidates = self.get_all_candidates()
        new_candidates = [c for c in candidates if c.get("id") != cand_id]
        if len(new_candidates) != len(candidates):
            self.save_candidates(new_candidates)
            return True
        return False

    # --- Jobs CRUD ---
    def get_all_jobs(self) -> List[Dict[str, Any]]:
        try:
            with open(ACTIVE_JOBS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def save_jobs(self, jobs: List[Dict[str, Any]]) -> None:
        with open(ACTIVE_JOBS_PATH, "w", encoding="utf-8") as f:
            json.dump(jobs, f, indent=2, ensure_ascii=False)

    def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        jobs = self.get_all_jobs()
        for job in jobs:
            if job.get("id") == job_id:
                return job
        return None

    def add_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        jobs = self.get_all_jobs()
        job_id = job_data.get("id") or f"job-{uuid.uuid4().hex[:6]}"
        job_data["id"] = job_id
        jobs.insert(0, job_data)
        self.save_jobs(jobs)
        return job_data

    def reset_to_defaults(self):
        """Reset storage to seed datasets."""
        if os.path.exists(CV_DATASET_PATH):
            with open(CV_DATASET_PATH, "r", encoding="utf-8") as f:
                self.save_candidates(json.load(f))
        if os.path.exists(JOBS_DATASET_PATH):
            with open(JOBS_DATASET_PATH, "r", encoding="utf-8") as f:
                self.save_jobs(json.load(f))


store = DataStore()

import time
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from backend.app.config import settings
from backend.app.logger import log_agent_call, logger
from backend.ai.embeddings import embedding_engine

class EmbedderAgent:
    # Handles text chunking, vector indexing, and in-memory vector search.
    # Embedding computation itself is delegated to the shared embedding_engine
    # singleton so model loading and the statistical fallback exist only once.
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "- ", ". ", " "]
        )

        # Use in-memory Qdrant instance for fast local search
        self.qdrant = QdrantClient(":memory:")
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self._setup_collection()

    @property
    def _fallback_mode(self) -> bool:
        return embedding_engine._fallback_mode

    def _setup_collection(self):
        try:
            collections = self.qdrant.get_collections().collections
            collection_names = [c.name for c in collections]
            if self.collection_name not in collection_names:
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=settings.VECTOR_DIMENSION, distance=Distance.COSINE)
                )
                logger.info(f"Initialized Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error setting up Qdrant collection: {e}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return embedding_engine.embed_texts(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return embedding_engine.embed_query(text)

    def chunk_and_index_candidate(self, candidate_id: str, profile_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Break candidate profile into searchable chunks and store in Qdrant
        start_time = time.time()
        chunks_to_index: List[Dict[str, Any]] = []

        # Skills chunk
        if profile_dict.get("skills"):
            skills_text = f"Candidate Skills: {', '.join(profile_dict['skills'])}"
            chunks_to_index.append({
                "chunk_text": skills_text,
                "chunk_type": "skills"
            })

        # Experience chunks
        if profile_dict.get("experience"):
            for exp in profile_dict["experience"]:
                exp_text = f"Experience: {exp.get('role', '')} at {exp.get('company', '')} ({exp.get('duration', '')}): {exp.get('description', '')}"
                chunks_to_index.append({
                    "chunk_text": exp_text,
                    "chunk_type": "experience"
                })

        # Education chunks
        if profile_dict.get("education"):
            for edu in profile_dict["education"]:
                edu_text = f"Education: {edu.get('degree', '')} from {edu.get('institution', '')} ({edu.get('year', '')})"
                chunks_to_index.append({
                    "chunk_text": edu_text,
                    "chunk_type": "education"
                })

        # Raw CV text chunks
        raw_text = profile_dict.get("raw_cv_text") or profile_dict.get("bio") or ""
        if raw_text:
            split_chunks = self.text_splitter.split_text(raw_text)
            for sc in split_chunks:
                if len(sc.strip()) > 20:
                    chunks_to_index.append({
                        "chunk_text": sc.strip(),
                        "chunk_type": "experience"
                    })

        if not chunks_to_index:
            chunks_to_index.append({
                "chunk_text": f"Candidate {profile_dict.get('name', 'Professional')} - {profile_dict.get('title', 'Engineer')}",
                "chunk_type": "skills"
            })

        texts = [c["chunk_text"] for c in chunks_to_index]
        vectors = self.embed_texts(texts)

        points = []
        indexed_records = []
        for i, chunk in enumerate(chunks_to_index):
            point_id = str(uuid.uuid4())
            vector_list = vectors[i].tolist()
            payload = {
                "candidate_id": str(candidate_id),
                "chunk_type": chunk["chunk_type"],
                "chunk_text": chunk["chunk_text"]
            }
            points.append(PointStruct(
                id=point_id,
                vector=vector_list,
                payload=payload
            ))
            indexed_records.append({
                "vector_id": point_id,
                "candidate_id": str(candidate_id),
                "chunk_type": chunk["chunk_type"],
                "chunk_text": chunk["chunk_text"]
            })

        # Remove any previous vectors for this candidate so re-indexing
        # (profile updates, server restart) never leaves stale duplicate chunks
        self.delete_candidate_chunks(str(candidate_id))

        if points:
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points
            )

        latency_ms = int((time.time() - start_time) * 1000)
        log_agent_call(
            "embedder_agent",
            candidate_id=str(candidate_id),
            chunks_indexed=len(points),
            latency_ms=latency_ms
        )

        return indexed_records

    def delete_candidate_chunks(self, candidate_id: str):
        # Drop all indexed vectors belonging to a candidate (Qdrant point ids
        # are per-chunk UUIDs, so deletion must go through the payload filter)
        try:
            self.qdrant.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="candidate_id", match=MatchValue(value=str(candidate_id)))]
                )
            )
        except Exception as e:
            logger.error(f"Error deleting vectors for candidate {candidate_id}: {e}")

    def search_similar_chunks(self, query_text: str, top_k: int = 50, candidate_id: Optional[str] = None) -> List[Dict[str, Any]]:
        # Query top matching chunks from Qdrant
        query_vector = self.embed_query(query_text).tolist()
        
        query_filter = None
        if candidate_id:
            query_filter = Filter(
                must=[FieldCondition(key="candidate_id", match=MatchValue(value=str(candidate_id)))]
            )

        try:
            if hasattr(self.qdrant, "query_points"):
                search_results = self.qdrant.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k
                ).points
            else:
                search_results = self.qdrant.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=top_k
                )
        except Exception:
            search_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k
            )

        results = []
        for hit in search_results:
            results.append({
                "vector_id": str(hit.id),
                "score": float(hit.score),
                "candidate_id": hit.payload.get("candidate_id"),
                "chunk_type": hit.payload.get("chunk_type"),
                "chunk_text": hit.payload.get("chunk_text")
            })
        return results

embedder_agent = EmbedderAgent()

import torch
import numpy as np
from typing import List, Union, Optional
import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# Canonical vocabulary corpus to ensure the TF-IDF fallback maintains a stable, identical vector space
REFERENCE_VOCAB_CORPUS = [
    "python java c++ c# golang rust typescript javascript php ruby swift kotlin scala",
    "pytorch tensorflow keras scikit-learn transformers huggingface llm slm rag langchain langgraph",
    "docker kubernetes aws gcp azure terraform linux git ci cd devops microservices api restful",
    "fastapi flask django spring react angular vue nodejs express nextjs html css tailwind",
    "sql postgresql mysql sqlite mongodb redis elasticsearch qdrant pinecone vector embeddings",
    "machine learning deep learning nlp computer vision data engineering analytics distributed systems",
    "software engineer architect developer senior junior lead principal fullstack backend frontend",
    "bachelor master phd computer science mathematics electrical engineering information systems",
    "performance optimization scalability latency throughput debugging testing unit integration agile"
]

class EmbeddingEngine:
    """
    PyTorch & HuggingFace Sentence-Transformers Embedding Engine.
    Computes dense semantic embeddings for Job Descriptions and Candidate Resumes.
    Features a persistent, stable vector-space fallback when transformers are offline.
    """
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._fallback_mode = False
        self._initialized = False
        self._vectorizer = None

    def _init_fallback_vectorizer(self):
        """Initializes a persistent, stable TF-IDF vectorizer fit against a canonical technical corpus."""
        if self._vectorizer is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=384,
                stop_words="english"
            )
            self._vectorizer.fit(REFERENCE_VOCAB_CORPUS)
            logger.info("Initialized persistent reference TF-IDF vectorizer space (384 features).")

    def _ensure_initialized(self):
        if self._initialized:
            return
        self._initialized = True
        try:
            logger.info(f"Attempting to load HuggingFace SentenceTransformer '{self.model_name}'...")
            from sentence_transformers import SentenceTransformer
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("SentenceTransformer loaded successfully.")
        except Exception as e:
            logger.info(f"SentenceTransformer not cached or download offline ({e}). Using persistent statistical vector space.")
            self._fallback_mode = True
            self._init_fallback_vectorizer()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generates dense vector embeddings for a list of text strings.
        Returns a numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        self._ensure_initialized()

        if not self._fallback_mode and self._model is not None:
            try:
                embeddings = self._model.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                if isinstance(embeddings, torch.Tensor):
                    return embeddings.cpu().numpy()
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                logger.error(f"Error during HF embedding inference: {e}. Falling back to persistent statistical vectorizer.")
                self._fallback_mode = True

        # Stable, persistent statistical vector embedding fallback
        self._init_fallback_vectorizer()
        try:
            # Transform against pre-fitted canonical vector space — NEVER refit per call
            tfidf_matrix = self._vectorizer.transform(texts).toarray()
            # Pad to 384 dimensions if vocabulary yielded fewer features
            if tfidf_matrix.shape[1] < 384:
                pad = np.zeros((tfidf_matrix.shape[0], 384 - tfidf_matrix.shape[1]), dtype=np.float32)
                tfidf_matrix = np.hstack([tfidf_matrix, pad])
            elif tfidf_matrix.shape[1] > 384:
                tfidf_matrix = tfidf_matrix[:, :384]

            # Normalize vectors for meaningful cosine similarity
            norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (tfidf_matrix / norms).astype(np.float32)
        except Exception as e:
            logger.critical(f"Critical error in fallback vectorizer transform: {e}")
            raise RuntimeError(f"Embedding transform failed: {e}")

    def embed_query(self, text: str) -> np.ndarray:
        """Embeds a single query or job description using the exact same vector space."""
        embeddings = self.embed_texts([text])
        return embeddings[0] if len(embeddings) > 0 else np.zeros(384, dtype=np.float32)

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Computes cosine similarity between two 1D or 2D vectors."""
        a = np.asarray(vec_a)
        b = np.asarray(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def batch_cosine_similarity(query_vec: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
        """
        Computes cosine similarities between one query vector and multiple document vectors using PyTorch/NumPy.
        """
        q = torch.tensor(query_vec, dtype=torch.float32)
        docs = torch.tensor(doc_vectors, dtype=torch.float32)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if docs.ndim == 1:
            docs = docs.unsqueeze(0)

        # Normalize
        q_norm = torch.nn.functional.normalize(q, p=2, dim=1)
        docs_norm = torch.nn.functional.normalize(docs, p=2, dim=1)

        sims = torch.mm(q_norm, docs_norm.t()).squeeze(0)
        return sims.cpu().numpy()

# Global singleton instance for efficient memory usage
embedding_engine = EmbeddingEngine()


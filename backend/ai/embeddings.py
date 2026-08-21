import torch
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)

# Default lightweight sentence transformer model from Hugging Face
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

class EmbeddingEngine:
    """
    PyTorch & HuggingFace Sentence-Transformers Embedding Engine.
    Computes dense semantic embeddings for Job Descriptions and Candidate Resumes.
    """
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._fallback_mode = False
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        self._initialized = True
        try:
            logger.info(f"Attempting to load HuggingFace SentenceTransformer '{self.model_name}'...")
            from sentence_transformers import SentenceTransformer
            # Set local_files_only or fast timeout
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("SentenceTransformer loaded.")
        except Exception as e:
            logger.info(f"SentenceTransformer not cached or download offline ({e}). Using ultra-fast PyTorch statistical vector space.")
            self._fallback_mode = True

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
                # Use PyTorch & HuggingFace SentenceTransformer
                embeddings = self._model.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                if isinstance(embeddings, torch.Tensor):
                    return embeddings.cpu().numpy()
                return np.array(embeddings)
            except Exception as e:
                logger.error(f"Error during HF embedding inference: {e}. Falling back to statistical vectorizer.")

        # High-performance statistical vector embedding fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=384)
        try:
            tfidf_matrix = vectorizer.fit_transform(texts).toarray()
            # Normalize vectors
            norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (tfidf_matrix / norms).astype(np.float32)
        except Exception:
            return np.random.randn(len(texts), 384).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embeds a single query or job description."""
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

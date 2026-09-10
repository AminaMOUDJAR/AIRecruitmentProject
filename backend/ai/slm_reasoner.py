import os
import logging
from typing import Optional
import torch

from backend.app.config import settings

logger = logging.getLogger(__name__)

# Preferred lightweight Small Language Model (SLM) from Hugging Face
DEFAULT_SLM_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

class SLMReasoner:
    """
    Thin loader/generator for a local Hugging Face Small Language Model.

    Deliberately contains NO domain logic: scoring, scorecards, interview
    questions, and resume tips live in matcher/rag_engine and analyst_agent.
    The SLM is only used for short free-text generation (e.g. executive
    summaries) — a 135M model cannot reliably produce structured JSON.
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
            logger.info(f"SLM local weights not pre-cached ({e}). Skipping local SLM tier.")
            self._is_slm_loaded = False

    def _generate_with_slm(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Executes generative inference with the loaded HuggingFace SLM."""
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

slm_reasoner = SLMReasoner()

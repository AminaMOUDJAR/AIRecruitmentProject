# Hugging Face Spaces Entrypoint (Gradio SDK compatible)
import os
import sys
import uvicorn

# ZeroGPU initialization hook if hosted on Hugging Face ZeroGPU
try:
    import spaces
    @spaces.GPU
    def _init_zero_gpu():
        return True
    _init_zero_gpu()
except Exception:
    pass

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app.main import app

if __name__ == "__main__":
    # Hugging Face Spaces listens on port 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=port)

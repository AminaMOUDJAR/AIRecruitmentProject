"""
TalentMatch AI - Application Launcher
Runs the FastAPI Backend & Serves the Modern Glassmorphic Frontend
"""
import sys
import os
import uvicorn
import webbrowser
import threading
import time

def open_browser():
    time.sleep(1.5)
    url = "http://127.0.0.1:8000"
    print(f"\n🚀 Opening TalentMatch AI in browser: {url}\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass

if __name__ == "__main__":
    # Ensure current directory is in sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("=" * 70)
    print(" 🎯 TalentMatch AI | Applied AI Recruiting & Resume Matching Platform")
    print(" Powered by SLMs, Dense RAG, HuggingFace, PyTorch & LangChain")
    print("=" * 70)
    print(" Server starting on: http://127.0.0.1:8000")
    print(" API Documentation:  http://127.0.0.1:8000/docs")
    print("=" * 70)

    # Automatically launch browser in a background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Run FastAPI app
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)

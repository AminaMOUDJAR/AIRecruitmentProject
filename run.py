# TalentMatch AI - Local Application Server
import sys
import os
import uvicorn
import webbrowser
import threading
import time

def open_browser():
    # Give the server a second to start up before launching browser
    time.sleep(1.2)
    url = "http://127.0.0.1:8000"
    print(f"\nOpening TalentMatch AI in browser: {url}\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("=" * 70)
    print(" TalentMatch AI | Intelligent Recruitment & Matching Engine")
    print(" Tech: FastAPI + SQLModel + Qdrant in-memory + NLP Transformers")
    print("=" * 70)
    print(" Web UI:           http://127.0.0.1:8000")
    print(" API Docs:         http://127.0.0.1:8000/docs")
    print(" Health Status:    http://127.0.0.1:8000/api/health")
    print("=" * 70)

    # Launch default web browser in the background
    threading.Thread(target=open_browser, daemon=True).start()

    # Run FastAPI app
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=False)

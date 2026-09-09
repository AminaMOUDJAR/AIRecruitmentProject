import logging
import os
import json
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from backend.app.config import settings

os.makedirs(settings.DATA_DIR, exist_ok=True)

logger = logging.getLogger("talentmatch")
logger.setLevel(logging.INFO)

# Standard console logging
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Keep up to 3 log rotations (5MB each)
try:
    file_handler = RotatingFileHandler(
        settings.LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    file_formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
except Exception:
    pass

def log_agent_call(agent_name: str, **kwargs):
    # Record agent operations in structured JSON
    payload = {
        "event": "agent_execution",
        "agent": agent_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **kwargs
    }
    logger.info(json.dumps(payload))

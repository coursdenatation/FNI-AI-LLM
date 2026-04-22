"""
Phase 1.5 - Chat Logger
Saves and loads conversation history as JSON
"""

import json
import os
from datetime import datetime

LOG_DIR = "data/chat_logs"
os.makedirs(LOG_DIR, exist_ok=True)


def save_conversation(messages, filename=None):
    if filename is None:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    path = os.path.join(LOG_DIR, filename)
    with open(path, "w") as f:
        json.dump({"timestamp": str(datetime.now()), "messages": messages}, f, indent=2)
    return path


def load_conversation(filename):
    path = os.path.join(LOG_DIR, filename)
    with open(path) as f:
        return json.load(f)


def list_conversations():
    return [f for f in os.listdir(LOG_DIR) if f.endswith(".json")]

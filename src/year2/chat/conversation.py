"""
Phase 2.5 - Conversation Manager
Stores history, manages context window, saves/loads sessions
"""

import json
import os
from datetime import datetime

LOG_DIR = "data/chat_logs"
os.makedirs(LOG_DIR, exist_ok=True)


class Conversation:
    def __init__(self, model_name="transformer-v0", language="english",
                 context_window=5):
        self.model_name = model_name
        self.language = language
        self.context_window = context_window
        self.messages = []
        self.created_at = str(datetime.now())

    def add(self, role, text):
        self.messages.append({
            "role": role,
            "text": text,
            "timestamp": str(datetime.now())
        })

    def get_context(self):
        """Return last N user messages as context string"""
        recent = [m["text"] for m in self.messages[-self.context_window:]
                  if m["role"] == "user"]
        return " ".join(recent)

    def save(self, filename=None):
        if filename is None:
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(LOG_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "model": self.model_name,
                "language": self.language,
                "created_at": self.created_at,
                "messages": self.messages
            }, f, indent=2, ensure_ascii=False)
        return path

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.model_name = data["model"]
        self.language   = data["language"]
        self.created_at = data["created_at"]
        self.messages   = data["messages"]

    def display(self):
        print(f"\n--- Conversation ({self.language}, {self.model_name}) ---")
        for m in self.messages:
            role = "You" if m["role"] == "user" else "FNI-LLM"
            print(f"{role}: {m['text']}")

    def clear(self):
        self.messages = []

    def __repr__(self):
        return (f"Conversation(messages={len(self.messages)}, "
                f"language={self.language})")

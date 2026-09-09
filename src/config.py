"""Configuration module - loads environment variables safely."""

import os

from dotenv import load_dotenv

load_dotenv()

# API Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Storage
CHROMA_PATH = os.getenv("CHROMA_PATH", "./storage/chroma")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./storage/decisions.db")

# System Parameters
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))


def validate_config():
    """Check that required configuration is present."""
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return True

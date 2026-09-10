"""
Configuration module — loads environment variables safely.

Design decisions:
    1. We support BOTH OpenRouter (primary, per pack requirements) and Groq (fallback).
       Why? OpenRouter is the pack's default provider. Groq was used during early
       development. Supporting both via LangChain's ChatOpenAI means switching
       providers is a one-line .env change — no code changes needed.
    2. The LLM is initialized once here and imported by classify/generate.
       Why? Creating a new ChatOpenAI per call wastes connection setup time.
       A shared instance reuses the underlying HTTP session.
    3. We use LangChain's ChatOpenAI, not the raw OpenAI SDK.
       Why? LangChain provides a consistent interface across providers.
       ChatOpenAI works with any OpenAI-compatible API (OpenRouter, Groq,
       Together, Ollama) by changing the base_url.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API Configuration — supports OpenRouter (primary) and Groq (fallback)
# ---------------------------------------------------------------------------

# Primary: OpenRouter (per capstone pack requirements)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Fallback: Groq (used during early development)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Model and embedding configuration
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/llama-3.1-8b-instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Storage
CHROMA_PATH = os.getenv("CHROMA_PATH", "./storage/chroma")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./storage/decisions.db")

# System Parameters
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))


def _resolve_api_config() -> tuple[str, str]:
    """
    Determine which API key and base URL to use.

    Priority: OpenRouter (pack default) > Groq (development fallback).

    Returns:
        Tuple of (api_key, base_url)

    Raises:
        ValueError: If neither API key is set
    """
    if OPENROUTER_API_KEY:
        return OPENROUTER_API_KEY, OPENROUTER_BASE_URL
    if GROQ_API_KEY:
        return GROQ_API_KEY, GROQ_BASE_URL
    raise ValueError(
        "No API key configured. Set OPENROUTER_API_KEY (recommended) "
        "or GROQ_API_KEY in your .env file. "
        "See .env.example for the template."
    )


def validate_config():
    """Check that required configuration is present."""
    _resolve_api_config()  # Raises if no key is set
    return True


def get_llm():
    """
    Create a LangChain ChatOpenAI instance for the configured provider.

    Uses ChatOpenAI which is compatible with any OpenAI-compatible API
    (OpenRouter, Groq, Together, Ollama, etc.) via the base_url parameter.

    Returns:
        A ChatOpenAI instance ready for use in classify and generate stages.
    """
    from langchain_openai import ChatOpenAI

    api_key, base_url = _resolve_api_config()

    return ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0.1,        # Default; overridden per-call where needed
        max_retries=0,          # We handle retries in the evaluation harness
    )

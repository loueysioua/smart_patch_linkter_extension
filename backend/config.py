"""
Centralised configuration for the SmartPatch backend.

All settings are read from environment variables (populated via .env).
Import this module instead of hard-coding values in app.py or elsewhere.
"""

import os
from dotenv import load_dotenv

# Load .env file from the backend directory (no-op if already loaded)
load_dotenv()

# ---------------------------------------------------------------------------
# API server
# ---------------------------------------------------------------------------
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "5000"))
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Dataset / project settings
# ---------------------------------------------------------------------------
# Root directory that contains one sub-folder per project with all_candidates.csv
DATA_DIR: str = os.getenv("DATA_DIR", "datasets")

# Comma-separated list of project keys to load at startup
# e.g. PROJECTS=onap,qt,android
PROJECTS: list[str] = [p.strip() for p in os.getenv("PROJECTS", "onap").split(",") if p.strip()]

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_WINDOW_DAYS: int = int(os.getenv("DEFAULT_WINDOW_DAYS", "14"))

# Retrieval strategy: "multi_query" | "hybrid" | "file_boost"
RETRIEVAL_STRATEGY: str = os.getenv("RETRIEVAL_STRATEGY", "multi_query")

# ---------------------------------------------------------------------------
# LLM API keys (optional — only needed for llm_rerank script)
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")


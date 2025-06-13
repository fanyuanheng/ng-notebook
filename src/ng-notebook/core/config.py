import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
UPLOAD_DIR = BASE_DIR / "uploads"

# Create directories if they don't exist
CHROMA_DB_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# LLM Configuration
LLM_MODEL = "llama2"
EMBEDDING_MODEL = "llama2"

# Document Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Frontend Configuration
FRONTEND_HOST = "localhost"
FRONTEND_PORT = 8501 
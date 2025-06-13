import os
from pathlib import Path
import logging

# Configure logging
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
SQLITE_DB_DIR = BASE_DIR / "sqlite_db"
UPLOAD_DIR = BASE_DIR / "uploads"

# Create directories if they don't exist
CHROMA_DB_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
SQLITE_DB_DIR.mkdir(exist_ok=True)

# LLM Configuration
LLM_MODEL = "llama3.3"
EMBEDDING_MODEL = "all-minilm"

# Document Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Frontend Configuration
FRONTEND_HOST = "localhost"
FRONTEND_PORT = 8501 
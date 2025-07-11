import os
from pathlib import Path
import logging

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
TMP_DIR = BASE_DIR / "tmp"
BASE_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)

# Configure logging
LOG_DIR = TMP_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
APP_LOG_FILE = LOG_DIR / "app.log"
VECTOR_STORE_LOG_FILE = LOG_DIR / "chroma_db.log"
SQLITE_LOG_FILE = LOG_DIR / "sqlite_db.log"

# Configure root logger
logging.basicConfig(
    level=logging.INFO,  # Set root logger to INFO to reduce console output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(APP_LOG_FILE),
        logging.StreamHandler()
    ]
)

# Configure httpx logger to suppress HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Configure vector store logger
vector_store_logger = logging.getLogger('ng_notebook.services.vector_store')
vector_store_logger.setLevel(logging.DEBUG)
vector_store_handler = logging.FileHandler(VECTOR_STORE_LOG_FILE)
vector_store_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
vector_store_logger.addHandler(vector_store_handler)

# Configure SQLite store logger
sqlite_logger = logging.getLogger('ng_notebook.services.sqlite_store')
sqlite_logger.setLevel(logging.DEBUG)
sqlite_handler = logging.FileHandler(SQLITE_LOG_FILE)
sqlite_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
sqlite_logger.addHandler(sqlite_handler)


CHROMA_DB_DIR = TMP_DIR / "chroma_db"
SQLITE_DB_DIR = TMP_DIR / "sqlite_db"
UPLOAD_DIR = TMP_DIR / "uploads"

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
API_HOST = "localhost"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}"
CHAT_TIMEOUT = 120  # seconds

# Frontend Configuration
FRONTEND_HOST = "localhost"
FRONTEND_PORT = 8501 
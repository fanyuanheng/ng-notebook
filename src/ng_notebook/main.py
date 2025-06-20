import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import logging
from .api.routes import router
from .core.config import API_HOST, API_PORT, UPLOAD_DIR
from .services.document_processor import DocumentProcessor
from .services.vector_store import VectorStore
from .services.sqlite_store import SQLiteStore
from .services.vector_store import get_vector_store, get_sqlite_store, get_document_processor, get_llm
from .services.chat_service import ChatService, ChatRequest
from .services.upload_service import UploadService

# Get logger
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Neogenesis Notebook API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize document processor
document_processor = DocumentProcessor()

# Initialize Ollama
llm = Ollama(model="llama3.3:latest")
embeddings = OllamaEmbeddings(model="all-minilm")

# Get vector store instance using singleton pattern
vector_store = VectorStore()

# Initialize services
chat_service = ChatService(vector_store, llm)
upload_service = UploadService(vector_store, document_processor, UPLOAD_DIR)

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    vector_store: VectorStore = Depends(get_vector_store),
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
    """Upload a file and process it into chunks."""
    return await upload_service.process_upload(file, vector_store, sqlite_store, document_processor)

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat with the AI about the documents."""
    try:
        return await chat_service.process_chat(request)
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Include API routes
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT) 
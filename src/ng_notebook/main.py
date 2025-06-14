import os
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import streamlit as st
import tempfile
import pandas as pd
from pptx import Presentation
import PyPDF2
import magic
import shutil
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from .api.routes import router
from .core.config import API_HOST, API_PORT, CHROMA_DB_DIR, UPLOAD_DIR
import logging
from .services.document_processor import DocumentProcessor
from langchain_core.documents import Document
from .services.vector_store import VectorStore
from .services.sqlite_store import SQLiteStore
from .services.vector_store import get_vector_store, get_sqlite_store, get_document_processor, get_llm
from .services.chat_service import ChatService, ChatRequest

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

# Track uploaded documents
uploaded_documents = set()

# Initialize document processor
document_processor = DocumentProcessor()

# Initialize Ollama
llm = Ollama(model="llama3.3:latest")
embeddings = OllamaEmbeddings(model="all-minilm")

# Get vector store instance using singleton pattern
vector_store = VectorStore()

# Initialize chat service
chat_service = ChatService(vector_store, llm)

def clean_metadata(metadata: dict) -> dict:
    """Clean metadata to ensure all values are simple types."""
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
        elif isinstance(value, list):
            cleaned[key] = ",".join(str(x) for x in value)
        else:
            cleaned[key] = str(value)
    return cleaned

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    vector_store: VectorStore = Depends(get_vector_store),
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    document_processor: DocumentProcessor = Depends(get_document_processor)
):
    """Upload a file and process it into chunks."""
    logger.info(f"Received file upload request for: {file.filename}")
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    logger.info(f"Saving uploaded file to: {file_path}")
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Verify file exists and has content
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
        
        file_size = os.path.getsize(file_path)
        logger.info(f"Saved file size: {file_size} bytes")
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Detect file type using python-magic
        file_type = magic.from_file(file_path, mime=True)
        logger.info(f"Detected file type: {file_type}")
        
        if not file_type:
            # Fallback to content_type if magic fails
            file_type = file.content_type
            logger.info(f"Using content_type as fallback: {file_type}")
        
        if not file_type:
            raise HTTPException(status_code=400, detail="Could not detect file type")
        
        # Process the document into chunks
        logger.info("Processing document into chunks")
        chunks = document_processor.process_document(file_path, file_type)
        logger.info(f"Generated {len(chunks)} chunks")
        
        # Add chunks to vector store
        logger.info("Adding chunks to vector store")
        for chunk in chunks:
            if isinstance(chunk, dict):
                # Convert dict to Document if needed
                doc = Document(
                    page_content=chunk.get("page_content", ""),
                    metadata=chunk.get("metadata", {})
                )
                vector_store.add_documents([doc])
            else:
                # If it's already a Document, use it directly
                vector_store.add_documents([chunk])
        
        logger.info("Successfully processed and stored document")
        return {"message": "File uploaded and processed successfully"}
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up the uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up file: {str(e)}", exc_info=True)

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat with the AI about the documents."""
    try:
        return await chat_service.process_chat(request)
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def get_collections():
    """Get information about the vector store collections."""
    try:
        # Get the vector store instance
        vector_store = get_vector_store()
        
        # Get collection information
        collection = vector_store._collection
        count = collection.count()
        
        # Get unique sources and types
        results = collection.get()
        unique_sources = set()
        unique_types = set()
        type_counts = {}
        type_samples = {}
        source_counts = {}
        
        if results and results.get("metadatas"):
            for metadata in results["metadatas"]:
                if metadata:
                    source = metadata.get("source", "Unknown")
                    doc_type = metadata.get("type", "Unknown")
                    
                    unique_sources.add(source)
                    unique_types.add(doc_type)
                    
                    # Count types
                    type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                    
                    # Count sources
                    source_counts[source] = source_counts.get(source, 0) + 1
                    
                    # Store samples
                    if doc_type not in type_samples:
                        type_samples[doc_type] = []
                    if len(type_samples[doc_type]) < 3:  # Keep up to 3 samples per type
                        type_samples[doc_type].append({
                            "content": results["documents"][len(type_samples[doc_type])],
                            "metadata": metadata
                        })
        
        return {
            "total_documents": count,
            "unique_sources": list(unique_sources),
            "document_types": list(unique_types),
            "collection_name": collection.name,
            "uploaded_files": list(uploaded_documents),
            "type_statistics": {
                "counts": type_counts,
                "samples": type_samples
            },
            "source_statistics": {
                "counts": source_counts
            }
        }
    except Exception as e:
        return {"error": f"Error getting collections: {str(e)}"}

# Include API routes
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT) 
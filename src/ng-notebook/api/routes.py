from fastapi import APIRouter, UploadFile, File
from typing import List, Dict
import os
import magic
from ..services.document_processor import DocumentProcessor
from ..services.vector_store import VectorStore
from ..models.document import Query, QueryResponse
from ..core.config import UPLOAD_DIR

router = APIRouter()
document_processor = DocumentProcessor()
vector_store = VectorStore()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document."""
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Detect file type
    file_type = magic.from_file(file_path, mime=True)
    
    # Process the document
    chunks = document_processor.process_document(file_path, file_type)
    
    # Add to vector store
    vector_store.add_documents(chunks)
    
    return {"message": "File processed successfully", "chunks": len(chunks)}

@router.post("/query", response_model=QueryResponse)
async def query_documents(query: Query):
    """Query the vector store."""
    result = vector_store.query(query.question, query.chat_history)
    return result

@router.get("/collections")
async def get_collections():
    """Get information about the vector store collections."""
    return vector_store.get_collections() 
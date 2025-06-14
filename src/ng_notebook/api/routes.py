from fastapi import APIRouter, UploadFile, File
from typing import List, Dict
import os
import magic
import logging
from ..services.document_processor import DocumentProcessor
from ..services.vector_store import VectorStore
from ..models.document import Query, QueryResponse
from ..core.config import UPLOAD_DIR, SQLITE_DB_DIR

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter()
document_processor = DocumentProcessor()
vector_store = VectorStore()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document."""
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    logger.debug(f"Processing upload for file: {file.filename}")
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        logger.debug(f"File saved to: {file_path}")
        
        # Verify file exists and has content
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File was not saved correctly at {file_path}")
        file_size = os.path.getsize(file_path)
        logger.debug(f"File size: {file_size} bytes")
        
        # Detect file type
        file_type = magic.from_file(file_path, mime=True)
        logger.debug(f"Detected file type: {file_type} for file: {file_path}")
        
        # Process the document
        chunks = document_processor.process_document(file_path, file_type)
        logger.debug(f"Document processed into {len(chunks)} chunks")
        
        # Add to vector store
        vector_store.add_documents(chunks)
        logger.debug("Documents added to vector store")
        
        return {"message": "File processed successfully", "chunks": len(chunks)}
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        return {"error": f"Error processing file: {str(e)}"}
    finally:
        # Clean up the uploaded file only after all processing is complete
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Cleaned up uploaded file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up file: {str(e)}", exc_info=True)

@router.post("/query", response_model=QueryResponse)
async def query_documents(query: Query):
    """Query documents using RAG."""
    try:
        response = vector_store.query(query.question, query.chat_history)
        return QueryResponse(
            answer=response["answer"],
            sources=response.get("source_documents", []),
            sqlite_results=response.get("sqlite_results", [])
        )
    except Exception as e:
        return QueryResponse(
            answer=f"Error querying documents: {str(e)}",
            sources=[],
            sqlite_results=[]
        )

@router.get("/collections")
async def get_collections():
    """Get information about the vector store collections."""
    return vector_store.get_collections() 
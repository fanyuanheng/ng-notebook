import os
import logging
from fastapi import HTTPException, UploadFile
import magic
from langchain_core.documents import Document
from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

class UploadService:
    def __init__(
        self,
        vector_store: VectorStore,
        document_processor: DocumentProcessor,
        upload_dir: str
    ):
        self.vector_store = vector_store
        self.document_processor = document_processor
        self.upload_dir = upload_dir
        self.uploaded_documents = set()

    def clean_metadata(self, metadata: dict) -> dict:
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

    async def process_upload(
        self,
        file: UploadFile,
        vector_store: VectorStore,
        sqlite_store: SQLiteStore,
        document_processor: DocumentProcessor
    ):
        """Process an uploaded file and store it in the vector store."""
        logger.info(f"Received file upload request for: {file.filename}")
        
        # Save the uploaded file
        file_path = os.path.join(self.upload_dir, file.filename)
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
            
            # Add to uploaded documents set
            self.uploaded_documents.add(file.filename)
            
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

    def get_uploaded_documents(self):
        """Get the set of uploaded documents."""
        return self.uploaded_documents 
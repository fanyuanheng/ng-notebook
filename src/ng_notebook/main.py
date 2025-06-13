import os
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
import tempfile
import pandas as pd
from pptx import Presentation
import PyPDF2
import magic
from langchain.prompts import PromptTemplate
import shutil
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from .api.routes import router
from .core.config import API_HOST, API_PORT
import logging
from .services.document_processor import DocumentProcessor
from langchain_core.documents import Document
from langchain.vectorstores import VectorStore
from .services.sqlite_store import SQLiteStore
from .services.vector_store import get_vector_store, get_sqlite_store, get_document_processor, get_llm

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
vector_store = get_vector_store()

# Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Create a more detailed prompt template for better document handling
template = """You are an AI assistant specialized in analyzing and explaining data from various document types. 
When dealing with different document types, pay special attention to:

For Excel/CSV files:
1. Numerical data and calculations
2. Column names and their relationships
3. Row indices and their context
4. Data types and formats
5. Sheet names and their relationships (for Excel)
6. Statistical information (mean, median, correlations)
7. Sample values and unique counts

For PDF documents:
1. Page numbers and their context
2. Document structure and sections
3. Headers and subheaders
4. Lists and bullet points
5. Tables and their content
6. Important figures and statistics
7. Key concepts and their relationships

For PowerPoint presentations:
1. Slide numbers and their sequence
2. Slide titles and subtitles
3. Bullet points and their hierarchy
4. Images and their captions
5. Speaker notes if available
6. Presentation flow and structure
7. Key messages and takeaways

Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Chat History: {chat_history}

Question: {question}

When answering:
1. For Excel/CSV data:
   - Reference specific columns, rows, or cells when relevant
   - Include numerical values in your response
   - Explain patterns or relationships in the data
   - Show calculations when asked
   - Mention sheet names for Excel files

2. For PDF documents:
   - Reference specific pages or sections
   - Maintain document structure in your response
   - Include relevant quotes or statistics
   - Explain relationships between concepts
   - Reference tables or figures when relevant

3. For PowerPoint presentations:
   - Reference specific slides
   - Maintain presentation flow
   - Include key points and their context
   - Explain relationships between slides
   - Reference visual elements when relevant

4. General guidelines:
   - Be specific and precise in your references
   - Provide context for your answers
   - Explain relationships and patterns
   - Include relevant statistics or numbers
   - Maintain document structure in your response

Answer:
"""

# Create the prompt
prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=template
)

# Initialize the chain with the custom prompt
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(
        search_type="similarity"
    ),
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": prompt}
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    chat_history: List[ChatMessage]

class ChatResponse(BaseModel):
    response: str

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
    
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    
    # Save the uploaded file
    file_path = os.path.join("uploads", file.filename)
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
async def chat(
    request: ChatRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    llm: Ollama = Depends(get_llm)
) -> ChatResponse:
    """Handle chat requests."""
    try:
        # Get relevant documents from vector store
        docs = vector_store.similarity_search(request.message)
        
        # Create context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt
        prompt = f"""Based on the following context, answer the question. If you don't know the answer, just say that you don't know.

Context:
{context}

Question: {request.message}

Answer:"""
        
        # Get response from LLM
        response = llm.invoke(prompt)
        
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-db")
async def clear_database():
    """Clear the Chroma database."""
    try:
        # Get the vector store instance
        vector_store = get_vector_store()
        
        # Delete the collection
        try:
            vector_store._collection.delete()
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
        
        # Delete the Chroma database directory
        if os.path.exists(CHROMA_DB_DIR):
            try:
                shutil.rmtree(CHROMA_DB_DIR)
            except Exception as e:
                logger.error(f"Error removing directory: {str(e)}")
                # Try to remove files individually
                for root, dirs, files in os.walk(CHROMA_DB_DIR, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except Exception as e:
                            logger.error(f"Error removing file {name}: {str(e)}")
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except Exception as e:
                            logger.error(f"Error removing directory {name}: {str(e)}")
                try:
                    os.rmdir(CHROMA_DB_DIR)
                except Exception as e:
                    logger.error(f"Error removing root directory: {str(e)}")
        
        # Create fresh directory with proper permissions
        os.makedirs(CHROMA_DB_DIR, mode=0o777, exist_ok=True)
        
        # Clear uploaded documents set
        uploaded_documents.clear()
        
        return {"message": "Database cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        return {"error": f"Error clearing database: {str(e)}"}

@app.get("/collections")
async def get_collections():
    """Get detailed information about Chroma collections."""
    try:
        if vector_store is None:
            return {"collections": [], "message": "No collections found"}
        
        # Get collection information
        collection = vector_store._collection
        count = collection.count()
        
        # Get all documents and metadata
        results = collection.get()
        documents = results["documents"]
        metadatas = results["metadatas"]
        
        # Process metadata to get unique values and statistics
        unique_sources = set()
        unique_types = set()
        type_counts = {}
        source_counts = {}
        
        for meta in metadatas:
            if meta:
                if "source" in meta:
                    unique_sources.add(meta["source"])
                    source_counts[meta["source"]] = source_counts.get(meta["source"], 0) + 1
                if "type" in meta:
                    unique_types.add(meta["type"])
                    type_counts[meta["type"]] = type_counts.get(meta["type"], 0) + 1
        
        # Get document samples for each type
        type_samples = {}
        for doc_type in unique_types:
            type_samples[doc_type] = []
            for i, meta in enumerate(metadatas):
                if meta and meta.get("type") == doc_type:
                    type_samples[doc_type].append({
                        "content": documents[i][:200] + "..." if len(documents[i]) > 200 else documents[i],
                        "metadata": meta
                    })
                    if len(type_samples[doc_type]) >= 3:  # Limit to 3 samples per type
                        break
        
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
import os
from typing import List
from fastapi import FastAPI, UploadFile, File
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

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Ollama
llm = Ollama(model="llama3.3:latest")
embeddings = OllamaEmbeddings(model="all-minilm")

# Initialize Chroma as the vector store
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db",
    collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)

# Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    chat_history: List[ChatMessage]

def process_document(file_path: str, file_type: str) -> str:
    """Process different types of documents and return text content."""
    if file_type == "application/pdf":
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    
    elif file_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        prs = Presentation(file_path)
        text = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        return text
    
    elif file_type in ["text/csv", "application/vnd.ms-excel", 
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        df = pd.read_csv(file_path) if file_type == "text/csv" else pd.read_excel(file_path)
        chunks = []
        for sheet_name in pd.ExcelFile(file_path).sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            chunks.append(f"Sheet: {sheet_name}\n{df.to_string()}")
        return "\n".join(chunks)
    
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload and process it for RAG."""
    global vector_store
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Detect file type
        file_type = magic.from_file(temp_file_path, mime=True)
        
        # Process the document
        text = process_document(temp_file_path, file_type)
        
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        
        # Create or update vector store
        if vector_store is None:
            vector_store = Chroma.from_texts(
                chunks,
                embeddings,
                persist_directory="./chroma_db"
            )
        else:
            vector_store.add_texts(chunks)
        
        return {"message": "File processed successfully"}
    
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat requests with RAG."""
    if vector_store is None:
        return {"response": "Please upload a document first."}
    
    # Create conversation chain with improved retrieval
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity",  # Use similarity search
            search_kwargs={
                "k": 3,  # Number of documents to retrieve
                "fetch_k": 5,  # Fetch more documents initially for better selection
                "maximal_marginal_relevance": True,  # Use MMR to ensure diversity
                "filter": None  # No filtering by default
            }
        ),
        memory=memory,
        return_source_documents=True,
        verbose=True  # Enable verbose output for debugging
    )
    
    # Get response
    response = qa_chain({"question": request.message})
    
    return {
        "response": response["answer"],
        "sources": [doc.page_content for doc in response["source_documents"]]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
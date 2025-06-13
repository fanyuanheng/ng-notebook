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
from langchain.prompts import PromptTemplate
import shutil
from langchain_community.vectorstores.utils import filter_complex_metadata

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

# Initialize text splitter with better chunking strategy
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    chat_history: List[ChatMessage]

def process_document(file_path: str, file_type: str) -> List[dict]:
    """Process different types of documents and return list of chunks with metadata."""
    if file_type == "application/pdf":
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return [{"content": text, "metadata": {"type": "pdf"}}]
    
    elif file_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        prs = Presentation(file_path)
        text = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        return [{"content": text, "metadata": {"type": "pptx"}}]
    
    elif file_type in ["text/csv", "application/vnd.ms-excel", 
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        chunks = []
        try:
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Add sheet metadata as a separate chunk
                sheet_metadata = {
                    "content": f"Sheet: {sheet_name}\n"
                              f"Number of rows: {len(df)}\n"
                              f"Number of columns: {len(df.columns)}\n"
                              f"Column names: {', '.join(df.columns)}",
                    "metadata": {
                        "type": "excel_metadata",
                        "sheet": sheet_name,
                        "row_count": str(len(df)),
                        "column_count": str(len(df.columns))
                    }
                }
                chunks.append(sheet_metadata)
                
                # Process each column's statistics
                for column in df.columns:
                    try:
                        if pd.api.types.is_numeric_dtype(df[column]):
                            stats = df[column].describe()
                            stats_text = f"Statistics for {column}:\n"
                            if 'mean' in stats:
                                stats_text += f"Mean: {stats['mean']:.2f}\n"
                            if '50%' in stats:
                                stats_text += f"Median: {stats['50%']:.2f}\n"
                            if 'min' in stats:
                                stats_text += f"Min: {stats['min']:.2f}\n"
                            if 'max' in stats:
                                stats_text += f"Max: {stats['max']:.2f}\n"
                            stats_text += f"Non-null values: {df[column].count()}"
                            
                            chunks.append({
                                "content": stats_text,
                                "metadata": {
                                    "type": "excel_stats",
                                    "sheet": sheet_name,
                                    "column": column,
                                    "data_type": "numeric"
                                }
                            })
                        else:
                            value_counts = df[column].value_counts().head(5)
                            stats_text = f"Top 5 values in {column}:\n"
                            for value, count in value_counts.items():
                                stats_text += f"{value}: {count}\n"
                            stats_text += f"Non-null values: {df[column].count()}"
                            
                            chunks.append({
                                "content": stats_text,
                                "metadata": {
                                    "type": "excel_stats",
                                    "sheet": sheet_name,
                                    "column": column,
                                    "data_type": "categorical"
                                }
                            })
                    except Exception as e:
                        chunks.append({
                            "content": f"Error processing column {column}: {str(e)}",
                            "metadata": {
                                "type": "excel_error",
                                "sheet": sheet_name,
                                "column": column
                            }
                        })
                
                # Process each row as a separate chunk
                for idx, row in df.iterrows():
                    row_data = {col: str(val) for col, val in row.items()}
                    chunks.append({
                        "content": f"Row {idx + 1}:\n" + "\n".join(f"{col}: {val}" for col, val in row_data.items()),
                        "metadata": {
                            "type": "excel_row",
                            "sheet": sheet_name,
                            "row_index": str(idx),
                            "columns": ",".join(df.columns)  # Convert list to string
                        }
                    })
            
            return chunks
        except Exception as e:
            return [{
                "content": f"Error processing Excel file: {str(e)}",
                "metadata": {"type": "error"}
            }]
    
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            return [{"content": text, "metadata": {"type": "text"}}]

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
        chunks = process_document(temp_file_path, file_type)
        
        # Add file metadata to each chunk and clean metadata
        for chunk in chunks:
            chunk["metadata"].update({
                "source": file.filename,
                "file_type": file_type
            })
            # Clean metadata to ensure all values are simple types
            chunk["metadata"] = clean_metadata(chunk["metadata"])
        
        # Create or update vector store
        if vector_store is None:
            vector_store = Chroma.from_texts(
                [chunk["content"] for chunk in chunks],
                embeddings,
                persist_directory="./chroma_db",
                metadatas=[chunk["metadata"] for chunk in chunks]
            )
        else:
            vector_store.add_texts(
                [chunk["content"] for chunk in chunks],
                metadatas=[chunk["metadata"] for chunk in chunks]
            )
        
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
            search_type="similarity",
            search_kwargs={
                "k": 3  # Number of documents to retrieve
            }
        ),
        memory=memory,
        return_source_documents=True,
        verbose=True,  # Enable verbose output for debugging
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate(
                template="""You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. 
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                
                Context: {context}
                
                Chat History: {chat_history}
                
                Question: {question}
                
                Answer:""",
                input_variables=["context", "chat_history", "question"]
            )
        }
    )
    
    # Get response
    response = qa_chain({"question": request.message})
    
    return {
        "response": response["answer"],
        "sources": [doc.page_content for doc in response["source_documents"]]
    }

@app.post("/clear-db")
async def clear_database():
    """Clear the Chroma database."""
    global vector_store
    try:
        # Delete the Chroma database directory
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        
        # Reset the vector store
        vector_store = None
        
        return {"message": "Database cleared successfully"}
    except Exception as e:
        return {"error": f"Error clearing database: {str(e)}"}

@app.get("/collections")
async def get_collections():
    """Get information about Chroma collections."""
    try:
        if vector_store is None:
            return {"collections": [], "message": "No collections found"}
        
        # Get collection information
        collection = vector_store._collection
        count = collection.count()
        
        # Get unique metadata values
        metadata = collection.get()["metadatas"]
        unique_sources = set()
        unique_types = set()
        
        for meta in metadata:
            if meta:
                if "source" in meta:
                    unique_sources.add(meta["source"])
                if "type" in meta:
                    unique_types.add(meta["type"])
        
        return {
            "total_documents": count,
            "unique_sources": list(unique_sources),
            "document_types": list(unique_types),
            "collection_name": collection.name
        }
    except Exception as e:
        return {"error": f"Error getting collections: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
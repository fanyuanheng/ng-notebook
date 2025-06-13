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
import chromadb
from chromadb.config import Settings
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

# Track uploaded documents
uploaded_documents = set()

# Initialize Ollama
llm = Ollama(model="llama3.3:latest")
embeddings = OllamaEmbeddings(model="all-minilm")

# Ensure chroma_db directory exists and has proper permissions
CHROMA_DB_DIR = "./chroma_db"
if not os.path.exists(CHROMA_DB_DIR):
    os.makedirs(CHROMA_DB_DIR, mode=0o777)

# Initialize Chroma client with proper settings
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR,
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True
    )
)

# Create or get the collection
try:
    collection = chroma_client.get_collection("documents")
except Exception:
    collection = chroma_client.create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )

# Initialize Chroma as the vector store
vector_store = Chroma(
    client=chroma_client,
    embedding_function=embeddings,
    collection_name="documents",
    persist_directory=CHROMA_DB_DIR
)

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

Answer:"""

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

def process_document(file_path: str, file_type: str) -> List[dict]:
    """Process different types of documents and return chunks with metadata."""
    chunks = []
    
    try:
        if file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            # Read all sheets from the Excel file
            excel_file = pd.ExcelFile(file_path)
            
            # Process each sheet
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Add sheet-level information
                sheet_info = f"Excel Sheet '{sheet_name}' Information:\n"
                sheet_info += f"Total Rows: {len(df)}\n"
                sheet_info += f"Total Columns: {len(df.columns)}\n"
                sheet_info += f"Column Names: {', '.join(df.columns)}\n"
                sheet_info += f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB\n"
                
                chunks.append({
                    "content": sheet_info,
                    "metadata": {
                        "source": file_path,
                        "type": "excel_sheet_info",
                        "sheet_name": sheet_name,
                        "row_count": str(len(df)),
                        "column_count": str(len(df.columns)),
                        "columns": str(list(df.columns))
                    }
                })
                
                # Process each column in the sheet
                for column in df.columns:
                    # Get column statistics
                    col_stats = {
                        "mean": df[column].mean() if pd.api.types.is_numeric_dtype(df[column]) else None,
                        "median": df[column].median() if pd.api.types.is_numeric_dtype(df[column]) else None,
                        "std": df[column].std() if pd.api.types.is_numeric_dtype(df[column]) else None,
                        "unique_values": df[column].nunique(),
                        "null_count": df[column].isnull().sum(),
                        "data_type": str(df[column].dtype)
                    }
                    
                    # Create a detailed description of the column
                    col_description = f"Column '{column}' in sheet '{sheet_name}':\n"
                    col_description += f"Data Type: {col_stats['data_type']}\n"
                    if pd.api.types.is_numeric_dtype(df[column]):
                        col_description += f"Mean: {col_stats['mean']:.2f}\n"
                        col_description += f"Median: {col_stats['median']:.2f}\n"
                        col_description += f"Standard Deviation: {col_stats['std']:.2f}\n"
                    col_description += f"Unique Values: {col_stats['unique_values']}\n"
                    col_description += f"Null Values: {col_stats['null_count']}\n"
                    
                    # Add sample values
                    sample_values = df[column].dropna().head(5).tolist()
                    col_description += f"Sample Values: {sample_values}\n"
                    
                    # Add column content
                    col_content = df[column].astype(str).str.cat(sep="\n")
                    col_description += f"\nColumn Content:\n{col_content}"
                    
                    chunks.append({
                        "content": col_description,
                        "metadata": {
                            "source": file_path,
                            "type": "excel_column",
                            "sheet_name": sheet_name,
                            "column_name": column,
                            "row_count": str(len(df)),
                            "column_index": str(df.columns.get_loc(column)),
                            "statistics": str(col_stats)
                        }
                    })
                
                # Add correlation information for numerical columns in this sheet
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    corr_info = f"Correlation Matrix for Numerical Columns in sheet '{sheet_name}':\n"
                    corr_info += str(corr_matrix)
                    
                    chunks.append({
                        "content": corr_info,
                        "metadata": {
                            "source": file_path,
                            "type": "excel_correlations",
                            "sheet_name": sheet_name,
                            "numeric_columns": str(list(numeric_cols))
                        }
                    })
                
                # Add row-level information for the first 100 rows (to avoid too many chunks)
                for idx, row in df.head(100).iterrows():
                    row_data = {col: str(val) for col, val in row.items()}
                    row_info = f"Row {idx + 1} in sheet '{sheet_name}':\n"
                    row_info += "\n".join(f"{col}: {val}" for col, val in row_data.items())
                    
                    chunks.append({
                        "content": row_info,
                        "metadata": {
                            "source": file_path,
                            "type": "excel_row",
                            "sheet_name": sheet_name,
                            "row_index": str(idx),
                            "columns": str(list(df.columns))
                        }
                    })
        
        elif file_type == "application/pdf":
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                )
                
                # Create chunks with metadata
                for i, chunk in enumerate(text_splitter.split_text(text)):
                    chunks.append({
                        "content": chunk,
                        "metadata": {
                            "source": file_path,
                            "type": "pdf",
                            "page": str(i + 1),
                            "total_pages": str(len(pdf_reader.pages))
                        }
                    })
        
        elif file_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            prs = Presentation(file_path)
            text = ""
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            # Create chunks with metadata
            for i, chunk in enumerate(text_splitter.split_text(text)):
                chunks.append({
                    "content": chunk,
                    "metadata": {
                        "source": file_path,
                        "type": "ppt",
                        "slide": str(i + 1),
                        "total_slides": str(len(prs.slides))
                    }
                })
        
        elif file_type in ["text/csv", "application/vnd.ms-excel"]:
            df = pd.read_csv(file_path)
            text = df.to_string()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            # Create chunks with metadata
            for i, chunk in enumerate(text_splitter.split_text(text)):
                chunks.append({
                    "content": chunk,
                    "metadata": {
                        "source": file_path,
                        "type": "csv",
                        "row_count": str(len(df)),
                        "column_count": str(len(df.columns))
                    }
                })
        
        else:
            # Handle unknown file types by reading as text
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            # Create chunks with metadata
            for i, chunk in enumerate(text_splitter.split_text(text)):
                chunks.append({
                    "content": chunk,
                    "metadata": {
                        "source": file_path,
                        "type": "text",
                        "chunk_index": str(i + 1)
                    }
                })
    
    except Exception as e:
        # If any error occurs, create a single chunk with error information
        chunks.append({
            "content": f"Error processing file: {str(e)}",
            "metadata": {
                "source": file_path,
                "type": "error",
                "error_message": str(e)
            }
        })
    
    return chunks

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
    global vector_store, uploaded_documents, collection
    
    # Check if file has already been uploaded
    if file.filename in uploaded_documents:
        return {"message": "File already uploaded"}
    
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
        
        # Ensure collection exists
        try:
            collection = chroma_client.get_collection("documents")
        except Exception:
            collection = chroma_client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
        
        # Add texts to vector store
        vector_store.add_texts(
            [chunk["content"] for chunk in chunks],
            metadatas=[chunk["metadata"] for chunk in chunks]
        )
        
        # Add to uploaded documents set
        uploaded_documents.add(file.filename)
        
        return {"message": "File processed successfully"}
    
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat requests with RAG."""
    if vector_store is None:
        return {"response": "Please upload a document first."}
    
    # Get response
    response = chain({"question": request.message})
    
    return {
        "response": response["answer"],
        "sources": [doc.page_content for doc in response["source_documents"]]
    }

@app.post("/clear-db")
async def clear_database():
    """Clear the Chroma database."""
    global vector_store, uploaded_documents, chroma_client, collection
    try:
        # First, try to delete the collection
        try:
            chroma_client.delete_collection("documents")
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")
        
        # Then delete the Chroma database directory
        if os.path.exists(CHROMA_DB_DIR):
            try:
                shutil.rmtree(CHROMA_DB_DIR)
            except Exception as e:
                print(f"Error removing directory: {str(e)}")
                # Try to remove files individually
                for root, dirs, files in os.walk(CHROMA_DB_DIR, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except Exception as e:
                            print(f"Error removing file {name}: {str(e)}")
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except Exception as e:
                            print(f"Error removing directory {name}: {str(e)}")
                try:
                    os.rmdir(CHROMA_DB_DIR)
                except Exception as e:
                    print(f"Error removing root directory: {str(e)}")
        
        # Create fresh directory with proper permissions
        os.makedirs(CHROMA_DB_DIR, mode=0o777, exist_ok=True)
        
        # Reinitialize Chroma client
        chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        
        # Create a new collection
        collection = chroma_client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Reset the vector store
        vector_store = Chroma(
            client=chroma_client,
            embedding_function=embeddings,
            collection_name="documents",
            persist_directory=CHROMA_DB_DIR
        )
        
        # Clear uploaded documents set
        uploaded_documents.clear()
        
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
            "collection_name": collection.name,
            "uploaded_files": list(uploaded_documents)
        }
    except Exception as e:
        return {"error": f"Error getting collections: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
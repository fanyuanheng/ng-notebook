# Neogenesis Notebook

An AI-powered document analysis assistant that can process and analyze PDF, Excel, and PowerPoint files.

## Project Structure

```
src/ng-notebook/
├── api/                 # API routes and endpoints
│   └── routes.py       # FastAPI route definitions
├── core/               # Core configuration and utilities
│   └── config.py       # Application configuration
├── frontend/           # Streamlit frontend
│   ├── app.py         # Main frontend application
│   ├── static/        # Static assets
│   │   └── css/      # CSS styles
│   └── templates/     # UI templates
├── models/             # Data models
│   └── document.py    # Document-related models
├── services/           # Business logic services
│   ├── document_processor.py  # Document processing service
│   └── vector_store.py       # Vector store service
└── main.py            # FastAPI application entry point
```

## Key Dependencies

- **LangChain**: Framework for building LLM applications
- **Ollama**: Local LLM server for running open-source models
- **Chroma**: Vector database for storing and retrieving document embeddings

## How It Works

This application implements a Retrieval-Augmented Generation (RAG) system using three main technologies:

1. **LangChain**
   - Provides the framework for building the RAG pipeline
   - Handles document loading, chunking, and processing
   - Manages the conversation chain with memory
   - Integrates with Ollama for LLM interactions
   - Connects with Chroma for vector storage

2. **Ollama**
   - Runs the local LLM (Llama2) for text generation
   - Provides embeddings for document chunks
   - Enables offline, private document processing
   - Handles the conversation context and responses

3. **Chroma**
   - Stores document chunks as vector embeddings
   - Enables semantic search across documents
   - Maintains metadata for each document chunk
   - Provides efficient similarity search

### RAG Pipeline

1. **Document Processing**
   - Documents are uploaded and processed by LangChain
   - Content is split into chunks for better context management
   - Each chunk is converted to embeddings using Ollama

2. **Vector Storage**
   - Embeddings are stored in Chroma with metadata
   - Enables efficient semantic search across documents
   - Maintains document context and relationships

3. **Query Processing**
   - User questions are converted to embeddings
   - Chroma finds relevant document chunks
   - Ollama generates responses using retrieved context
   - LangChain manages the conversation flow

## Features

- Document processing for PDF, Excel, and PowerPoint files
- AI-powered document analysis and querying
- Modern web interface with real-time chat
- Document collection management
- Vector-based semantic search

## Setup

1. Install dependencies:
```bash
pip install -e .
```

2. Start the backend server:
```bash
uvicorn ng_notebook.main:app --reload
```

3. Start the frontend:
```bash
streamlit run src/ng_notebook/app.py
```

## Usage

1. Open your browser and navigate to `http://localhost:8501`
2. Upload your documents using the sidebar
3. Ask questions about your documents in the chat interface
4. View document collections and statistics in the sidebar

## Development

The project is structured into several key components:

- **API**: FastAPI routes for handling file uploads and queries
- **Services**: Core business logic for document processing and vector storage
- **Frontend**: Streamlit-based user interface with modular templates and styles
- **Models**: Pydantic models for data validation and serialization

## License

MIT 
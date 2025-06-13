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
│   └── app.py         # Main frontend application
├── models/             # Data models
│   └── document.py    # Document-related models
├── services/           # Business logic services
│   ├── document_processor.py  # Document processing service
│   └── vector_store.py       # Vector store service
├── main.py            # FastAPI application entry point
└── run_frontend.py    # Frontend application entry point
```

## Features

- Document processing for PDF, Excel, and PowerPoint files
- AI-powered document analysis and querying
- Modern web interface with real-time chat
- Document collection management
- Vector-based semantic search

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the backend server:
```bash
python -m src.ng-notebook.main
```

3. Start the frontend:
```bash
python -m src.ng-notebook.run_frontend
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
- **Frontend**: Streamlit-based user interface
- **Models**: Pydantic models for data validation and serialization

## License

MIT 
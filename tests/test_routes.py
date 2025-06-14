import os
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from ng_notebook.api.routes import router
from ng_notebook.models.document import Query, QueryResponse

app = FastAPI()
app.include_router(router)
client = TestClient(app)

@pytest.fixture
def mock_document_processor():
    with patch("ng_notebook.api.routes.document_processor") as mock:
        mock.process_document.return_value = [{"content": "Test chunk", "metadata": {"source": "test.txt"}}]
        yield mock

@pytest.fixture
def mock_vector_store():
    with patch("ng_notebook.api.routes.vector_store") as mock:
        mock.add_documents = MagicMock()
        mock.query.return_value = {"answer": "Test answer", "source_documents": [], "sqlite_results": []}
        mock.get_collections.return_value = {"total_documents": 1, "unique_sources": 1, "samples": []}
        yield mock

def test_upload_file(mock_document_processor, mock_vector_store):
    # Create a test file
    test_file_content = b"test content"
    test_file = ("test.txt", test_file_content, "text/plain")
    response = client.post("/upload", files={"file": test_file})
    assert response.status_code == 200
    assert response.json()["message"] == "File processed successfully"
    assert response.json()["chunks"] == 1
    mock_document_processor.process_document.assert_called_once()
    mock_vector_store.add_documents.assert_called_once()

def test_query_documents(mock_vector_store):
    query = Query(question="What is the capital of France?", chat_history=[])
    response = client.post("/query", json=query.dict())
    assert response.status_code == 200
    assert response.json()["answer"] == "Test answer"
    mock_vector_store.query.assert_called_once()

def test_get_collections(mock_vector_store):
    response = client.get("/collections")
    assert response.status_code == 200
    assert response.json()["total_documents"] == 1
    mock_vector_store.get_collections.assert_called_once() 
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from ng_notebook.services.vector_store import VectorStore

@pytest.fixture
def test_chroma_dir(tmp_path):
    """Create a temporary directory for test Chroma database."""
    chroma_dir = tmp_path / "test_chroma_db"
    chroma_dir.mkdir(exist_ok=True)
    return chroma_dir

@pytest.fixture
def mock_retriever():
    """Create a mock retriever."""
    retriever = Mock(spec=BaseRetriever)
    retriever.get_relevant_documents.return_value = [
        Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
        Document(page_content="Test document 2", metadata={"source": "test2.txt"})
    ]
    return retriever

@pytest.fixture
def mock_chroma(mock_retriever):
    """Create a mock Chroma instance."""
    mock = Mock(spec=Chroma)
    mock._collection = Mock()
    mock._collection.get.return_value = {
        "ids": ["1", "2"],
        "documents": ["doc1", "doc2"],
        "metadatas": [{"source": "test1.txt"}, {"source": "test1.txt"}]  # Both documents have same source
    }
    mock.as_retriever.return_value = mock_retriever
    mock.persist = Mock()  # Add persist method
    return mock

@pytest.fixture
def vector_store(mock_chroma):
    """Create a VectorStore instance with in-memory Chroma database and mocked Chroma."""
    # Create store instance with mocked Chroma
    with patch('ng_notebook.services.vector_store.Chroma', return_value=mock_chroma), \
         patch('ng_notebook.services.vector_store.ConversationalRetrievalChain.from_llm') as mock_chain, \
         patch('ng_notebook.services.vector_store.OllamaLLM') as mock_llm:
        # Mock the chain
        mock_chain_instance = Mock()
        mock_chain_instance.return_value = {
            "answer": "Test answer",
            "source_documents": [
                Document(page_content="Test document", metadata={"source": "test.txt"})
            ]
        }
        mock_chain.return_value = mock_chain_instance
        
        # Mock the LLM
        mock_llm_instance = Mock()
        mock_llm_instance.predict.return_value = "Test answer"
        mock_llm.return_value = mock_llm_instance
        
        store = VectorStore()
        store.chain = mock_chain_instance
        store.llm = mock_llm_instance
        yield store

def test_add_documents(vector_store, mock_chroma):
    """Test adding documents to the vector store."""
    # Create test documents
    documents = [
        {
            "content": "This is a test document about AI.",
            "metadata": {"source": "test1.txt", "type": "text"}
        },
        {
            "content": "Another test document about machine learning.",
            "metadata": {"source": "test2.txt", "type": "text"}
        }
    ]
    
    # Add documents
    vector_store.add_documents(documents)
    
    # Verify add_texts was called with correct arguments
    mock_chroma.add_texts.assert_called_once()
    call_args = mock_chroma.add_texts.call_args[1]
    assert len(call_args["texts"]) == 2
    assert len(call_args["metadatas"]) == 2
    assert call_args["texts"][0] == documents[0]["content"]
    assert call_args["metadatas"][0] == documents[0]["metadata"]
    # Verify persist was called
    mock_chroma.persist.assert_called_once()

def test_query(vector_store):
    """Test querying the vector store."""
    # Query the store
    response = vector_store.query("What is the capital of France?")
    
    # Verify response
    assert response["answer"] == "Test answer"
    assert len(response["source_documents"]) == 1
    # Verify chain was called
    vector_store.chain.assert_called_once()

def test_get_collections(vector_store, mock_chroma):
    """Test getting collection information."""
    # Get collections
    collections = vector_store.get_collections()
    
    # Verify collections info
    assert collections["total_documents"] == 2
    assert collections["unique_sources"] == 1  # Only one unique source
    assert len(collections["samples"]) == 2 
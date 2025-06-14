from typing import List, Dict, Optional
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from ..core.config import CHROMA_DB_DIR, LLM_MODEL, EMBEDDING_MODEL
from .sqlite_store import SQLiteStore
from .document_processor import DocumentProcessor
import logging
import os
from pathlib import Path
import shutil

# Get the dedicated vector store logger
logger = logging.getLogger('ng_notebook.services.vector_store')

# Initialize services
_vector_store: Optional[Chroma] = None
_sqlite_store: Optional[SQLiteStore] = None
_document_processor: Optional[DocumentProcessor] = None
_llm: Optional[OllamaLLM] = None

def get_llm() -> OllamaLLM:
    """Get or create the LLM instance."""
    global _llm
    if _llm is None:
        logger.info(f"Initializing LLM with model: {LLM_MODEL}")
        try:
            _llm = OllamaLLM(model=LLM_MODEL)
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}", exc_info=True)
            raise
    return _llm

def get_vector_store() -> Chroma:
    """
    Get or create the vector store instance using a singleton pattern.
    Returns:
        Chroma: The vector store instance
    """
    global _vector_store
    
    if _vector_store is None:
        try:
            logger.info("Initializing vector store with directory: %s", CHROMA_DB_DIR)
            
            # Ensure the directory exists with proper permissions
            os.makedirs(CHROMA_DB_DIR, exist_ok=True)
            os.chmod(CHROMA_DB_DIR, 0o755)
            
            # Initialize embeddings
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            logger.debug("Initialized embeddings with model: %s", EMBEDDING_MODEL)
            
            # Initialize Chroma
            _vector_store = Chroma(
                persist_directory=str(CHROMA_DB_DIR),
                embedding_function=embeddings
            )
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize vector store: %s", str(e), exc_info=True)
            raise
    
    return _vector_store

def get_sqlite_store() -> SQLiteStore:
    """Get or create the SQLite store instance."""
    global _sqlite_store
    if _sqlite_store is None:
        logger.info("Initializing SQLite store")
        _sqlite_store = SQLiteStore()
        logger.info("SQLite store initialized")
    return _sqlite_store

def get_document_processor() -> DocumentProcessor:
    """Get or create the document processor instance."""
    global _document_processor
    if _document_processor is None:
        logger.info("Initializing document processor")
        _document_processor = DocumentProcessor(get_sqlite_store())
        logger.info("Document processor initialized")
    return _document_processor

class VectorStore:
    def __init__(self):
        self._embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.llm = OllamaLLM(model=LLM_MODEL)
        self.vector_store = Chroma(
            persist_directory=str(CHROMA_DB_DIR),
            embedding_function=self._embeddings
        )
        self._setup_chain()

    def _setup_chain(self):
        template = """You are an AI assistant for Neogenesis Notebook. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer concise.

        Context: {context}

        Chat History:
        {chat_history}

        Human: {question}
        Assistant:"""

        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template,
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

    def add_documents(self, chunks: List[Dict]):
        """Add documents to the vector store."""
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        self.vector_store.persist()

    def query(self, question: str, chat_history: List[Dict] = None) -> Dict:
        """Query vector store with a question."""
        if chat_history is None:
            chat_history = []
        
        # Get results from vector store
        vector_result = self.chain({"question": question, "chat_history": chat_history})
        
        return {
            "answer": vector_result["answer"],
            "source_documents": vector_result.get("source_documents", [])
        }

    def get_collections(self) -> Dict:
        """Get information about vector store collections."""
        # Get vector store collections
        collections = self.vector_store._collection.get()
        
        # Calculate statistics
        total_documents = len(collections["ids"])
        unique_sources = set(doc["source"] for doc in collections["metadatas"])
        
        # Get document samples
        sample_size = min(5, total_documents)
        samples = []
        for i in range(sample_size):
            samples.append({
                "content": collections["documents"][i],
                "metadata": collections["metadatas"][i]
            })
        
        return {
            "total_documents": total_documents,
            "unique_sources": len(unique_sources),
            "samples": samples
        }

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        """Search for similar documents."""
        return self.vector_store.similarity_search(query, k=k, **kwargs)

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None, **kwargs) -> List[str]:
        """Add texts to the vector store."""
        return self.vector_store.add_texts(texts=texts, metadatas=metadatas, **kwargs)

    def as_retriever(self, **kwargs):
        """Get a retriever for the vector store."""
        return self.vector_store.as_retriever(**kwargs)

    def clear(self):
        """Clear all documents from the vector store."""
        try:
            # Get all document IDs first
            results = self.vector_store._collection.get()
            if results and results.get("ids"):
                self.vector_store._collection.delete(ids=results["ids"])
            
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
            
            logger.info("Vector store cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise

def search_documents(query: str, k: int = 4) -> List[dict]:
    """
    Search for similar documents in the vector store.
    Args:
        query (str): The search query
        k (int): Number of results to return
    Returns:
        List[dict]: List of similar documents with their metadata
    """
    try:
        vector_store = get_vector_store()
        logger.info("Searching vector store for query: %s", query)
        
        results = vector_store.similarity_search_with_score(query, k=k)
        logger.debug("Found %d results", len(results))
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]
        
    except Exception as e:
        logger.error("Failed to search vector store: %s", str(e), exc_info=True)
        raise

def clear_vector_store() -> None:
    """
    Clear all documents from the vector store.
    """
    try:
        vector_store = get_vector_store()
        logger.info("Clearing vector store")
        
        # Delete all documents from the collection
        vector_store.delete(ids=vector_store.get()["ids"])  # Delete all documents by their IDs
        
        logger.info("Vector store cleared successfully")
        
    except Exception as e:
        logger.error("Failed to clear vector store: %s", str(e), exc_info=True)
        raise 
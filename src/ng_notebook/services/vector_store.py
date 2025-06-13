from typing import List, Dict, Optional
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from ..core.config import CHROMA_DB_DIR, LLM_MODEL, EMBEDDING_MODEL
from .sqlite_store import SQLiteStore
from .document_processor import DocumentProcessor
import logging

# Get logger
logger = logging.getLogger(__name__)

# Initialize services
_vector_store: Optional[Chroma] = None
_sqlite_store: Optional[SQLiteStore] = None
_document_processor: Optional[DocumentProcessor] = None
_llm: Optional[Ollama] = None

def get_llm() -> Ollama:
    """Get or create the LLM instance."""
    global _llm
    if _llm is None:
        logger.info(f"Initializing LLM with model: {LLM_MODEL}")
        try:
            _llm = Ollama(model=LLM_MODEL)
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}", exc_info=True)
            raise
    return _llm

def get_vector_store() -> Chroma:
    """Get or create the vector store instance."""
    global _vector_store
    if _vector_store is None:
        logger.info("Initializing vector store")
        try:
            # Initialize embeddings with all-minilm
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            logger.info(f"Initialized embeddings with model: {EMBEDDING_MODEL}")
            
            # Initialize vector store
            _vector_store = Chroma(
                persist_directory=str(CHROMA_DB_DIR),
                embedding_function=embeddings
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}", exc_info=True)
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
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.llm = Ollama(model=LLM_MODEL)
        self.vector_store = Chroma(
            persist_directory=str(CHROMA_DB_DIR),
            embedding_function=self.embeddings
        )
        self.sqlite_store = SQLiteStore()
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
        """Query both vector store and SQLite database with a question."""
        if chat_history is None:
            chat_history = []
        
        # Get results from vector store
        vector_result = self.chain({"question": question, "chat_history": chat_history})
        
        # Get results from SQLite store
        sqlite_results = self.sqlite_store.query_data(question)
        
        # Combine results
        combined_context = vector_result["answer"]
        if sqlite_results:
            combined_context += "\n\nSQLite Data:\n"
            for result in sqlite_results:
                combined_context += f"\nTable: {result['table']}\n"
                combined_context += f"Data: {result['data']}\n"
        
        # Generate final response using combined context
        final_response = self.llm.predict(
            f"""Based on the following context, answer the question:
            
            Context:
            {combined_context}
            
            Question: {question}
            
            Answer:"""
        )
        
        return {
            "answer": final_response,
            "source_documents": vector_result.get("source_documents", []),
            "sqlite_results": sqlite_results
        }

    def get_collections(self) -> Dict:
        """Get information about both vector store and SQLite collections."""
        # Get vector store collections
        collections = self.vector_store._collection.get()
        
        # Calculate statistics
        total_documents = len(collections["ids"])
        unique_sources = len(set(doc["source"] for doc in collections["metadatas"]))
        
        # Get document samples
        sample_size = min(5, total_documents)
        samples = []
        for i in range(sample_size):
            samples.append({
                "content": collections["documents"][i],
                "metadata": collections["metadatas"][i]
            })
        
        # Get SQLite metadata
        sqlite_metadata = []
        for filename in unique_sources:
            metadata = self.sqlite_store.get_file_metadata(filename)
            if metadata:
                sqlite_metadata.append(metadata)
        
        return {
            "total_documents": total_documents,
            "unique_sources": unique_sources,
            "samples": samples,
            "sqlite_metadata": sqlite_metadata
        } 
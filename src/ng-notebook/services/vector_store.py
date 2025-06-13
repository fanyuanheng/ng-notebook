from typing import List, Dict
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from ..core.config import CHROMA_DB_DIR, LLM_MODEL, EMBEDDING_MODEL

class VectorStore:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.llm = Ollama(model=LLM_MODEL)
        self.vector_store = Chroma(
            persist_directory=str(CHROMA_DB_DIR),
            embedding_function=self.embeddings
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
        """Query the vector store with a question."""
        if chat_history is None:
            chat_history = []
        
        result = self.chain({"question": question, "chat_history": chat_history})
        return {
            "answer": result["answer"],
            "source_documents": result.get("source_documents", [])
        }

    def get_collections(self) -> Dict:
        """Get information about the vector store collections."""
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
        
        return {
            "total_documents": total_documents,
            "unique_sources": unique_sources,
            "samples": samples
        } 
from typing import List, Dict, Any
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from .vector_store import VectorStore
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    chat_history: List[ChatMessage]

class ChatResponse(BaseModel):
    response: str

class ChatService:
    def __init__(self, vector_store: VectorStore, llm: Ollama):
        self.vector_store = vector_store
        self.llm = llm
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.chain = self._initialize_chain()

    def _get_prompt_template(self) -> str:
        """Get the prompt template for document analysis."""
        return """You are an AI assistant specialized in analyzing and explaining data from various document types. 
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

Answer:
"""

    def _initialize_chain(self) -> ConversationalRetrievalChain:
        """Initialize the conversation chain with the custom prompt."""
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=self._get_prompt_template()
        )

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_type="similarity"
            ),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt}
        )

    async def process_chat(self, request: ChatRequest) -> Dict[str, Any]:
        """Process a chat request and return the response."""
        try:
            # Get response from vector store
            vector_result = self.vector_store.query(request.message)
            
            return {
                "answer": vector_result["answer"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in vector_result["source_documents"]
                ]
            }
        except Exception as e:
            raise Exception(f"Error in chat: {str(e)}") 
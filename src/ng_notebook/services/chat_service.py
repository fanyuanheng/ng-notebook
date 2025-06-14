from typing import List, Dict, Any
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from .vector_store import VectorStore
from .sqlite_store import SQLiteStore
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    chat_history: List[ChatMessage]

class ChatResponse(BaseModel):
    response: str

class ChatService:
    def __init__(self, vector_store: VectorStore, llm: Ollama, sqlite_store: SQLiteStore = None):
        self.vector_store = vector_store
        self.sqlite_store = sqlite_store or SQLiteStore()
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

        qa_chain = load_qa_chain(
            llm=self.llm,
            chain_type="stuff",
            prompt=prompt
        )

        # Create a question generator chain
        question_generator = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["chat_history", "question"],
                template="""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
            )
        )

        return ConversationalRetrievalChain(
            retriever=self.vector_store.as_retriever(
                search_type="similarity"
            ),
            combine_docs_chain=qa_chain,
            question_generator=question_generator,
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )

    async def process_chat(self, request: ChatRequest) -> Dict[str, Any]:
        """Process a chat request and return the response."""
        try:
            # Get relevant data from SQLite store
            sqlite_context = self._get_sqlite_context(request.message)
            
            # Get response from chain
            response = self.chain.invoke({
                "question": request.message,
                "chat_history": request.chat_history
            })
            
            # Add SQLite context to the answer
            final_answer = f"{response['answer']}\n\nAdditional Context from Database:\n{sqlite_context}"
            
            return {
                "answer": final_answer,
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in response["source_documents"]
                ]
            }
        except Exception as e:
            raise Exception(f"Error in chat: {str(e)}")

    def _get_sqlite_context(self, query: str) -> str:
        """Get relevant data from SQLite store based on the query."""
        try:
            if not self.sqlite_store:
                return "No SQLite store available."

            # Get all uploaded files
            with self.sqlite_store.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, filename, file_type
                    FROM uploaded_files
                    ORDER BY upload_date DESC
                """)
                files = cursor.fetchall()
                
                context_parts = []
                
                for file_id, filename, file_type in files:
                    if file_type == "excel":
                        # Get sheet information
                        cursor.execute("""
                            SELECT sheet_name, row_count, column_count
                            FROM excel_sheets
                            WHERE file_id = ?
                        """, (file_id,))
                        sheets = cursor.fetchall()
                        
                        for sheet_name, row_count, column_count in sheets:
                            # Get sample data from the sheet
                            safe_sheet_name = sheet_name.lower().replace(' ', '_').replace('&', 'and')
                            table_name = f"excel_{file_id}_{safe_sheet_name}"
                            try:
                                # Verify table exists
                                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
                                if cursor.fetchone():
                                    cursor.execute(f"SELECT * FROM [{table_name}] LIMIT 5")
                                    sample_data = cursor.fetchall()
                                    
                                    if sample_data:
                                        context_parts.append(f"""
Excel File: {filename}
Sheet: {sheet_name}
Dimensions: {row_count} rows x {column_count} columns
Sample Data:
{chr(10).join(str(row) for row in sample_data)}
""")
                            except Exception as e:
                                logger.warning(f"Could not fetch data from table {table_name}: {str(e)}")
                                continue
                    elif file_type == "csv":
                        # Get sample data from CSV
                        safe_filename = filename.lower().replace('.', '_').replace('&', 'and')
                        table_name = f"csv_{file_id}_{safe_filename}"
                        try:
                            # Verify table exists
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
                            if cursor.fetchone():
                                cursor.execute(f"SELECT * FROM [{table_name}] LIMIT 5")
                                sample_data = cursor.fetchall()
                                
                                if sample_data:
                                    context_parts.append(f"""
CSV File: {filename}
Sample Data:
{chr(10).join(str(row) for row in sample_data)}
""")
                        except Exception as e:
                            logger.warning(f"Could not fetch data from table {table_name}: {str(e)}")
                            continue
                
                return "\n".join(context_parts) if context_parts else "No relevant data found in database."
                
        except Exception as e:
            logger.error(f"Error retrieving SQLite context: {str(e)}")
            return f"Error retrieving database context: {str(e)}" 
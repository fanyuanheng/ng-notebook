from pydantic import BaseModel
from typing import List, Dict, Optional

class DocumentChunk(BaseModel):
    content: str
    metadata: Dict[str, str]

class Document(BaseModel):
    file_path: str
    file_type: str
    chunks: List[DocumentChunk]

class Query(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]] = None

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, str]] 
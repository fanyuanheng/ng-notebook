import streamlit as st
import requests
import pandas as pd
from typing import List, Dict
import os
from .templates.layout import (
    load_css,
    init_page_config,
    render_header,
    render_sidebar,
    render_chat_interface
)
from ..core.config import API_URL

def init_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "collections" not in st.session_state:
        st.session_state.collections = None

def display_collection_details():
    """Display collection details including both vector store and SQLite data."""
    try:
        response = requests.get(f"{API_URL}/collections")
        data = response.json()
        
        # Display vector store statistics
        st.markdown('<h2 class="subtitle">Vector Store Statistics</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Documents", data["total_documents"])
        with col2:
            st.metric("Unique Sources", data["unique_sources"])
        
        # Display SQLite metadata
        if data.get("sqlite_metadata"):
            st.markdown('<h2 class="subtitle">SQLite Database</h2>', unsafe_allow_html=True)
            for metadata in data["sqlite_metadata"]:
                with st.expander(f"File: {metadata['filename']}"):
                    st.write(f"Type: {metadata['file_type']}")
                    st.write(f"Upload Date: {metadata['upload_date']}")
                    
                    # Display table information
                    for table in metadata["tables"]:
                        st.subheader(f"Table: {table['name']}")
                        st.write(f"Rows: {table['row_count']}")
                        
                        # Display column information
                        columns_df = pd.DataFrame(table["columns"])
                        st.dataframe(columns_df, use_container_width=True)
        
        # Display sample documents
        if data["samples"]:
            st.markdown('<h2 class="subtitle">Sample Documents</h2>', unsafe_allow_html=True)
            for sample in data["samples"]:
                with st.expander(f"Document from {sample['metadata']['source']}"):
                    st.markdown('<div class="sample-container">', unsafe_allow_html=True)
                    st.markdown(f'<div class="sample-content">{sample["content"][:200]}...</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a button to refresh the data
        if st.button("Refresh Data", type="secondary"):
            st.experimental_rerun()
            
    except Exception as e:
        st.error(f"Error fetching collection details: {str(e)}")

def display_chat_response(response: Dict):
    """Display chat response including both vector store and SQLite results."""
    st.write(response["answer"])
    
    # Display source documents
    if response.get("source_documents"):
        with st.expander("Source Documents"):
            for doc in response["source_documents"]:
                st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                st.markdown(f"**Type:** {doc.metadata.get('type', 'Unknown')}")
                st.markdown(f"**Content:** {doc.page_content[:200]}...")
                st.markdown("---")
    
    # Display SQLite results
    if response.get("sqlite_results"):
        with st.expander("SQLite Data"):
            for result in response["sqlite_results"]:
                st.markdown(f"**Table:** {result['table']}")
                st.dataframe(pd.DataFrame([result['data']]), use_container_width=True)
                st.markdown("---")

def upload_file(file):
    """Upload a file to the API."""
    try:
        files = {"file": file}
        response = requests.post(f"{API_URL}/upload", files=files)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return None

def query_documents(question: str, chat_history: List[Dict]):
    """Query the documents through the API."""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question, "chat_history": chat_history}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error querying documents: {str(e)}")
        return None

def get_collections():
    """Get collections information from the API."""
    try:
        response = requests.get(f"{API_URL}/collections")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error getting collections: {str(e)}")
        return None

def chat(message: str, chat_history: List[Dict] = None):
    """Send a chat message to the API."""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"message": message, "chat_history": chat_history or []}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error sending chat message: {str(e)}")
        return None

def main():
    # Initialize page and load CSS
    init_page_config()
    load_css()
    
    # Initialize session state
    init_session_state()
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar(
        upload_file=upload_file,
        on_upload=lambda file: upload_file(file),
        collections=st.session_state.collections
    )
    
    # Render chat interface
    render_chat_interface(
        chat_history=st.session_state.chat_history,
        on_query=lambda question, history: query_documents(question, history)
    )

if __name__ == "__main__":
    main() 
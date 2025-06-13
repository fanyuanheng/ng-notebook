import streamlit as st
import requests
from typing import List, Dict
import os
from .templates.layout import (
    load_css,
    init_page_config,
    render_header,
    render_sidebar,
    render_chat_interface
)

# API Configuration
API_URL = "http://localhost:8000"

def init_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "collections" not in st.session_state:
        st.session_state.collections = None

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
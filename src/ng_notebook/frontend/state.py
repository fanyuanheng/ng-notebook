import streamlit as st
from typing import Set, List, Dict

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()

def add_message(role: str, content: str):
    """Add a message to the chat history."""
    st.session_state.messages.append({"role": role, "content": content})

def get_chat_history() -> List[Dict[str, str]]:
    """Get the chat history for API requests."""
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[:-1]
    ]

def add_uploaded_file(filename: str):
    """Add a file to the list of uploaded files."""
    st.session_state.uploaded_files.add(filename)

def is_file_uploaded(filename: str) -> bool:
    """Check if a file has already been uploaded."""
    return filename in st.session_state.uploaded_files 
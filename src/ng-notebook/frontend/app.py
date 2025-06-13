import streamlit as st
import requests
import json
from typing import List, Dict
import os

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
    st.set_page_config(
        page_title="Neogenesis Notebook",
        page_icon="📚",
        layout="wide"
    )
    
    init_session_state()
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .chat-message.user {
            background-color: #2b313e;
            color: white;
        }
        .chat-message.assistant {
            background-color: #f0f2f6;
        }
        .chat-message .content {
            display: flex;
            margin-top: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("Neogenesis Notebook")
    st.markdown("Your AI-powered document analysis assistant")
    
    # Sidebar
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "xlsx", "pptx"],
            help="Upload PDF, Excel, or PowerPoint files"
        )
        
        if uploaded_file:
            with st.spinner("Processing document..."):
                result = upload_file(uploaded_file)
                if result:
                    st.success(f"Processed {result['chunks']} chunks")
                    st.session_state.collections = get_collections()
        
        st.header("Collections")
        if st.session_state.collections:
            st.metric(
                "Total Documents",
                st.session_state.collections["total_documents"]
            )
            st.metric(
                "Unique Sources",
                st.session_state.collections["unique_sources"]
            )
            
            if st.session_state.collections["samples"]:
                st.subheader("Sample Documents")
                for sample in st.session_state.collections["samples"]:
                    with st.expander(f"Document from {sample['metadata']['source']}"):
                        st.text(sample["content"][:200] + "...")
    
    # Main chat interface
    st.header("Chat with Your Documents")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_documents(prompt, st.session_state.chat_history)
                if response:
                    st.write(response["answer"])
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["answer"]
                    })

if __name__ == "__main__":
    main() 
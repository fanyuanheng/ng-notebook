import streamlit as st
import requests
import pandas as pd
from typing import Dict, List, Any

# Custom CSS styling
st.set_page_config(
    page_title="Neogenesis Notebook",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main title styling */
    .main .title {
        color: #1E88E5;
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Subtitle styling */
    .main .subtitle {
        color: #424242;
        font-size: 1.5rem !important;
        font-weight: 400 !important;
        margin-bottom: 2rem !important;
    }
    
    /* Card styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        background-color: #f8f9fa;
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #1E88E5;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Metrics styling */
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Chat input styling */
    .stChatInput {
        border-radius: 10px;
        border: 2px solid #1E88E5;
    }
    
    /* Success message styling */
    .stSuccess {
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Error message styling */
    .stError {
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = set()

# Title and description
st.markdown('<h1 class="title">Neogenesis Notebook</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subtitle">AI-Powered Document Analysis Platform</h2>', unsafe_allow_html=True)
st.markdown("""
Neogenesis Notebook is an advanced document analysis platform that combines cutting-edge AI technologies to help you understand and interact with your documents.

Get started by uploading your documents and asking questions about their content.
""")

# File uploader
st.markdown('<h2 class="subtitle">Upload Document</h2>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["pdf", "xlsx", "csv", "pptx", "txt"],
    help="Supported formats: PDF, Excel, CSV, PowerPoint, and text files"
)

# Process uploaded file
if uploaded_file is not None and uploaded_file.name not in st.session_state.uploaded_files:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    response = requests.post("http://localhost:8000/upload", files=files)
    
    if response.status_code == 200:
        st.success("Document processed successfully!")
        st.session_state.uploaded_files.add(uploaded_file.name)
    else:
        st.error("Error processing document. Please try again.")

# Chat interface
st.markdown('<h2 class="subtitle">Chat</h2>', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your document"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare chat history for API
    chat_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[:-1]
    ]
    
    # Send request to backend
    response = requests.post(
        "http://localhost:8000/chat",
        json={"message": prompt, "chat_history": chat_history}
    )
    
    if response.status_code == 200:
        response_data = response.json()
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_data["answer"]
        })
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response_data["answer"])
            
            # Display sources if available
            if response_data.get("source_documents"):
                with st.expander("View Sources"):
                    for source in response_data["source_documents"]:
                        st.markdown(f"**Source:** {source['metadata'].get('source', 'Unknown')}")
                        st.markdown(f"**Content:** {source['content'][:200]}...")
                        st.markdown("---")
    else:
        st.error("Error getting response. Please try again.") 
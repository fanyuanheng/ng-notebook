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

def display_collection_details():
    """Display detailed information about the Chroma DB collection."""
    st.markdown('<h1 class="title">Chroma DB Collection Details</h1>', unsafe_allow_html=True)
    
    # Fetch collection information
    response = requests.get("http://localhost:8000/collections")
    if response.status_code != 200:
        st.error("Error fetching collection information. Please try again.")
        return
    
    data = response.json()
    if "error" in data:
        st.error(data["error"])
        return
    
    # Display basic collection information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", data["total_documents"])
    with col2:
        st.metric("Document Types", len(data["document_types"]))
    with col3:
        st.metric("Unique Sources", len(data["unique_sources"]))
    
    # Display document types with counts
    st.markdown('<h2 class="subtitle">Document Types</h2>', unsafe_allow_html=True)
    type_data = []
    for doc_type, count in data["type_statistics"]["counts"].items():
        type_data.append({
            "Type": doc_type,
            "Count": count
        })
    doc_types_df = pd.DataFrame(type_data)
    st.dataframe(doc_types_df, use_container_width=True)
    
    # Display document type samples
    st.markdown('<h2 class="subtitle">Document Type Samples</h2>', unsafe_allow_html=True)
    for doc_type, samples in data["type_statistics"]["samples"].items():
        with st.expander(f"Sample {doc_type} Documents"):
            for i, sample in enumerate(samples, 1):
                st.markdown(f"**Sample {i}**")
                st.markdown("**Content:**")
                st.text(sample["content"])
                st.markdown("**Metadata:**")
                st.json(sample["metadata"])
                st.markdown("---")
    
    # Display uploaded files with counts
    st.markdown('<h2 class="subtitle">Uploaded Files</h2>', unsafe_allow_html=True)
    file_data = []
    for source, count in data["source_statistics"]["counts"].items():
        file_data.append({
            "Filename": source,
            "Chunks": count
        })
    files_df = pd.DataFrame(file_data)
    st.dataframe(files_df, use_container_width=True)
    
    # Add a button to refresh the data
    if st.button("Refresh Data", type="secondary"):
        st.experimental_rerun()

# Title and description
st.markdown('<h1 class="title">Neogenesis Notebook</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subtitle">AI-Powered Document Analysis Platform</h2>', unsafe_allow_html=True)
st.markdown("""
Neogenesis Notebook is an advanced document analysis platform that combines cutting-edge AI technologies to help you understand and interact with your documents.

Get started by uploading your documents and asking questions about their content.
""")

# Add database management buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Clear Database", type="secondary"):
        response = requests.post("http://localhost:8000/clear-db")
        if response.status_code == 200:
            st.success("Database cleared successfully!")
            # Clear chat history
            st.session_state.messages = []
        else:
            st.error("Error clearing database. Please try again.")

with col2:
    if st.button("View Collections", type="secondary"):
        display_collection_details()

# File uploader
st.markdown('<h2 class="subtitle">Upload Document</h2>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["pdf", "xlsx", "csv", "pptx", "txt"],
    help="Supported formats: PDF, Excel, CSV, PowerPoint, and text files"
)

# Process uploaded file
if uploaded_file is not None:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    response = requests.post("http://localhost:8000/upload", files=files)
    
    if response.status_code == 200:
        st.success("Document processed successfully!")
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
            "content": response_data["response"]
        })
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response_data["response"])
            
            # Display sources if available
            if response_data.get("sources"):
                with st.expander("View Sources"):
                    for source in response_data["sources"]:
                        st.markdown(f"```\n{source}\n```")
    else:
        st.error("Error getting response. Please try again.") 
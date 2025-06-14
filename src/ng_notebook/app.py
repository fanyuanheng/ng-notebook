import streamlit as st
from typing import Dict, List
from ng_notebook.frontend.api_client import APIClient
from ng_notebook.frontend.templates.markdown import (
    get_title_markdown,
    get_subtitle_markdown,
    get_description_markdown,
    get_upload_section_markdown,
    get_chat_section_markdown,
    get_source_markdown
)
from ng_notebook.frontend.state import initialize_session_state, add_message, get_chat_history, add_uploaded_file, is_file_uploaded
from ng_notebook.frontend.utils.css import load_css

# Initialize API client
api_client = APIClient()

# Load custom CSS
load_css()

# Initialize session state
initialize_session_state()

# Title and description
st.markdown(get_title_markdown(), unsafe_allow_html=True)
st.markdown(get_subtitle_markdown(), unsafe_allow_html=True)
st.markdown(get_description_markdown())

# File uploader
st.markdown(get_upload_section_markdown(), unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["pdf", "xlsx", "csv", "pptx", "txt"],
    help="Supported formats: PDF, Excel, CSV, PowerPoint, and text files"
)

# Process uploaded file
if uploaded_file is not None and not is_file_uploaded(uploaded_file.name):
    try:
        if api_client.upload_file(uploaded_file.getvalue(), uploaded_file.name):
            st.success("Document processed successfully!")
            add_uploaded_file(uploaded_file.name)
        else:
            st.error("Error processing document. Please try again.")
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")

# Chat interface
st.markdown(get_chat_section_markdown(), unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your document"):
    # Add user message to chat history
    add_message("user", prompt)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    try:
        # Get chat history and send request
        chat_history = get_chat_history()
        response_data = api_client.send_chat_message(prompt, chat_history)
        
        # Add assistant response to chat history
        add_message("assistant", response_data["answer"])
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response_data["answer"])
            
            # Display sources if available
            if response_data.get("source_documents"):
                with st.expander("View Sources"):
                    for source in response_data["source_documents"]:
                        st.markdown(get_source_markdown(source))
    except Exception as e:
        st.error(f"Error getting response: {str(e)}") 
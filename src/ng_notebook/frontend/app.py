import streamlit as st
from .state import initialize_session_state, add_message, get_chat_history, add_uploaded_file, is_file_uploaded
from .api_client import APIClient
from ..core.config import API_URL

# Initialize API client
api_client = APIClient(base_url=API_URL)

# Set page config
st.set_page_config(
    page_title="Neogenesis Notebook",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open("src/ng_notebook/frontend/static/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state
initialize_session_state()

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
st.markdown('<h2 class="subtitle">Chat</h2>', unsafe_allow_html=True)

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
                        st.markdown(f"**Source:** {source['metadata'].get('source', 'Unknown')}")
                        st.markdown(f"**Content:** {source['content'][:200]}...")
                        st.markdown("---")
    except Exception as e:
        st.error(f"Error getting response: {str(e)}") 
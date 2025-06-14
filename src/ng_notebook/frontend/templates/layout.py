import streamlit as st
from pathlib import Path

def load_css():
    """Load the CSS file."""
    css_file = Path(__file__).parent.parent / "static" / "css" / "style.css"
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def init_page_config():
    """Initialize the page configuration."""
    st.set_page_config(
        page_title="Neogenesis Notebook",
        page_icon="📚",
        layout="wide"
    )

def render_header():
    """Render the page header."""
    st.title("Neogenesis Notebook")
    st.markdown("Your AI-powered document analysis assistant")

def render_sidebar(upload_file, on_upload):
    """Render the sidebar with upload section."""
    with st.sidebar:
        st.header("Upload Documents")
        with st.container():
            st.markdown('<div class="upload-container">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["pdf", "xlsx", "pptx"],
                help="Upload PDF, Excel, or PowerPoint files"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_file:
                with st.spinner("Processing document..."):
                    result = on_upload(uploaded_file)
                    if result:
                        st.success(f"Processed {result['chunks']} chunks")

def render_chat_interface(chat_history, on_query):
    """Render the chat interface."""
    st.header("Chat with Your Documents")
    
    # Display chat history
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = on_query(prompt, chat_history)
                if response:
                    st.write(response["answer"])
                    chat_history.append({
                        "role": "assistant",
                        "content": response["answer"]
                    }) 
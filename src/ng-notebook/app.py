import streamlit as st
import requests

# Configure Streamlit page
st.set_page_config(
    page_title="Document Chat",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and description
st.title("Document Chat with Llama 3.3")
st.markdown("""
This application allows you to chat with your documents using Llama 3.3.
Upload PDF, Excel, CSV, or PowerPoint files to get started.
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
        response = requests.get("http://localhost:8000/collections")
        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                st.error(data["error"])
            else:
                with st.expander("Collection Information", expanded=True):
                    st.write(f"**Collection Name:** {data['collection_name']}")
                    st.write(f"**Total Documents:** {data['total_documents']}")
                    
                    if data['unique_sources']:
                        st.write("**Uploaded Files:**")
                        for source in data['unique_sources']:
                            st.write(f"- {source}")
                    
                    if data['document_types']:
                        st.write("**Document Types:**")
                        for doc_type in data['document_types']:
                            st.write(f"- {doc_type}")
        else:
            st.error("Error fetching collection information. Please try again.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload a document",
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
st.markdown("---")
st.subheader("Chat")

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
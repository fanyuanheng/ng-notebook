def get_title_markdown() -> str:
    """Returns the title markdown."""
    return '<h1 class="title">Neogenesis Notebook</h1>'

def get_subtitle_markdown() -> str:
    """Returns the subtitle markdown."""
    return '<h2 class="subtitle">AI-Powered Document Analysis Platform</h2>'

def get_description_markdown() -> str:
    """Returns the description markdown."""
    return """
Neogenesis Notebook is an advanced document analysis platform that combines cutting-edge AI technologies to help you understand and interact with your documents.

Get started by uploading your documents and asking questions about their content.
"""

def get_upload_section_markdown() -> str:
    """Returns the upload section markdown."""
    return '<h2 class="subtitle">Upload Document</h2>'

def get_chat_section_markdown() -> str:
    """Returns the chat section markdown."""
    return '<h2 class="subtitle">Chat</h2>'

def get_source_markdown(source: dict) -> str:
    """Returns the source document markdown."""
    return f"""
**Source:** {source['metadata'].get('source', 'Unknown')}
**Content:** {source['content'][:200]}...
---
""" 
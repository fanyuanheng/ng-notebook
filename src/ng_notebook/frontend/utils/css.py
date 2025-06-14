import streamlit as st
from pathlib import Path

def load_css() -> None:
    """Load and apply custom CSS styles from the frontend static directory."""
    css_path = Path(__file__).parent.parent / "static" / "css" / "styles.css"
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True) 
import streamlit.web.cli as stcli
import sys
import os
from pathlib import Path

def main():
    # Get the path to the frontend app
    frontend_path = Path(__file__).parent / "frontend" / "app.py"
    
    # Set up sys.argv for streamlit
    sys.argv = [
        "streamlit",
        "run",
        str(frontend_path),
        "--server.port=8501",
        "--server.address=localhost"
    ]
    
    # Run the Streamlit app
    sys.exit(stcli.main())

if __name__ == "__main__":
    main() 
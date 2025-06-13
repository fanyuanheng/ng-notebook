# Neogenesis Notebook: Document Chat with LangChain & Llama 3.3

This project is a web-based Retrieval-Augmented Generation (RAG) chat application. It allows you to upload documents (PDF, Excel, CSV, PowerPoint, and text files) and chat with them using a local Llama 3.3 model via Ollama. The backend is powered by FastAPI, and the frontend uses Streamlit.

---

## Features
- **Document Upload:** PDF, Excel, CSV, PowerPoint, and text files
- **RAG Chat:** Ask questions about your documents
- **Local LLM:** Uses Llama 3.3 via Ollama
- **Web UI:** Simple chat interface with source viewing

---

## Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com/) installed and running locally
- Llama 3.3 model pulled in Ollama (`ollama pull llama3:3.3`)
- [Homebrew](https://brew.sh/) (for macOS users)
- `libmagic` system library (for file type detection)

### Install libmagic (macOS)
```sh
brew install libmagic
```

---

## Setup
1. **Clone the repository and enter the project directory:**
   ```sh
   git clone <your-repo-url>
   cd jupyter-cursor-project
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -e .
   ```

---

## Running the Application

### 1. Start Ollama with Llama 3.3
Make sure Ollama is running and the Llama 3.3 model is available:
```sh
ollama run llama3:3.3
```

### 2. Start the Backend (FastAPI)
```sh
uvicorn ng-notebook.main:app --reload
```
The backend will be available at [http://localhost:8000](http://localhost:8000)

### 3. Start the Frontend (Streamlit)
```sh
streamlit run src/ng-notebook/app.py
```
The frontend will open in your browser (usually at [http://localhost:8501](http://localhost:8501))

---

## Usage
1. **Upload a document** (PDF, Excel, CSV, PowerPoint, or text file) in the web UI.
2. **Ask questions** about the content of your document.
3. **View sources** for each answer in the chat.

---

## Troubleshooting
- If you see `ImportError: failed to find libmagic. Check your installation`, make sure you have installed `libmagic` and, if needed, set the `MAGIC_LIBRARY` environment variable:
  ```sh
  export MAGIC_LIBRARY=/opt/homebrew/lib/libmagic.dylib
  ```
- Ensure Ollama is running and the Llama 3.3 model is available.

---

## License
MIT 
# Legal Document RAG Assistant

A Python-based legal document chatbot using RAG (Retrieval Augmented Generation) to provide easy-to-understand explanations of legal documents.

## Features
- IPC (Indian Penal Code) mode with persistent document storage
- Custom PDF document support with document caching
- Conversational memory maintains context across questions
- Uses Llama3-70B through Groq API for high-quality responses
- ChromaDB for efficient document storage and retrieval
- Streamlit-based user interface

## Setup
```bash
# Create virtual environment
python -m venv env
source env/bin/activate

# Install dependencies 
pip install -r requirements.txt

# Add GROQ API key to .streamlit/secrets.toml
# Place IPC_pdf.pdf in project root
streamlit run app.py
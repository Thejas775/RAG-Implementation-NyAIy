import streamlit as st
import os
import asyncio
from typing import List, Tuple
from law_rag import LawDocumentRAG
import tempfile
import chromadb

class LawChatbot:
    def __init__(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        if 'current_mode' not in st.session_state:
            st.session_state.current_mode = "IPC"
        if 'last_uploaded_file' not in st.session_state:
            st.session_state.last_uploaded_file = None

    def initialize_rag_system(self):
        if not st.session_state.rag_system:
            api_key = st.secrets["GROQ_API_KEY"]
            st.session_state.rag_system = LawDocumentRAG(groq_api_key=api_key)

    def save_temp_file(self, uploaded_file) -> str:
        if uploaded_file == st.session_state.last_uploaded_file:
            return None
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.last_uploaded_file = uploaded_file
            return tmp_file.name

    def load_document(self, file_path: str, mode: str) -> str:
        try:
            if mode == "IPC":
                return "IPC document ready"
            else:
                return st.session_state.rag_system.load_custom_document(file_path)
        except Exception as e:
            return f"Error loading document: {str(e)}"

    def get_chat_history(self):
        return st.session_state.messages

    def get_chatbot_response(self, user_input: str) -> str:
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            chat_history = self.get_chat_history()
            
            if st.session_state.current_mode == "IPC":
                response = loop.run_until_complete(
                    st.session_state.rag_system.query_ipc(user_input, chat_history)
                )
            else:
                response = loop.run_until_complete(
                    st.session_state.rag_system.query_custom_document(user_input, chat_history)
                )
            return response
        except Exception as e:
            return f"Error getting response: {str(e)}"

    def create_sidebar(self):
        with st.sidebar:
            st.title("📚 Law Document Chat")
            mode = st.radio("Select Mode:", ["IPC", "Custom Document"])
            
            if mode != st.session_state.current_mode:
                st.session_state.current_mode = mode
                st.session_state.messages = []
            
            if mode == "Custom Document":
                uploaded_file = st.file_uploader(
                    "Upload PDF Document", 
                    type="pdf",
                    help="Upload custom legal document in PDF format"
                )
                
                if uploaded_file:
                    temp_file_path = self.save_temp_file(uploaded_file)
                    if temp_file_path:
                        with st.spinner("Loading document..."):
                            result = self.load_document(temp_file_path, mode)
                            st.success(result)
                            os.unlink(temp_file_path)

    def create_chat_interface(self):
        st.title("🤖 Legal Assistant")
        
        mode_text = "Indian Penal Code Mode" if st.session_state.current_mode == "IPC" else "Custom Document Mode"
        st.markdown(f"**Current Mode:** {mode_text}")
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask your question here..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self.get_chatbot_response(prompt)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    def add_sidebar_info(self):
        with st.sidebar:
            st.markdown("---")
            if st.session_state.current_mode == "IPC":
                st.markdown("""
                ### IPC Mode
                Currently using the Indian Penal Code as reference.
                Ask questions about:
                - Specific sections
                - Legal definitions
                - Penalties and punishments
                - Legal procedures
                """)
            else:
                st.markdown("""
                ### Custom Document Mode
                Upload any legal document to get assistance with:
                - Understanding complex legal text
                - Extracting key information
                - Clarifying legal terms
                - Summarizing sections
                """)

    def run(self):
        st.set_page_config(
            page_title="Legal Document Assistant",
            page_icon="⚖️",
            layout="wide"
        )
        self.initialize_rag_system()
        self.create_sidebar()
        self.create_chat_interface()
        self.add_sidebar_info()

if __name__ == "__main__":
    chatbot = LawChatbot()
    chatbot.run()
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import chromadb
import os
import hashlib

class LawDocumentRAG:
    def __init__(self, groq_api_key):
        os.environ["GROQ_API_KEY"] = groq_api_key
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.llm = ChatGroq(model="llama3-70b-8192", temperature=0.1)
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        self.ipc_vectorstore = None
        self.custom_vectorstore = None
        self.current_custom_doc_hash = None
        self.load_default_ipc()
        
        self.LEGAL_RAG_SYSTEM_PROMPT = """You are a helpful legal assistant that explains legal concepts in simple, easy-to-understand language. 
        Use the following pieces of information to answer the human's questions:

        1. Retrieved Context (relevant legal text):
        ```
        {context}
        ```

        2. Conversation History:
        ```
        {chat_history}
        ```

        Current Question: {input}

        When answering:
        1. Use the context and chat history to provide comprehensive answers
        2. Maintain consistency with previous responses
        3. If referring to previous discussion, be explicit about it
        4. Avoid using complex legal jargon and explain any technical terms
        5. If you don't know something based on the provided context, say so
        """
        
        self.RAG_HUMAN_PROMPT = "{input}"
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", self.LEGAL_RAG_SYSTEM_PROMPT),
            ("human", self.RAG_HUMAN_PROMPT)
        ])

    def load_default_ipc(self):
        try:
            try:
                self.ipc_vectorstore = Chroma(
                    client=self.chroma_client,
                    collection_name="ipc_law",
                    embedding_function=self.embed_model
                )
            except ValueError:
                ipc_path = "IPC_pdf.pdf"
                if os.path.exists(ipc_path):
                    self.load_ipc_document(ipc_path)
        except Exception as e:
            print(f"Error loading default IPC document: {str(e)}")

    def get_file_hash(self, file_path):
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def format_chat_history(self, messages):
        formatted_history = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted_history.append(f"{role.capitalize()}: {content}")
        return "\n".join(formatted_history)

    def load_ipc_document(self, ipc_pdf_path):
        try:
            if self.ipc_vectorstore is None:
                loader = PyPDFLoader(ipc_pdf_path)
                pages = loader.load()
                texts = self.text_splitter.split_documents(pages)
                self.ipc_vectorstore = Chroma.from_documents(
                    documents=texts,
                    embedding=self.embed_model,
                    collection_name="ipc_law",
                    client=self.chroma_client
                )
            return "IPC document loaded and indexed successfully"
        except Exception as e:
            return f"Error loading IPC document: {str(e)}"

    def load_custom_document(self, pdf_path):
        try:
            new_doc_hash = self.get_file_hash(pdf_path)
            if new_doc_hash == self.current_custom_doc_hash and self.custom_vectorstore is not None:
                return "Document already loaded"
            
            try:
                self.chroma_client.delete_collection("custom_doc")
            except ValueError:
                pass
            
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            texts = self.text_splitter.split_documents(pages)
            self.custom_vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embed_model,
                collection_name="custom_doc",
                client=self.chroma_client
            )
            self.current_custom_doc_hash = new_doc_hash
            return "Custom document loaded and indexed successfully"
        except Exception as e:
            return f"Error loading custom document: {str(e)}"

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def create_rag_chain(self, vectorstore, chat_history):
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        rag_chain = (
            {
                "context": retriever | self.format_docs,
                "chat_history": lambda x: self.format_chat_history(chat_history),
                "input": RunnablePassthrough()
            }
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    async def query_ipc(self, question: str, chat_history: list) -> str:
        if self.ipc_vectorstore is None:
            return "Error: IPC document not properly loaded"
        rag_chain = self.create_rag_chain(self.ipc_vectorstore, chat_history)
        response = await rag_chain.ainvoke(question)
        return response

    async def query_custom_document(self, question: str, chat_history: list) -> str:
        if self.custom_vectorstore is None:
            return "Please load a custom document first"
        rag_chain = self.create_rag_chain(self.custom_vectorstore, chat_history)
        response = await rag_chain.ainvoke(question)
        return response

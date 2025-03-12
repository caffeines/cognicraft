import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
import openai

print("Loading environment variables...")
load_dotenv()
hf_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
)

st.title("GROQ - RAG Conversation with Chat History")
st.write("This is a Streamlit app that uses the GROQ model to answer questions based on a chat history.")
st.sidebar.title("Settings")
uploaded_files = st.sidebar.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

documents = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        documents.append(PyPDFDirectoryLoader(uploaded_file))
        temp_pdf = f"./temp/{uploaded_file.name}"
        with open(temp_pdf, "wb") as f:
            f.write(uploaded_file.getvalue())
        documents.append(PyPDFDirectoryLoader(temp_pdf))


session_id = st.sidebar.text_input("Session ID", "default-session")

if "store" not in st.session_state:
    st.session_state.store = {}
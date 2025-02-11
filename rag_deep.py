import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Constants
PDF_STORAGE_PATH = "document_store/pdfs/"
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")
VECTOR_DB = None  # Global FAISS instance

PROMPT_TEMPLATE = """
You are an expert in Anti-Money Laundering (AML) regulations as per RBI guidelines. 
Your task is to provide users with a detailed yet concise step-by-step guide on AML processes, explaining key compliance requirements and best practices. 
Each step should be clear, precise, and factual, offering a thorough understanding while maintaining brevity (max 5 sentences per step). If any information is uncertain, 
explicitly state that you don't know instead of making assumptions. Focus on summarizing each step with clarity while ensuring users grasp the full scope of AML compliance.
Query: {user_query} 
Context: {document_context} 
Answer:-
"""

def save_uploaded_file(uploaded_file):
    """Saves the uploaded file and returns the file path."""
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def process_pdf(file_path):
    """Processes the uploaded PDF synchronously and indexes the content into FAISS."""
    global VECTOR_DB  # Use global FAISS instance

    # Load document
    document_loader = PDFPlumberLoader(file_path)
    raw_docs = document_loader.load()

    # Efficient text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    document_chunks = text_splitter.split_documents(raw_docs)

    # Create FAISS index with embeddings
    texts = [doc.page_content for doc in document_chunks]
    VECTOR_DB = FAISS.from_texts(texts, EMBEDDING_MODEL)

def find_related_documents(query):
    """Finds documents related to the query using FAISS."""
    return VECTOR_DB.similarity_search(query) if VECTOR_DB else []

def generate_answer(user_query, context_documents):
    """Generates an AI response based on the retrieved documents."""
    context_text = "\n\n".join(doc.page_content for doc in context_documents)
    response_chain = ChatPromptTemplate.from_template(PROMPT_TEMPLATE) | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

import streamlit as st

st.title("DeepSeek R1 DEMO")
st.markdown("### Your Intelligent Document Guide Assistant ")
st.markdown("---")

uploaded_pdf = st.file_uploader("Upload Research Document (PDF)", type="pdf", help="Select a PDF document for analysis")

# Use session state to track processed document
if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.saved_path = None

if uploaded_pdf and not st.session_state.processed:
    st.session_state.saved_path = save_uploaded_file(uploaded_pdf)
    
    with st.spinner("Processing document... 🚀"):
        process_pdf(st.session_state.saved_path)

    st.session_state.processed = True
    st.success("✅ Document processed successfully! Ask your questions below.")

# Display success message if already processed
if st.session_state.processed:
    st.success("✅ Document processed successfully! Ask your questions below.")

    user_input = st.chat_input("Enter your question about the document...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)

        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)

import os
import pdfplumber
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_manger.rag import Rag
from utils.utils import read_sample_pdf, read_uploaded_pdf

class PDFHandler:
    """Handles PDF processing and vector database creation."""

    def __init__(self, rag: Rag):
        """
        Initialize the PDF handler with the selected language model and embeddings model.

        Args:
            llm_model (str): The name of the selected language model.
            embeddings_model (str): The name of the embeddings model.
        """
        self.rag = rag


    def load_sample_pdf(self, sample_path):
        """Loads and processes a sample PDF."""
        if os.path.exists(sample_path):
            with st.spinner("Processing sample PDF..."):
                data = read_sample_pdf(sample_path)
                self.vector_db = self.rag.create_vector_db(data)

        
        else:
            st.error("Sample PDF file not found in the current directory.")
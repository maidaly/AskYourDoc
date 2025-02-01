import os
import subprocess
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from utils.logging_utils import logger
from langchain_community.document_loaders import UnstructuredWordDocumentLoader


def read_uploaded_docx(file_upload) -> str:
    logger.info(f"Processing file: {file_upload.name}")
    # Save file temporarily
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        loader = UnstructuredWordDocumentLoader(path)
        data = loader.load()
        logger.info(f"Loaded {len(data)} documents from {file_upload.name}")
    return data

def read_uploaded_pdf(file_upload) -> str:
    """
    Read the content of a PDF file.

    Args:
        file_path: The path to the PDF file.

    Returns:
        The content of the PDF file.
    """
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        loader = PyPDFLoader(path)
        data = loader.load()
    return data

def read_sample_pdf(sample_path:str) -> str:
    """
    Read the content of a sample PDF file.

    Args:
        sample_path: The path to the sample PDF file.

    Returns:
        The content of the sample PDF file.
    """
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            loader = PyPDFLoader(f)
            data = loader.load()
            return data
    else:
        st.error("Sample PDF file not found in the current directory.")

def get_ollama_models():
    """
    Get a list of Ollama models.

    Returns:
        A list of Ollama models.
    """
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        models = [line.split()[0] for line in result.stdout.split("\n") if line]
        return models
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return []
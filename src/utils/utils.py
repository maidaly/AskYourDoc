import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from utils.logging_utils import logger

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

def extract_model_names(models_info):
    """
    Extract the names of the available models.

    Args:
        models_info: The list of available models.

    Returns:
        The names of the available models.
    """
    return tuple(item.model for item in models_info if hasattr(item, 'model'))
"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.

This application allows users to upload a PDF, process it,
and then ask questions about the content using a selected language model.
"""

from dataclasses import asdict
import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from streamlit.runtime.uploaded_file_manager import UploadedFile

import chromadb.api

# chroma_client = chromadb.Client()
# collection = chroma_client.create_collection(name="myRAG")

# Set protobuf environment variable to avoid error messages
# This might cause some issues with latency but it's a tradeoff
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# @st.cache_resource(show_spinner=True)
def extract_model_names(models_info):
    # if models_info[0]:
    #     print(type(models_info[0]))
        # models_dicts = [asdict(model) for model in models_info]
        # print(models_dicts)
        return tuple(models.model for models in models_info)
    
def read_files_and_split(file_uploads: list[UploadedFile]) -> list:
    """
    Read multiple files, extract their text, and split them into chunks.

    Args:
        file_uploads (list[st.UploadedFile]): List of Streamlit file upload objects containing PDFs.

    Returns:
        list: A list of document chunks.
    """
    logger.info("Reading files and splitting into chunks")
    temp_dir = tempfile.mkdtemp()
    all_chunks = []

    try:
        for file_upload in file_uploads:
            logger.info(f"Processing file: {file_upload.name}")
            # Save file temporarily
            path = os.path.join(temp_dir, file_upload.name)
            with open(path, "wb") as f:
                f.write(file_upload.getvalue())
                logger.info(f"File saved to temporary path: {path}")
                
                if path.endswith(".pdf"):
                    # Load the PDF
                    loader = PyPDFLoader(path)
                elif path.endswith(".docx"):
                    loader = UnstructuredWordDocumentLoader(path)
                data = loader.load()
                logger.info(f"Loaded {len(data)} documents from {file_upload.name}")

                # Split the text into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
                chunks = text_splitter.split_documents(data)
                all_chunks.extend(chunks)  # Accumulate chunks

        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir)
        logger.info(f"Temporary directory {temp_dir} removed")


def create_vector_store(chunks: list, collection_name: str = "myRAG", persist_directory: str = "./app") -> Chroma:
    """
    Create a vector store from document chunks.

    Args:
        chunks (list): List of document chunks.
        collection_name (str): Name of the vector store collection.
        persist_directory (str): Path to save the vector store.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info("Creating vector store")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    logger.info("Vector store created successfully")
    return vector_db


def create_vector_db(file_uploads: list[UploadedFile]) -> Chroma:
    """
    Create a vector database from multiple uploaded PDF files.

    Args:
        file_uploads (list[st.UploadedFile]): List of Streamlit file upload objects containing PDFs.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info("Creating vector DB from file uploads")
    
    # Step 1: Read files and split them into chunks
    chunks = read_files_and_split(file_uploads)

    # Step 2: Create the vector store from chunks
    vector_db = create_vector_store(chunks)

    return vector_db



def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """
    Process a user question using the vector database and selected language model.

    Args:
        question (str): The user's question.
        vector_db (Chroma): The vector database containing document embeddings.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    logger.info(f"Processing question: {question} using model: {selected_model}")
    
    # Initialize LLM
    llm = ChatOllama(model=selected_model)
    
    # Query prompt template
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 2
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    # Set up retriever
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    # RAG prompt template
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Create chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages


def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.

    Args:
        vector_db (Optional[Chroma]): The vector database to be deleted.
    """
    logger.info("Deleting vector DB")
    if vector_db is not None:
        vector_db.delete_collection()
        st.session_state.pop("pdf_pages", None)
        st.session_state.pop("file_upload", None)
        st.session_state.pop("vector_db", None)
        st.success("Collection and temporary files deleted successfully.")
        logger.info("Vector DB and related session state cleared")
        st.rerun()
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")


def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    st.subheader("🧠 Ollama PDF RAG playground", divider="gray", anchor=False)

    # Get available models
    models_info = ollama.list()
    # available_models = extract_model_names(models_info)
    # print(available_models)
    print(models_info)
    # Create layout
    col1, col2 = st.columns([1.5, 2])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "use_sample" not in st.session_state:
        st.session_state["use_sample"] = False

    # # Model selection
    # if available_models:
    #     selected_model = col2.selectbox(
    #         "Pick a model available locally on your system ↓", 
    #         available_models,
    #         key="model_select"
    #     )

    # Add checkbox for sample PDF
    use_sample = col1.toggle(
        "Use sample PDF (Scammer Agent Paper)", 
        key="sample_checkbox"
    )
    
    # Clear vector DB if switching between sample and upload
    if use_sample != st.session_state.get("use_sample"):
        if st.session_state["vector_db"] is not None:
            st.session_state["vector_db"].delete_collection()
            st.session_state["vector_db"] = None
            st.session_state["pdf_pages"] = None
        st.session_state["use_sample"] = use_sample

    if use_sample:
        # Use the sample PDF
        sample_path = "scammer-agent.pdf"
        if os.path.exists(sample_path):
            if st.session_state["vector_db"] is None:
                with st.spinner("Processing sample PDF..."):
                    loader = UnstructuredPDFLoader(file_path=sample_path)
                    data = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
                    chunks = text_splitter.split_documents(data)
                    st.session_state["vector_db"] = Chroma.from_documents(
                        documents=chunks,
                        embedding=OllamaEmbeddings(model="nomic-embed-text"),
                        collection_name="myRAG"
                    )
                    # Open and display the sample PDF
                    with pdfplumber.open(sample_path) as pdf:
                        pdf_pages = [page.to_image().original for page in pdf.pages]
                        st.session_state["pdf_pages"] = pdf_pages
        else:
            st.error("Sample PDF file not found in the current directory.")
    else:
        # Regular file upload with unique key
        file_uploads = col1.file_uploader(
            "Upload PDF files ↓", 
            type=["pdf","docx"], 
            accept_multiple_files=True,  # Allow multiple file uploads
            key="pdf_uploader"
        )

        if file_uploads:  # Check if files were uploaded
            if st.session_state.get("vector_db") is None:  # Check if the vector DB is already created
                with st.spinner("Processing uploaded PDFs..."):
                    # Process multiple files to create the vector DB
                    st.session_state["vector_db"] = create_vector_db(file_uploads)
                    st.success("Vector database created successfully!")


    # Display PDF if pages are available
    # if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
    #     # PDF display controls
    #     zoom_level = col1.slider(
    #         "Zoom Level", 
    #         min_value=100, 
    #         max_value=1000, 
    #         value=700, 
    #         step=50,
    #         key="zoom_slider"
    #     )

        # # Display PDF pages
        # with col1:
        #     with st.container(height=410, border=True):
        #         # Removed the key parameter from st.image()
        #         for page_image in st.session_state["pdf_pages"]:
        #             st.image(page_image, width=zoom_level)

    # Delete collection button
    delete_collection = col1.button(
        "⚠️ Delete collection", 
        type="secondary",
        key="delete_button"
    )

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    # Chat interface
    with col2:
        message_container = st.container(height=500, border=True)

        # Display chat history
        for i, message in enumerate(st.session_state["messages"]):
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Chat input and processing
        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            try:
                # Add user message to chat
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)

                # Process and display assistant response
                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], "llama3.2"
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a PDF file first.")

                # Add assistant response to chat history
                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file or use the sample PDF to begin chat...")


if __name__ == "__main__":
    main()

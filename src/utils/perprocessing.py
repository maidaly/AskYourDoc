import os
import tempfile
import shutil
from pdfplumber import open as pdf_open
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader


def extract_all_pages_as_images(file_upload_or_path):
    if not os.path.exists(file_upload_or_path):
        print(f"Error: File '{file_upload_or_path}' does not exist.")
        return None

    with pdf_open(file_upload_or_path) as pdf:
        pages = [page.to_image().original for page in pdf.pages]
        print(f"Extracted {len(pages)} pages as images.")
        return pages


def create_vector_db(file_upload_or_path):
    if not os.path.exists(file_upload_or_path):
        print(f"Error: File '{file_upload_or_path}' does not exist.")
        return None

    temp_dir = tempfile.mkdtemp()
    try:
        loader = PyPDFLoader(file_upload_or_path)
        print("Loading PDF and splitting into chunks...")
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s) from PDF.")
        
        chunks = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100).split_documents(documents)
        print(f"Created {len(chunks)} chunks from the documents.")

        print("Creating vector database...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
            collection_name="myRAG",
            persist_directory="./chroma_db",  # Path to save the database
        )
        print("Vector database created successfully.")
        vector_db.persist()  # Save the database to disk
        return vector_db
    finally:
        shutil.rmtree(temp_dir)
        print(f"Temporary directory '{temp_dir}' deleted.")


if __name__ == "__main__":
    file_path = r"R:\\4.Self-Study\\Missing-Children-Family-Reunion-using-Face-Recognition.pdf"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file at path '{file_path}' does not exist.")
    else:
        # Extract images
        print("Extracting images from PDF...")
        images = extract_all_pages_as_images(file_path)
        if images:
            print(f"Extracted {len(images)} pages as images.")
        
        # Create vector database
        print("Creating vector database...")
        vector_db = create_vector_db(file_upload_or_path=file_path)
        
        if vector_db:
            print("Vector database details:")
            print(f"Collection Name: {vector_db._collection.name}")
            print(f"Number of Vectors: {vector_db._collection.count()}")
        else:
            print("Failed to create vector database.")

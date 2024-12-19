import tempfile
import shutil
from pdfplumber import open as pdf_open
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings



def extract_all_pages_as_images(file_upload_or_path):
    with pdf_open(file_upload_or_path) as pdf:
        return [page.to_image().original for page in pdf.pages]

def create_vector_db(file_upload_or_path):
    temp_dir = tempfile.mkdtemp()
    loader = UnstructuredPDFLoader(file_upload_or_path)
    chunks = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100).split_documents(loader.load())

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        collection_name="myRAG",
    )
    shutil.rmtree(temp_dir)
    return vector_db

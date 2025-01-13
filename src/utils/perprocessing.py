import os
import tempfile
import shutil
from pdfplumber import open as pdf_open
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from logging_utils import logger


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
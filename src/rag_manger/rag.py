
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logging_utils import logger
import streamlit as st

class Rag:
    def __init__(self, llm_model: str, embeddings_model: str):
        """
        Initialize the RAG manager with a vector database and selected language model.
        
        Args:
            vector_db (Chroma): The vector database containing document embeddings.
            selected_model (str): The name of the selected language model.
            embeddings_model (str): The name of the embeddings model.
        """
        self.llm = ChatOllama(model=llm_model)
        self.embeddings_model = OllamaEmbeddings(model=embeddings_model)
        self.vector_db = None
        logger.info("RAG manager initialized")
    
    def create_chuncks(self, data) -> list:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        logger.info("Document split into chunks")
        return chunks
    
    def create_vector_db(self, data) -> Chroma:
        """
        Create a vector database from an uploaded PDF file.

        Args:
            data (str): The content of the PDF file.

        Returns:
            Chroma: A vector store containing the processed document chunks.
        """
        logger.info("Creating vector DB from file upload")
        chunks = self.create_chuncks(data)
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings_model,
            collection_name="myRAG",
            persist_directory="./chroma_db"
        )
        logger.info("Vector DB created")
        return self.vector_db
    
    def run(self, question: str, vector_db:Chroma) -> str:

        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate 2
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )
        self.retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), 
            self.llm,
            prompt=QUERY_PROMPT
        )

        logger.info(f"Processing question: {question} using model: {self.llm}")

        # RAG prompt template
        template = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        # Create chain
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(question)
        logger.info("Question processed and response generated")
        return response
    
    def delete_vector_db(self) -> None:
        """
        Delete the vector database and clear related session state.

        Args:
            vector_db (Optional[Chroma]): The vector database to be deleted.
        """
        logger.info("Deleting vector DB")
        if self.vector_db is not None:
            self.vector_db.delete_collection()
            st.session_state.pop("pdf_pages", None)
            st.session_state.pop("file_upload", None)
            st.session_state.pop("vector_db", None)
            st.success("Collection and temporary files deleted successfully.")
            logger.info("Vector DB and related session state cleared")
            st.rerun()
        else:
            st.error("No vector database found to delete.")
            logger.warning("Attempted to delete vector DB, but none was found")
    

import os
import sys
import streamlit as st
import ollama
from config import PAGE_CONFIG
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
import chromadb.api
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.rag_manger.rag import Rag
from src.utils.utils import *
from src.utils.logging_utils import logger
from src.streamlit_manger.pdf_handler import PDFHandler
from src.streamlit_manger.session_handler import SessionStateManager
# from langchain.llms import Ollama

# Streamlit page configuration
st.set_page_config(**PAGE_CONFIG)
chromadb.api.client.SharedSystemClient.clear_system_cache()
rag = Rag(llm_model="llama3.2", embeddings_model="nomic-embed-text")
pdf_handler = PDFHandler(rag)

def main():
    """
    Main function to run the Streamlit application.
    """

    st.subheader("📖 Ask your Document", divider="gray", anchor=False)
    models_info = ollama.list()
    print(models_info)
    # Create layout
    col1, col2 = st.columns([1.5, 2])
    session = SessionStateManager()
    session.initialize_state()
    # Model selection
    # selected_model = col2.selectbox("Select a model", available_models) if available_models else None

    # File upload and sample usage toggle
    use_sample = col1.toggle(
        "Use sample PDF (Scammer Agent Paper)", 
        key="sample_checkbox"
    )

    if use_sample != session.get("use_sample"):
        if session.get("vector_db") is not None:
            session.get("vector_db").delete_collection()
            session.set("vector_db", None)
            session.set("pdf_pages", None)
        session.set("use_sample", use_sample)
    
    if use_sample:
        # Use the sample PDF
        sample_path = "scammer-agent.pdf"
        if os.path.exists(sample_path):
            if session.get("vector_db") is None:
                with st.spinner("Processing sample PDF..."):
                    data = read_sample_pdf(sample_path)
                    session.set("vector_db", rag.create_vector_db(data))
    else:
        # Regular file upload with unique key
        file_upload = col1.file_uploader(
            "Upload a file ↓", 
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        if file_upload:
            print(file_upload)
            file_name = file_upload[0].name
            if st.session_state["vector_db"] is None:
                with st.spinner("Processing uploaded document..."):
                    if file_name.endswith(".pdf"):
                        data = read_uploaded_pdf(file_upload[0])
                    elif file_name.endswith(".docx"):
                        data = read_uploaded_docx(file_upload[0])
                    vector_db=rag.create_vector_db(data)
                    session.set("vector_db", vector_db)
        # Delete collection button
    delete_collection = col1.button(
        "⚠️ Delete collection", 
        type="secondary",
        key="delete_button"
    )
    if delete_collection:
        rag.delete_vector_db()
        session.clear_all()


    # # Display PDF
    # if "pdf_pages" in st.session_state:
    #     col1.image(st.session_state["pdf_pages"])

    # Chat interface
    with col2:
        message_container = st.container(height=500, border=True)

        # Display chat history
        for i, message in enumerate(st.session_state["messages"]):
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            try:
                # Add user message to chat
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="😎"):
                    st.markdown(prompt)
                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if session.get("vector_db") is not None:
                            vector_db = session.get("vector_db")
                            response = rag.run(prompt, vector_db)
                            st.markdown(response)
                        else:
                            st.warning("Please upload a PDF file to continue.")
                        if session.get("vector_db") is not None:
                            session.get("messages").append(
                                {"role": "assistant", "content": response}
                            )
            except Exception as e:
                logger.error(e)
                st.error("An error occurred. Please try again.")
            
            else:
                if session.get("vector_db") is None:
                    st.warning("Upload a PDF file or use the sample PDF to begin chat...")
                    

if __name__ == "__main__":
    main()

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


def main():
    """
    Main function to run the Streamlit application.
    """

    st.subheader("📖 Ask your Document", divider="gray", anchor=False)
    # Create layout
    col1, col2 = st.columns([1.5, 2])
    session = SessionStateManager()
    session.initialize_state()
    models = get_ollama_models()[1:]
    selected_model = None
    if models:
        session.set("available_models", models)

    # Create layout
    with col1:
        if st.session_state.get("available_models"):
            available_models = session.get("available_models")
            selected_model = st.selectbox("Select a model", available_models)
            if selected_model:
                session.set("selected_model", selected_model)

        else:
            st.error("No models found")

        # File upload and sample usage toggle
        file_upload = col1.file_uploader(
            "Upload a file ↓",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="file_uploader",
        )
        if session.get("selected_model"):
            selected_model = session.get("selected_model")
            rag = Rag(selected_model, embeddings_model="nomic-embed-text:latest")
            # Regular file upload with unique key
            if file_upload:
                file_name = file_upload[0].name
                if st.session_state["vector_db"] is None:
                    with st.spinner("Processing uploaded document..."):
                        if file_name.endswith(".pdf"):
                            data = read_uploaded_pdf(file_upload[0])
                        elif file_name.endswith(".docx"):
                            data = read_uploaded_docx(file_upload[0])
                        else:
                            st.error("Unsupported file type")
                        vector_db = rag.create_vector_db(data)
                        session.set("vector_db", vector_db)
            else:
                st.warning("Upload a PDF file to begin chat...")
                # Delete collection button
            delete_collection = col1.button(
                "⚠️ Delete collection", type="secondary", key="delete_button"
            )
            if delete_collection:
                rag.delete_vector_db()
                session.clear_all()

    # Chat interface
    with col2:
        st.subheader("💬 Start Chatting .... ", divider="gray", anchor=False)
        message_container = st.container(height=500, border=True)

        # Display chat history
        for i, message in enumerate(st.session_state["messages"]):
            avatar = "🤖" if message["role"] == "assistant" else "⌨"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            try:
                # Add user message to chat
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container.chat_message("user", avatar="⌨"):
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
                    st.warning(
                        "Upload a PDF file or use the sample PDF to begin chat..."
                    )


if __name__ == "__main__":
    main()

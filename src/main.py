import streamlit as st
import ollama
from config import PAGE_CONFIG
from utils.perprocessing import extract_all_pages_as_images, create_vector_db
from utils.vector_db import delete_vector_db
from utils.model_utils import extract_model_names, process_question
from utils.logging_utils import logger
# from langchain.llms import Ollama

# Streamlit page configuration
st.set_page_config(**PAGE_CONFIG)

def main():
    """
    Main function to run the Streamlit application.
    """
    st.subheader("🧠 Ollama PDF RAG playground", divider="gray", anchor=False)
    # ollama_client = Ollama()
    # Load available models
    models_info = ollama.list()
    print(models_info)
    # models=[Model(model='llama3.2:latest', modified_at=datetime.datetime(2024, 12, 8, 0, 53, 9, 82336, tzinfo=TzInfo(+02:00)), 
    #               digest='a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72',
    #               size=2019393189, details=ModelDetails(parent_model='', format='gguf', 
    #                                                     family='llama', families=['llama'], parameter_size='3.2B', quantization_level='Q4_K_M'))]
    available_models = extract_model_names(models_info)

    # Layout setup
    col1, col2 = st.columns([1.5, 2])

    # Session state initialization
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("vector_db", None)
    st.session_state.setdefault("use_sample", False)

    # Model selection
    selected_model = col2.selectbox("Select a model", available_models) if available_models else None

    # File upload and sample usage toggle
    use_sample = col1.checkbox("Use sample PDF")
    file_upload = col1.file_uploader("Upload a PDF", type="pdf") if not use_sample else None

    if use_sample or file_upload:
        # Process sample or uploaded file
        file_name = "scammer-agent.pdf" if use_sample else file_upload.name
        print(file_upload.name)
        if st.session_state["vector_db"] is None:
            with st.spinner("Processing PDF..."):
                st.session_state["vector_db"] = create_vector_db(file_upload.name or file_name)
                st.session_state["pdf_pages"] = extract_all_pages_as_images(file_upload or file_name)

    # Display PDF
    if "pdf_pages" in st.session_state:
        col1.image(st.session_state["pdf_pages"])

    # Chat interface
    with col2:
        for message in st.session_state["messages"]:
            st.chat_message(message["role"]).markdown(message["content"])

        if prompt := st.chat_input("Ask a question..."):
            response = process_question(prompt, st.session_state["vector_db"], selected_model)
            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.markdown(response)

    # Clear vector database
    if col1.button("Delete vector DB"):
        delete_vector_db(st.session_state["vector_db"])

if __name__ == "__main__":
    main()

# AskYourDoc

# README: RAG Pipeline for Document Chat

## Introduction
This repository implements a Retrieval-Augmented Generation (RAG) pipeline to interact with documents through a Streamlit-based web application. The app leverages Ollama models for text generation and embedding creation, enabling effective retrieval and conversation with your documents.

## Prerequisites
Before using this application, ensure the following prerequisites are met:

1. **Ollama Installation**:
   - Install and configure [Ollama](https://ollama.ai/). It will be used for hosting the backend models.

2. **Models**:
   - Pull the required models using Ollama:
     - `llama3.2` for text generation.
     - `nomic-embed-text` for embedding creation.
     ```bash
     ollama pull llama3.2
     ollama pull nomic-embed-text
     ```

3. **Python**:
   - Python 3.11 or later installed.

4. **Required Libraries**:
   - All required Python libraries are listed in `requirements.txt` in the repository’s root directory.

## Installation
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify that Ollama is running and the required models are pulled.

## Running the Application
1. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Upload your documents and interact with them through the chat interface.

## Notes
- Ensure Ollama’s backend server is running and configured correctly by visting `http://localhost:11434` and have ```Ollama is running```message on the webpage.
- If you encounter any issues, check your Ollama and Python setup or refer to the logs for debugging.

Enjoy exploring and chatting with your documents!


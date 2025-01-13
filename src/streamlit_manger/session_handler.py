import streamlit as st
from typing import Any
from src.streamlit_manger.pdf_handler import PDFHandler
from src.rag_manger.rag import Rag

class SessionStateManager:
    def __init__(self):
        self.initialize_state()
        
    def initialize_state(self):
        """Initialize session state variables with default values."""
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        if "vector_db" not in st.session_state:
            st.session_state["vector_db"] = None
        if "rag" not in st.session_state:
            st.session_state["rag"] = None
        if "use_sample" not in st.session_state:
            st.session_state["use_sample"] = False
        if "pdf_pages" not in st.session_state:
            st.session_state["pdf_pages"] = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from session state."""
        return st.session_state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in session state."""
        st.session_state[key] = value

    def reset(self, key: str) -> None:
        """Reset a specific session state variable."""
        if key in st.session_state:
            del st.session_state[key]

    def clear_all(self):
        """Clear all session state variables."""
        st.session_state.clear()

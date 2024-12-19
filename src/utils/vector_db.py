import streamlit as st

def delete_vector_db(vector_db):
    if vector_db:
        vector_db.delete_collection()
        st.session_state.clear()
        st.success("Vector DB cleared!")

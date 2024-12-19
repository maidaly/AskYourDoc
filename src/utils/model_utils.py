from langchain.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama

def extract_model_names(models_info):
     return tuple(item.model for item in models_info if hasattr(item, 'model'))

def process_question(question, vector_db, selected_model):
    retriever = vector_db.as_retriever()
    llm = ChatOllama(model=selected_model)
    query_prompt = PromptTemplate(input_variables=["question"], template="Question: {question}")
    
    response = llm.chat_prompt(question, retriever, query_prompt)
    return response
